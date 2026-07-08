import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


from kvcache.zipcache import ZipCache
from eval_zipcache import EvalConfig, load_model_and_tokenizer

class DiagnosticZipCache(ZipCache):
    """
    Subclass of ZipCache with new window commit step 
    to calculate similarity metrics for non-salient value vectors.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store matrices for all windows. 
        # Each entry will be a numpy array of shape [heads, rest_seq, rest_seq]
        self.window_sim_matrices = []

    def _commit_window(self, layer_idx: int, window_keys: torch.Tensor, window_values: torch.Tensor):
        batch, heads, seq, head_dim = window_keys.shape
        top_k = max(1, int(seq * self.top_k_ratio))

        norm_keys, k_scale = self._suppress_keys(window_keys)
        norm_values, v_scale = self._suppress_values(window_values)

        saliency = self._compute_saliency(norm_keys)
        _, ranked = saliency.sort(descending=True)
        
        # the non salient tokens split, as the other commit window 
        rest_idx = ranked[top_k:]

        # grab the values from the first batch across ALL heads isolating our metrics 
        # the avg between all heads would have given a false reading
        # for a better accuracy we should have processed eahc head independently
        # for the inference part of the merging we should process each head independently as well as the merging
        
        #non_salient_values = norm_values[0, :, rest_idx, :] 
        non_salient_keys = norm_keys[0, :, rest_idx, :]
        # L2 Normalize the value vectors which is needed for cosine similarity
        
        #values_l2 = non_salient_values / non_salient_values.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        keys_l2 = non_salient_keys / non_salient_keys.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        
        # Batch matrix multiplication to get similarity matrices for all heads at once
        # Resulting shape: [heads, len(rest_idx), len(rest_idx)]
        
        #sim_matrix_heads = torch.bmm(values_l2, values_l2.transpose(1, 2))
        sim_matrix_heads = torch.bmm(keys_l2, keys_l2.transpose(1, 2))
        
        # Store the matrices for this window to plot later
        self.window_sim_matrices.append(sim_matrix_heads.float().cpu().numpy())

        # 3. Proceed with standard parent logic to keep cache functional
        # Doing manual super() assignment to match your exact caching strategy
        self.int8_cache.update(
            norm_keys[:, :, ranked[:top_k], :], norm_values[:, :, ranked[:top_k], :], layer_idx
        )
        self.int4_cache.update(
            norm_keys[:, :, rest_idx, :], norm_values[:, :, rest_idx, :], layer_idx
        )

        offset = self.committed_lengths.get(layer_idx, 0)
        self.global_salient_idx.setdefault(layer_idx, []).append(ranked[:top_k] + offset)
        self.global_rest_idx.setdefault(layer_idx, []).append(rest_idx + offset)
        self.window_k_scales.setdefault(layer_idx, []).append(k_scale)
        self.window_v_scales.setdefault(layer_idx, []).append(v_scale)
        self.committed_lengths[layer_idx] = offset + seq
def plot_diagnostics(task_name: str, cache: DiagnosticZipCache, output_dir: str):
    """Generates and saves Key similarity heatmap plots per window and head."""
    if not cache.window_sim_matrices:
        print(f"[{task_name}] No similarities recorded.")
        return

    sns.set_theme(style="whitegrid")
    
    # Retrieve the global indices for layer 0 to label the heatmap axes
    rest_indices_list = cache.global_rest_idx.get(0, [])
    
    for w_idx, window_matrices in enumerate(cache.window_sim_matrices[:4]):
        if w_idx >= len(rest_indices_list):
            break
            
        current_rest_idx = rest_indices_list[w_idx].cpu().numpy()
        
        window_dir = os.path.join(output_dir, task_name, f"w{w_idx}")
        os.makedirs(window_dir, exist_ok=True)
        
        for head_idx, head_matrix in enumerate(window_matrices):
            plt.figure(figsize=(10, 8))
            
            sns.heatmap(
                head_matrix, 
                cmap="magma", 
                vmin=0.0, 
                vmax=1.0,
                xticklabels=current_rest_idx,
                yticklabels=current_rest_idx
            )
            
            plt.title(f"Key Similarity Heatmap - {task_name}\nWindow {w_idx}, Head {head_idx}")
            plt.xlabel("Original Sequence Position (Non-Salient)")
            plt.ylabel("Original Sequence Position (Non-Salient)")
            
            heat_path = os.path.join(window_dir, f"head_{head_idx}.png")
            plt.savefig(heat_path, bbox_inches='tight')
            plt.close()

    print(f"[{task_name}] Saved Key diagnostics to {os.path.join(output_dir, task_name)}")
    
def run_single_sample_diagnostic():
    config = EvalConfig()
    diag_output_dir = os.path.join(config.output_dir, "diagnostics")
    
    model, tokenizer = load_model_and_tokenizer(config.model_id)
    
    for task_name in config.tasks:
        task_path = os.path.join(config.benchmark_dir, task_name)
        if not os.path.exists(task_path):
            print(f"Task path {task_path} not found. Skipping.")
            continue
            
        print(f"\n--- Running Value Diagnostic for: {task_name} ---")
        dataset = load_from_disk(task_path)
        task_data = dataset["validation"] if isinstance(dataset, dict) else dataset
        
        # Sticking to a single sample for the dataset
        example = task_data[0]
        prompt = example["input"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        cache = DiagnosticZipCache(
            window_size=config.window_size,
            top_k_ratio=config.top_k_ratio,
            group_size=config.group_size,
        )

        print(f"Processing sample (Length: {inputs.input_ids.shape[1]} tokens)...")
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                past_key_values=cache,
                use_cache=True,
                num_return_sequences=1,
                do_sample=False,
                return_dict_in_generate=True,
            )
            
        plot_diagnostics(task_name, cache, diag_output_dir)

if __name__ == "__main__":
    run_single_sample_diagnostic()