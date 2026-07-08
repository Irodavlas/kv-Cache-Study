import os
import json
import gc
from dataclasses import dataclass, field
from typing import Callable, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm

from utils.memory import measure_weight_footprint, kv_cache_size_mb
from kvcache.int8 import Int8Cache
from kvcache.int4 import Int4Cache

@dataclass
class EvalConfig:
    model_id: str = "model/llama-3.2-3b"
    benchmark_dir: str = "benchmark/ruler"
    
    # "int8", "int4"
    precision_mode: str = "int8" 
    
    tasks: list = field(default_factory=lambda: ["cwe_4k", "qa_1_4k", "qa_2_4k", "vt_4k"])
    max_new_tokens: int = 50
    group_size: int = 32  # Only used for int4

    @property
    def output_dir(self) -> str:
        """Dynamically switches output directories based on chosen KV layout"""
        return f"results/{self.precision_mode}"


def string_match_part(prediction: str, answers: list[str]) -> float:
    pred_lower = prediction.lower()
    return 1.0 if any(ans.lower() in pred_lower for ans in answers) else 0.0
 
 
def string_match_all(prediction: str, answers: list[str]) -> float:
    if not answers:
        return 0.0
    pred_lower = prediction.lower()
    hits = sum(1 for ans in answers if ans.lower() in pred_lower)
    return hits / len(answers)


METRIC_BY_TASK_PREFIX: dict[str, Callable[[str, list[str]], float]] = {
    "cwe": string_match_all,
    "vt": string_match_all,
    "qa_1": string_match_part,
    "qa_2": string_match_part,
}
 
def get_metric_fn(task_name: str) -> Callable[[str, list[str]], float]:
    base = task_name.rsplit("_", 1)[0]
    if base not in METRIC_BY_TASK_PREFIX:
        raise ValueError(f"No metric registered for task '{task_name}' (base '{base}').")
    return METRIC_BY_TASK_PREFIX[base]


def load_model_and_tokenizer(model_path):
    print(f"Loading model and tokenizer from local path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},           # Anchors weights to GPU 0 to block OOM CPU swaps
        attn_implementation="sdpa"     # Employs optimized PyTorch attention kernels
    )
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def get_cache_factory(config: EvalConfig) -> Union[Callable, None]:
    """Returns a callable factory to generate empty instances per prompt execution."""
    if config.precision_mode == "baseline":
        return None
    elif config.precision_mode == "int8":
        return lambda: Int8Cache()
    elif config.precision_mode == "int4":
        return lambda: Int4Cache(group_size=config.group_size)
    else:
        raise ValueError(f"Unknown precision mode: {config.precision_mode}")


def generate_prediction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int,
    cache_factory: Union[Callable, None],
) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
 
    # Instantiate a clean, isolated cache container for this sample
    past_key_values = cache_factory() if cache_factory is not None else None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            past_key_values=past_key_values,
            use_cache=True,
            num_return_sequences=1,
            do_sample=False,
            return_dict_in_generate=True,
        )
 
    generated_tokens = outputs.sequences[0][input_length:]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return prediction, kv_cache_size_mb(outputs.past_key_values)


def evaluate_task(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    task_name: str, 
    config: EvalConfig, 
    weights_mb: float,
    cache_factory: Union[Callable, None]
) -> None:
    task_path = os.path.join(config.benchmark_dir, task_name)
    if not os.path.exists(task_path):
        print(f"Task path {task_path} not found. Skipping.")
        return
 
    print(f"\n--- Running [{config.precision_mode.upper()}] Evaluation on: {task_name} ---")
 
    dataset: Union[Dataset, DatasetDict] = load_from_disk(task_path)
    task_data = dataset["validation"] if isinstance(dataset, dict) else dataset
    metric_fn = get_metric_fn(task_name)
 
    results = []
    total_score = 0.0
    kv_cache_sizes_mb = []
 
    for idx, example in enumerate(tqdm(task_data)):
        prompt = example["input"]
        answers = example["answers"]
 
        prediction, kv_mb = generate_prediction(model, tokenizer, prompt, config.max_new_tokens, cache_factory)
        score = metric_fn(prediction, answers)
        total_score += score
        kv_cache_sizes_mb.append(kv_mb)
 
        results.append({
            "index": example.get("index", idx),
            "prompt_length": example.get("length"),
            "ground_truth_options": answers,
            "prediction": prediction,
            "score": score,
            "correct": score == 1.0,
            "kv_cache_mb": kv_mb,
        })

        if idx % 10 == 0:
            gc.collect()https://github.com/simonerufo/Motion-Neural-Cellular-Automata
            torch.cuda.empty_cache()
 
    mean_score = total_score / len(results) if results else 0.0
    avg_kv_mb = sum(kv_cache_sizes_mb) / len(kv_cache_sizes_mb) if kv_cache_sizes_mb else 0.0
    peak_kv_mb = max(kv_cache_sizes_mb) if kv_cache_sizes_mb else 0.0
    
    print(
        f"Task: {task_name} | Mean Score: {mean_score:.4f} | "
        f"Avg KV cache: {avg_kv_mb:.1f} MB | Peak KV cache: {peak_kv_mb:.1f} MB"
    )

    os.makedirs(config.output_dir, exist_ok=True)
    output_file = os.path.join(config.output_dir, f"{task_name}_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "task": task_name,
            "precision_mode": config.precision_mode,
            "score": mean_score,
            "num_examples": len(results),
            "weights_mb": weights_mb,
            "avg_kv_cache_mb": avg_kv_mb,
            "peak_kv_cache_mb": peak_kv_mb,
            "predictions": results,
        }, f, indent=4)
 
 
def main():
    config = EvalConfig()
    model, tokenizer = load_model_and_tokenizer(config.model_id)
    
    weight_info = measure_weight_footprint(model)
    print(f"Weight footprint: {weight_info['weights_mb']:.1f} MB on {weight_info['devices']}")
 
    # Fetch cache constructor based on chosen configuration string
    cache_factory = get_cache_factory(config)
 
    for task_name in config.tasks:
        evaluate_task(model, tokenizer, task_name, config, weight_info["weights_mb"], cache_factory)
 
 
if __name__ == "__main__":
    main()