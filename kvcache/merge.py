from transformers.cache_utils import Cache
import torch
from typing import Optional, Dict, Any, Tuple

from kvcache.int4 import Int4Cache
from kvcache.int8 import Int8Cache


class MergeCache(Cache):
    def __init__(
        self,
        window_size: int = 100,
        top_k_ratio: float = 0.3,
        group_size: int = 32,
        merge_ratio: float = 0.5,
        min_tokens_to_merge: int = 4,
    ):
        
        self.window_size = window_size
        self.top_k_ratio = top_k_ratio
        self.merge_ratio = merge_ratio
        self.min_tokens_to_merge = min_tokens_to_merge

        self.int8_cache = Int8Cache(bits=8)
        self.int4_cache = Int4Cache(group_size=group_size)

        # committed history
        self.global_salient_idx: Dict[int, list] = {}
        self.global_rest_idx: Dict[int, list] = {}
        self.global_rest_rows: Dict[int, list] = {}  
        self.window_k_scales: Dict[int, list] = {}
        self.window_v_scales: Dict[int, list] = {}
        self.committed_lengths: Dict[int, int] = {}
        self.int4_row_lengths: Dict[int, int] = {} # full lenght of processed tokens 

        # keys and values being processed on the window
        self.staging_keys: Dict[int, torch.Tensor] = {}
        self.staging_values: Dict[int, torch.Tensor] = {}

        self.layers = []
        self.layer_class_to_replicate = None

    # normalization 
    def _suppress_keys(self, tensor: torch.Tensor):
        scale = tensor.abs().amax(dim=2, keepdim=True).clamp(min=1e-5)
        return tensor / scale, scale

    def _suppress_values(self, tensor: torch.Tensor):
        scale = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        return tensor / scale, scale

    def _compute_saliency(self, keys: torch.Tensor) -> torch.Tensor:
        head_dim = keys.shape[-1]
        attn_logits = torch.matmul(keys, keys.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = torch.softmax(attn_logits, dim=-1)

        col_sums = attn_weights.sum(dim=-2)
        nonzero = (attn_weights > 0).sum(dim=-2).clamp(min=1).float()
        saliency = col_sums / nonzero

        return saliency.mean(dim=(0, 1))

    # bipartite match + merge for the non-salient tokens of one window.
    # rest_idx are local indices (0..window_size-1) into norm_keys/norm_values.
    # returns the key/value tensors to store, plus a
    # rows tensor of length == len(rest_idx) mapping each rest token to its
    # row in the returned tensors (duplicate rows for merged pairs).
    def _tome_merge(self, keys: torch.Tensor, values: torch.Tensor, rest_idx: torch.Tensor):
        n = rest_idx.shape[0]
        batch, heads, _, head_dim = keys.shape
        device = keys.device

        if n < self.min_tokens_to_merge: 
            rows = torch.arange(n, device=device)
            return keys[:, :, rest_idx, :], values[:, :, rest_idx, :], rows
        # divide into 2 sets beased on index odd even 
        set_a = rest_idx[0::2]
        set_b = rest_idx[1::2]
        len_a, len_b = set_a.shape[0], set_b.shape[0]

        k_a, k_b = keys[:, :, set_a, :], keys[:, :, set_b, :]
        v_a, v_b = values[:, :, set_a, :], values[:, :, set_b, :]
       
        # cos similarity between keys 
        k_a_n = torch.nn.functional.normalize(k_a, dim=-1)
        k_b_n = torch.nn.functional.normalize(k_b, dim=-1)
        sim = torch.matmul(k_a_n, k_b_n.transpose(-2, -1)).mean(dim=(0, 1))  # (len_a, len_b)

        best_val, best_idx = sim.max(dim=-1) # greedy partnering approach
        order = torch.argsort(best_val, descending=True).tolist() # ordering the pairs by simil

        r = int(min(len_a, len_b) * self.merge_ratio) # max pairs to merge 

        merged_a, merged_b = set(), set()
        a_to_row = [-1] * len_a
        b_to_row = [-1] * len_b
        merged_k, merged_v = [], [] # to avoid duplicated merging

        row = 0
        for a_i in order:
            if row >= r:
                break
            b_i = best_idx[a_i].item()
            if b_i in merged_b:
                continue
            merged_a.add(a_i)
            merged_b.add(b_i)
            a_to_row[a_i] = row
            b_to_row[b_i] = row
            # merging their keys and values 
            merged_k.append((k_a[:, :, a_i, :] + k_b[:, :, b_i, :]) / 2) 
            merged_v.append((v_a[:, :, a_i, :] + v_b[:, :, b_i, :]) / 2)
            row += 1

        # handling the indeces for the reconstruction 
        # we save it out of order and create the mapping.
        # e.g [0, 1, 2, 3]
        # A = 0, 2 | B = 1, 3
        # we merge 0 and 3 and assign to row 0
        # 1 and 2 to rows 1 and 2 (no merge)
        # final = [avg(0,3), 2, 1] rows = [0, 2, 1, 0] (indeces)
        unmerged_a = [i for i in range(len_a) if i not in merged_a]
        unmerged_b = [i for i in range(len_b) if i not in merged_b]

        next_row = row
        for i in unmerged_a:
            a_to_row[i] = next_row
            next_row += 1
        for i in unmerged_b:
            b_to_row[i] = next_row
            next_row += 1

        if merged_k:
            merged_k_t = torch.stack(merged_k, dim=2)
            merged_v_t = torch.stack(merged_v, dim=2)
        else:
            merged_k_t = keys.new_empty((batch, heads, 0, head_dim))
            merged_v_t = values.new_empty((batch, heads, 0, head_dim))

        final_keys = torch.cat([merged_k_t, k_a[:, :, unmerged_a, :], k_b[:, :, unmerged_b, :]], dim=2)
        final_values = torch.cat([merged_v_t, v_a[:, :, unmerged_a, :], v_b[:, :, unmerged_b, :]], dim=2)

        rows = torch.empty(n, dtype=torch.long, device=device)
        rows[0::2] = torch.tensor(a_to_row, device=device)
        rows[1::2] = torch.tensor(b_to_row, device=device)

        return final_keys, final_values, rows

    # same as the zip cache approach, but before routing non salient tokens to int4 
    # we pass them to function tome for merging
    def _commit_window(self, layer_idx: int, window_keys: torch.Tensor, window_values: torch.Tensor):
        batch, heads, seq, head_dim = window_keys.shape
        top_k = max(1, int(seq * self.top_k_ratio))

        norm_keys, k_scale = self._suppress_keys(window_keys)
        norm_values, v_scale = self._suppress_values(window_values)

        saliency = self._compute_saliency(norm_keys)
        _, ranked = saliency.sort(descending=True)
        salient_idx = ranked[:top_k]
        rest_idx = ranked[top_k:]

        self.int8_cache.update(
            norm_keys[:, :, salient_idx, :], norm_values[:, :, salient_idx, :], layer_idx
        )

        merged_k, merged_v, rest_rows = self._tome_merge(norm_keys, norm_values, rest_idx)
        self.int4_cache.update(merged_k, merged_v, layer_idx)

        offset = self.committed_lengths.get(layer_idx, 0)
        row_offset = self.int4_row_lengths.get(layer_idx, 0)

        self.global_salient_idx.setdefault(layer_idx, []).append(salient_idx + offset)
        self.global_rest_idx.setdefault(layer_idx, []).append(rest_idx + offset)
        self.global_rest_rows.setdefault(layer_idx, []).append(rest_rows + row_offset)
        self.window_k_scales.setdefault(layer_idx, []).append(k_scale)
        self.window_v_scales.setdefault(layer_idx, []).append(v_scale)

        self.committed_lengths[layer_idx] = offset + seq
        self.int4_row_lengths[layer_idx] = row_offset + merged_k.shape[2]

    
    def _reconstruct_history(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        full_int8_k, full_int8_v = self.int8_cache.get_full_history(layer_idx)
        full_int4_k, full_int4_v = self.int4_cache.get_full_history(layer_idx)

        total_seq = self.committed_lengths[layer_idx]
        batch, heads, _, head_dim = full_int8_k.shape
        device, dtype = full_int8_k.device, full_int8_k.dtype

        recon_k = torch.empty(batch, heads, total_seq, head_dim, device=device, dtype=dtype)
        recon_v = torch.empty(batch, heads, total_seq, head_dim, device=device, dtype=dtype)

        master_salient = torch.cat(self.global_salient_idx[layer_idx])
        master_rest = torch.cat(self.global_rest_idx[layer_idx])
        master_rest_rows = torch.cat(self.global_rest_rows[layer_idx])

        recon_k[:, :, master_salient, :] = full_int8_k
        recon_v[:, :, master_salient, :] = full_int8_v
        # merged rows get gathered onto both original positions they came from
        recon_k[:, :, master_rest, :] = full_int4_k[:, :, master_rest_rows, :]
        recon_v[:, :, master_rest, :] = full_int4_v[:, :, master_rest_rows, :]

        for w_idx, (k_scale, v_scale) in enumerate(zip(self.window_k_scales[layer_idx], self.window_v_scales[layer_idx])):
            start = w_idx * self.window_size
            end = start + self.window_size

            recon_k[:, :, start:end, :] *= k_scale
            recon_v[:, :, start:end, :] *= v_scale

        return recon_k, recon_v

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.staging_keys:
            self.staging_keys[layer_idx] = key_states
            self.staging_values[layer_idx] = value_states
        else:
            self.staging_keys[layer_idx] = torch.cat([self.staging_keys[layer_idx], key_states], dim=2)
            self.staging_values[layer_idx] = torch.cat([self.staging_values[layer_idx], value_states], dim=2)

        while self.staging_keys[layer_idx].shape[2] >= self.window_size:
            window_k = self.staging_keys[layer_idx][:, :, :self.window_size, :]
            window_v = self.staging_values[layer_idx][:, :, :self.window_size, :]

            self._commit_window(layer_idx, window_k, window_v)

            self.staging_keys[layer_idx] = self.staging_keys[layer_idx][:, :, self.window_size:, :]
            self.staging_values[layer_idx] = self.staging_values[layer_idx][:, :, self.window_size:, :]

        staging_k = self.staging_keys[layer_idx]
        staging_v = self.staging_values[layer_idx]

        if layer_idx not in self.committed_lengths or self.committed_lengths[layer_idx] == 0:
            return staging_k, staging_v

        hist_k, hist_v = self._reconstruct_history(layer_idx)

        full_k = torch.cat([hist_k, staging_k], dim=2)
        full_v = torch.cat([hist_v, staging_v], dim=2)
        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        committed_len = self.committed_lengths.get(layer_idx, 0)
        staging_len = self.staging_keys[layer_idx].shape[2] if layer_idx in self.staging_keys else 0
        return committed_len + staging_len

    def get_max_length(self) -> Optional[int]:
        return None