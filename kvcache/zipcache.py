# The idea now is to better the 4bit kv cache 
# to do so our claim starts from the zipcache paper implementation 
# which identifies HH tokens based on normalized attention scores 
'''
So a for a new token they compute saliency (after 100 tokens) using the attention matrix
generated from processing those 100 tokens (how much each token attends to others)
They take that matrix and apply Normalization by dividing the column sums by the n# of non zero elements.
Which gives us the saliency of each token.
The 100 tokens are ranked by saliency and the top mosts are assigned higher precision

Before quantizing they normalize over each channel to supress outliers,
find the max value and divide by it.

They quantize keys with per channel symmetric quant 
and values using per token Symmetric quant 
'''

from transformers.cache_utils import Cache
import torch
from typing import Optional, Dict, Any, Tuple

from kvcache.int4 import Int4Cache
from kvcache.int8 import Int8Cache

class ZipCache(Cache):
    def __init__(
        self,
        window_size: int = 100,
        top_k_ratio: float = 0.3,
        group_size: int = 32,
    ):

        self.window_size = window_size
        self.top_k_ratio = top_k_ratio

        self.int8_cache = Int8Cache(bits=8)
        self.int4_cache = Int4Cache(group_size=group_size)

        # committed history 
        self.global_salient_idx: Dict[int, list] = {}
        self.global_rest_idx: Dict[int, list] = {}
        self.window_k_scales: Dict[int, list] = {}
        self.window_v_scales: Dict[int, list] = {}
        self.committed_lengths: Dict[int, int] = {}

        # keys and values being processed on the windw
        self.staging_keys: Dict[int, torch.Tensor] = {}
        self.staging_values: Dict[int, torch.Tensor] = {}

        self.layers = []
        self.layer_class_to_replicate = None
    
    # for outliers suppression
    # for keys we scale across the whole seq to avoid attention drifts
    def _suppress_keys(self, tensor: torch.Tensor):
        scale = tensor.abs().amax(dim=2, keepdim=True).clamp(min=1e-5)
        return tensor / scale, scale

    def _suppress_values(self, tensor: torch.Tensor):
        scale = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        return tensor / scale, scale
    
    #  Q = K 
    # in the paper they use probe tokens to be compatible with flash attn
    # here since the window is 100 token we can use the whole 100s as probe set 
    def _compute_saliency(self, keys: torch.Tensor) -> torch.Tensor:
        head_dim = keys.shape[-1]
        # dot prod between each K and rest of attn func
        attn_logits = torch.matmul(keys, keys.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        
        col_sums = attn_weights.sum(dim=-2)
        nonzero = (attn_weights > 0).sum(dim=-2).clamp(min=1).float()
        saliency = col_sums / nonzero
        
        return saliency.mean(dim=(0, 1))
    
    def _commit_window(self, layer_idx: int, window_keys: torch.Tensor, window_values: torch.Tensor):
        batch, heads, seq, head_dim = window_keys.shape
        top_k = max(1, int(seq * self.top_k_ratio))
    
        norm_keys, k_scale = self._suppress_keys(window_keys)
        norm_values, v_scale = self._suppress_values(window_values)

        saliency = self._compute_saliency(norm_keys)
        _, ranked = saliency.sort(descending=True) # ranked here has the idnexes of the sorted scores which we can use to reconstract the original sequence mantaining order 
        # we perform the split to route each to the corresponding cache 
        salient_idx = ranked[:top_k]
        rest_idx = ranked[top_k:]

        # salient to int8 cache update Re-apply the scale multipliers window by window
        self.int8_cache.update(
            norm_keys[:, :, salient_idx, :], norm_values[:, :, salient_idx, :], layer_idx
        )
        # and non salient go to int4 to save vram 
        self.int4_cache.update(
            norm_keys[:, :, rest_idx, :], norm_values[:, :, rest_idx, :], layer_idx
        )

        # how many tokens long is the commited cache, w1 offset=0, w2=offset=100
        offset = self.committed_lengths.get(layer_idx, 0)
        # needed to reconstruct the history preserving order 
        self.global_salient_idx.setdefault(layer_idx, []).append(salient_idx + offset) # givesthe token an index outside of the window for from a fixed (0, 99) to global index 
        self.global_rest_idx.setdefault(layer_idx, []).append(rest_idx + offset)
        self.window_k_scales.setdefault(layer_idx, []).append(k_scale)
        self.window_v_scales.setdefault(layer_idx, []).append(v_scale)
        
        self.committed_lengths[layer_idx] = offset + seq # update for the next window 

    def _reconstruct_history(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # fetch the complete dequantized sequence divided in the 2 caches 
        full_int8_k, full_int8_v = self.int8_cache.get_full_history(layer_idx)
        full_int4_k, full_int4_v = self.int4_cache.get_full_history(layer_idx)
        
        total_seq = self.committed_lengths[layer_idx]
        batch, heads, _, head_dim = full_int8_k.shape
        device, dtype = full_int8_k.device, full_int8_k.dtype
        
        # placehlder for the entire squence 
        recon_k = torch.empty(batch, heads, total_seq, head_dim, device=device, dtype=dtype)
        recon_v = torch.empty(batch, heads, total_seq, head_dim, device=device, dtype=dtype)

        # because we processed 100 tokens window we merge them with cat and order later
        # so now we obtain the indeces for each token wrt to the full sequence 
        master_salient = torch.cat(self.global_salient_idx[layer_idx])
        master_rest = torch.cat(self.global_rest_idx[layer_idx])

        # we assign tokens back to original position using the indeces 
        # so recon_k now is filled with the tensor in the correct order
        recon_k[:, :, master_salient, :] = full_int8_k
        recon_v[:, :, master_salient, :] = full_int8_v
        recon_k[:, :, master_rest, :] = full_int4_k
        recon_v[:, :, master_rest, :] = full_int4_v

        # The values from the step before are still normalized to account for outlier suppression 
        # so for each window we iterate and restore using the specific scale 
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
        # we need to collected tokens up to the window size to be able to perform our steps 
        if layer_idx not in self.staging_keys:
            self.staging_keys[layer_idx] = key_states
            self.staging_values[layer_idx] = value_states
        else:
            self.staging_keys[layer_idx] = torch.cat([self.staging_keys[layer_idx], key_states], dim=2)
            self.staging_values[layer_idx] = torch.cat([self.staging_values[layer_idx], value_states], dim=2)

        # we move to _commit window after collectin window_size tokens 
        while self.staging_keys[layer_idx].shape[2] >= self.window_size:
            window_k = self.staging_keys[layer_idx][:, :, :self.window_size, :]
            window_v = self.staging_values[layer_idx][:, :, :self.window_size, :]
            
            self._commit_window(layer_idx, window_k, window_v)
            
            self.staging_keys[layer_idx] = self.staging_keys[layer_idx][:, :, self.window_size:, :]
            self.staging_values[layer_idx] = self.staging_values[layer_idx][:, :, self.window_size:, :]

        staging_k = self.staging_keys[layer_idx]
        staging_v = self.staging_values[layer_idx]
        # before 100 tokens are gathered with return the raw tokens 

        if layer_idx not in self.committed_lengths or self.committed_lengths[layer_idx] == 0:
            return staging_k, staging_v
            
        hist_k, hist_v = self._reconstruct_history(layer_idx)
        
        full_k = torch.cat([hist_k, staging_k], dim=2)
        full_v = torch.cat([hist_v, staging_v], dim=2)
        # tokens in the trainling window will be left at full precision since they wont enter commit window and pass to the int4 or int8 caches 
        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        committed_len = self.committed_lengths.get(layer_idx, 0)
        staging_len = self.staging_keys[layer_idx].shape[2] if layer_idx in self.staging_keys else 0
        return committed_len + staging_len

    def get_max_length(self) -> Optional[int]:
        return None