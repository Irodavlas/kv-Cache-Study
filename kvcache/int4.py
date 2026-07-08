from transformers.cache_utils import Cache
import torch
from typing import Optional, Dict, Any, Tuple

# bsae 4int kv cache 
class Int4Cache(Cache):
    def __init__(self, group_size: int = 32):
        #super().__init__()
        self.bits = 4
        self.group_size = group_size

        self.layers = []
        self.layer_class_to_replicate = None
        
        self.keys: Dict[int, torch.Tensor] = {}
        self.values: Dict[int, torch.Tensor] = {}

        self.key_scales: Dict[int, torch.Tensor] = {}
        self.key_zps: Dict[int, torch.Tensor] = {}

        self.value_scales: Dict[int, torch.Tensor] = {}
        self.value_zps: Dict[int, torch.Tensor] = {}
       
    def _pack(self, quantized: torch.Tensor):
        even = quantized[..., 0::2].to(torch.uint8) & 0x0F
        odd  = quantized[..., 1::2].to(torch.uint8) & 0x0F
        # Ox0F is 0000 1111
        # in bit packing, We shift even_vals by 4 bits and do the OR with 0000
        # whilst odd_Vals does bit-w AND with 1111
        return (even << 4) | odd

    def _unpack(self, packed_tensor: torch.Tensor, dtype, device) -> torch.Tensor:
        even = (packed_tensor >> 4) & 0x0F
        odd  = packed_tensor & 0x0F
        unpacked = torch.stack([even, odd], dim=-1).view(*packed_tensor.shape[:-1], -1)
        return unpacked.to(dtype)

    # asymmetric quantization for keys 
    # computing the quant parameters in groups rather than 
    # for the entire head_dim 
    def _quantize_keys(self, tensor: torch.Tensor):
        batch, heads, seq, head_dim = tensor.shape
        num_groups = head_dim // self.group_size
        grouped = tensor.view(batch, heads, seq, num_groups, self.group_size)
        # min max values in each group.
        mn = grouped.amin(dim=-1, keepdim=True)
        mx = grouped.amax(dim=-1, keepdim=True)
        scale = (mx - mn).clamp(min=1e-5) / 15
        zero_point = torch.round(-mn / scale).clamp(0, 15)

        quantized = torch.round(grouped / scale) + zero_point
        quantized = quantized.clamp(0, 15).to(torch.uint8)
        
        return self._pack(quantized), scale, zero_point

    # asymmetric, per-token over the full head dim
    def _quantize_values(self, tensor: torch.Tensor):
        mn = tensor.amin(dim=-1, keepdim=True)
        mx = tensor.amax(dim=-1, keepdim=True)
        scale = (mx - mn).clamp(min=1e-5) / 15
        zero_point = torch.round(-mn / scale).clamp(0, 15)
        
        quantized = torch.round(tensor / scale) + zero_point
        quantized = quantized.clamp(0, 15).to(torch.uint8)
        
        return self._pack(quantized), scale, zero_point

    def _dequantize_keys(self, packed_keys: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        # unpack the two 4-bit values from the tensor
        unpacked = self._unpack(packed_keys, scale.dtype, scale.device)
        
        batch, heads, seq, num_groups, group_size = unpacked.shape 

        dequantized = (unpacked - zero_point) * scale

        # return back to head dim doing g_size * n_groups 
        return dequantized.view(batch, heads, seq, num_groups * group_size)

    def _dequantize_values(self, packed_values: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        unpacked = self._unpack(packed_values, scale.dtype, scale.device)
        return (unpacked - zero_point) * scale

    # the update step needs to initialize the keys and values storage if first time 
    # otw we concat to old history after quantizing and computing scale and zero_p
    def update(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int, 
        cache_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        q_keys, k_scale, k_zp = self._quantize_keys(key_states)
        q_vals, v_scale, v_zp = self._quantize_values(value_states)

        if layer_idx not in self.keys:
            self.keys[layer_idx], self.key_scales[layer_idx], self.key_zps[layer_idx] = q_keys, k_scale, k_zp
            self.values[layer_idx], self.value_scales[layer_idx], self.value_zps[layer_idx] = q_vals, v_scale, v_zp
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], q_keys], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], q_vals], dim=2)
            
            self.key_scales[layer_idx] = torch.cat([self.key_scales[layer_idx], k_scale], dim=2)
            self.key_zps[layer_idx] = torch.cat([self.key_zps[layer_idx], k_zp], dim=2)
            
            self.value_scales[layer_idx] = torch.cat([self.value_scales[layer_idx], v_scale], dim=2)
            self.value_zps[layer_idx] = torch.cat([self.value_zps[layer_idx], v_zp], dim=2)

        # we return the full unquantized history for the attention computation 
        dequantized_keys = self._dequantize_keys(self.keys[layer_idx], self.key_scales[layer_idx], self.key_zps[layer_idx])
        dequantized_values = self._dequantize_values(self.values[layer_idx], self.value_scales[layer_idx], self.value_zps[layer_idx])

        return dequantized_keys, dequantized_values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self.keys:
            return 0
        return self.keys[layer_idx].shape[2]

    def get_max_length(self) -> Optional[int]:
        return None
    
    # implmeneted along the zipCache code to obtain the full history instead of using the Update 
    def get_full_history(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.keys:
            raise ValueError(f"Layer {layer_idx} not found in cache.")
            
        dequant_k = self._dequantize_keys(
            self.keys[layer_idx], 
            self.key_scales[layer_idx], 
            self.key_zps[layer_idx]
        )
        dequant_v = self._dequantize_values(
            self.values[layer_idx], 
            self.value_scales[layer_idx], 
            self.value_zps[layer_idx]
        )
        
        return dequant_k, dequant_v