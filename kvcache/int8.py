from transformers.cache_utils import Cache
import torch
from typing import Optional, Dict, Any, Tuple

class Int8Cache(Cache):
    def __init__(self, bits=8):
        #super().__init__() 
        self.bits = bits

        self.layers = []
        self.layer_class_to_replicate = None

        self.keys: Dict[int, torch.Tensor] = {}
        self.values: Dict[int, torch.Tensor] = {}
        self.key_scales: Dict[int, torch.Tensor] = {}
        self.value_scales: Dict[int, torch.Tensor] = {}

    def _quantize_tensor(self, tensor: torch.Tensor):
        """
        We perform Symmetric quantization (max abs) on the input tensor  
        tensor shape: (batch, heads, seq, head_dim):
            seq -> tokens the model saw during apply_chat_template()
            head_dim -> size of the key vector 
        """

        # new boundaries for the quantized representation bits=8 then [-128, 127] 
        qmax = (2 ** (self.bits - 1)) - 1
        qmin = -(2 ** (self.bits - 1))

        scale = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / qmax

        quantized = torch.round(tensor / scale).clamp(qmin, qmax).to(torch.int8)
        return quantized, scale


    def _dequantize(self, quantized_tensor: torch.Tensor, metadata):
        scale = metadata[0]
        return quantized_tensor.to(scale.dtype) * scale

    def update(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int, 
        cache_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        q_keys, k_scale = self._quantize_tensor(key_states)
        q_vals, v_scale = self._quantize_tensor(value_states)

        if layer_idx not in self.keys:
            self.keys[layer_idx] = q_keys
            self.values[layer_idx] = q_vals
            self.key_scales[layer_idx] = k_scale
            self.value_scales[layer_idx] = v_scale
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], q_keys], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], q_vals], dim=2)
            self.key_scales[layer_idx] = torch.cat([self.key_scales[layer_idx], k_scale], dim=2)
            self.value_scales[layer_idx] = torch.cat([self.value_scales[layer_idx], v_scale], dim=2)
            
        dequantized_keys = self._dequantize(self.keys[layer_idx], self.key_scales[layer_idx])
        dequantized_values = self._dequantize(self.values[layer_idx], self.value_scales[layer_idx])
        
        return dequantized_keys, dequantized_values
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self.keys:
            return 0
        return self.keys[layer_idx].shape[2]

    def get_max_length(self) -> Optional[int]:
        return None
    
    def get_full_history(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.keys:
            raise ValueError(f"Layer {layer_idx} not found in cache.")
        
        dequant_k = self._dequantize(self.keys[layer_idx], [self.key_scales[layer_idx]])
        dequant_v = self._dequantize(self.values[layer_idx], [self.value_scales[layer_idx]])
        
        return dequant_k, dequant_v