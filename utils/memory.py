import torch 


def measure_weight_footprint(model) -> dict:
    """Captures the fixed VRAM cost of the loaded weights"""
    devices = sorted(
        {p.device for p in model.parameters() if p.device.type == "cuda"},
        key=lambda d: d.index,
    )
    for d in devices:
        torch.cuda.synchronize(d)
    weights_bytes = sum(torch.cuda.memory_allocated(d) for d in devices)
    return {"devices": [str(d) for d in devices], "weights_mb": weights_bytes / (1024 ** 2)}
 
 
def kv_cache_size_mb(cache) -> float:
    """Exact VRAM footprint of a KV cache object, in MB. whcih will work for hf newer imple, legacy tuple with kv cache and our custom impl"""
  
    total_bytes = 0
 
    def walk(obj):
        nonlocal total_bytes
        if torch.is_tensor(obj):
            total_bytes += obj.element_size() * obj.nelement()
        elif isinstance(obj, dict):
            for value in obj.values():
                walk(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)
        elif hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                walk(value)
 
    walk(cache)
    return total_bytes / (1024 ** 2)