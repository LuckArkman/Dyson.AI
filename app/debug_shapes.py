import json
import os
import numpy as np

def debug_ckpt_shapes():
    ckpt_path = os.path.join("checkpoints", "v1.2_compressed", "weights", "weight_registry.json")
    with open(ckpt_path, 'r') as f:
        reg = json.load(f)
    
    for name, meta in reg['layers'].items():
        print(f"Layer: {name}")
        print(f"  Registry Shape: {meta['shape']}")
        if os.path.exists(meta['path']):
            data = np.load(meta['path'], mmap_mode='r')
            print(f"  Actual File Shape: {data.shape}")
            if hasattr(data, 'dtype'): print(f"  Dtype: {data.dtype}")
        else:
            print(f"  [ERR] File not found: {meta['path']}")

if __name__ == "__main__":
    debug_ckpt_shapes()
