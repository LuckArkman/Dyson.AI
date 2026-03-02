import numpy as np
import io
import struct
import uuid
import threading
from typing import Dict, Optional
from gpu.gpu_math_engine import GpuMathEngine

class DiskSwapManager:
    def __init__(self):
        self._memory_swap: Dict[str, bytes] = {}
        self._lock = threading.Lock()
        print("[DiskSwap] ⚡ Modo Híbrido Ativo (Python).")

    def swap_out(self, tensor_data: np.ndarray, label: str) -> str:
        swap_id = f"mem://{label}_{uuid.uuid4().hex}"
        
        # Serialização eficiente para bytes
        with io.BytesIO() as buf:
            # Metadata: Rank, Shape, Length
            buf.write(struct.pack("i", len(tensor_data.shape)))
            for dim in tensor_data.shape:
                buf.write(struct.pack("i", int(dim)))
            buf.write(struct.pack("q", int(tensor_data.size)))
            
            # Data
            buf.write(tensor_data.tobytes())
            data_bytes = buf.getvalue()

        with self._lock:
            self._memory_swap[swap_id] = data_bytes
        
        return swap_id

    def load_from_swap(self, swap_id: str) -> np.ndarray:
        with self._lock:
            if swap_id not in self._memory_swap:
                raise FileNotFoundError(f"Swap ID não encontrado: {swap_id}")
            data_bytes = self._memory_swap[swap_id]

        with io.BytesIO(data_bytes) as buf:
            shape_rank = struct.unpack("i", buf.read(4))[0]
            shape = tuple(struct.unpack("i", buf.read(4))[0] for _ in range(shape_rank))
            length = struct.unpack("q", buf.read(8))[0]
            
            raw_data = buf.read()
            return np.frombuffer(raw_data, dtype=np.float32).reshape(shape)

    def delete_swap_file(self, swap_id: str):
        with self._lock:
            if swap_id in self._memory_swap:
                del self._memory_swap[swap_id]

    def clear_all_swap(self):
        with self._lock:
            self._memory_swap.clear()
