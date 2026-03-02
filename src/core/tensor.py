import numpy as np
from typing import Tuple, List, Optional
import os
import struct

class Tensor:
    def __init__(self, shape: Tuple[int, ...], data: Optional[np.ndarray] = None):
        self.shape = tuple(shape)
        self.length = np.prod(shape)
        self.data = data if data is not None else np.zeros(shape, dtype=np.float32)

    @classmethod
    def from_host_data(cls, data: np.ndarray, shape: Tuple[int, ...]):
        return cls(shape, data.astype(np.float32))

class IndividualFileTensorManager:
    def __init__(self, session_id: str, base_dir: str = "Dayson/TensorCache"):
        self.session_id = session_id
        self.tensor_dir = os.path.abspath(os.path.join(base_dir, session_id))
        self.tensor_index = {}
        self.next_tensor_id = 0

        if os.path.exists(self.tensor_dir):
            for f in os.listdir(self.tensor_dir):
                if f.endswith(".bin"):
                    tid = f[:-4]
                    # Tenta carregar o shape do arquivo se essencial
                    self.tensor_index[tid] = None
                    try:
                        self.next_tensor_id = max(self.next_tensor_id, int(tid.split("_")[-1]))
                    except: pass
        else:
            os.makedirs(self.tensor_dir, exist_ok=True)
        print(f"[TensorManager] Cache de pesos em: {self.tensor_dir}")

    def get_path_for_id(self, tensor_id: str) -> str:
        return os.path.join(self.tensor_dir, f"{tensor_id}.bin")

    def store_tensor(self, tensor: Tensor, name: str) -> str:
        self.next_tensor_id += 1
        safe_name = name.replace(" ", "_")
        tensor_id = f"{safe_name}_{self.next_tensor_id:08d}"
        file_path = self.get_path_for_id(tensor_id)

        with open(file_path, "wb") as f:
            f.write(struct.pack("i", len(tensor.shape)))
            for dim in tensor.shape:
                f.write(struct.pack("i", int(dim)))
            f.write(struct.pack("q", int(tensor.length)))
            f.write(tensor.data.tobytes())

        self.tensor_index[tensor_id] = tensor.shape
        return tensor_id

    def load_tensor(self, tensor_id: str) -> Tensor:
        file_path = self.get_path_for_id(tensor_id)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tensor {tensor_id} não encontrado em {file_path}")

        with open(file_path, "rb") as f:
            shape_rank = struct.unpack("i", f.read(4))[0]
            shape = tuple(struct.unpack("i", f.read(4))[0] for _ in range(shape_rank))
            length = struct.unpack("q", f.read(8))[0]
            data_bytes = f.read()
            data = np.frombuffer(data_bytes, dtype=np.float32).reshape(shape).copy()
            self.tensor_index[tensor_id] = shape
            return Tensor(shape, data)

    def overwrite_tensor(self, tensor_id: str, tensor: Tensor):
        file_path = self.get_path_for_id(tensor_id)
        with open(file_path, "wb") as f:
            f.write(struct.pack("i", len(tensor.shape)))
            for dim in tensor.shape:
                f.write(struct.pack("i", int(dim)))
            f.write(struct.pack("q", int(tensor.length)))
            f.write(tensor.data.tobytes())
        self.tensor_index[tensor_id] = tensor.shape
