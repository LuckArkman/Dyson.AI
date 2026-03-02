import pyopencl as cl
import numpy as np
from typing import Dict, Optional, Any
from gpu.gpu_math_engine import GpuMathEngine

class AdamOptimizer:
    def __init__(self, t_manager: Any):
        self.t_manager = t_manager
        self._m: Dict[int, Any] = {} # Momentos (VRAM buffers)
        self._v: Dict[int, Any] = {} # RMSProp (VRAM buffers)
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        print("[AdamOptimizer] Ativado em Python/GPU.")

    def reset(self):
        self._m.clear()
        self._v.clear()
        self.t = 0

    def update_parameters_gpu(self, layer_index: int, param_buffer, grad_buffer, math_engine: GpuMathEngine, size: int):
        if layer_index not in self._m:
            # Inicializa m e v como buffers de zeros na GPU
            self._m[layer_index] = math_engine.create_buffer(size=size * 4) # float32 = 4 bytes
            self._v[layer_index] = math_engine.create_buffer(size=size * 4)
            math_engine.fill(self._m[layer_index], 0.0, size)
            math_engine.fill(self._v[layer_index], 0.0, size)

        self.t += 1
        
        # Kernel Adam Update (assumindo que o kernel está no GpuMathEngine)
        math_engine.program.adam_update(
            math_engine.queue, (size,), None,
            param_buffer, grad_buffer, self._m[layer_index], self._v[layer_index],
            np.float32(0.001), # Learning Rate default se não passado
            np.float32(self.beta1), np.float32(self.beta2),
            np.float32(self.epsilon), np.int32(self.t)
        )
