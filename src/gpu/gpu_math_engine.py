import pyopencl as cl
import numpy as np
from typing import Tuple, List, Optional

class GpuMathEngine:
    def __init__(self):
        self.platforms = cl.get_platforms()
        self.device = None
        
        # Procura por uma GPU em qualquer plataforma disponível
        for platform in self.platforms:
            for device in platform.get_devices():
                if device.type == cl.device_type.GPU:
                    self.device = device
                    break
            if self.device: break
            
        # Fallback para o primeiro dispositivo disponível se nenhuma GPU for encontrada
        if not self.device:
            self.device = self.platforms[0].get_devices()[0]
            
        print(f"[GpuMathEngine] Usando dispositivo: {self.device.name}")
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        
        # Coleção COMPLETA de Kernels migrados do C#
        self.program_source = """
    __kernel void matrix_multiply(__global const float* A, __global const float* B, __global float* C, int M, int N, int P) { 
        int i = get_global_id(0); int j = get_global_id(1); 
        if (i < M && j < P) { float sum = 0.0f; for (int k = 0; k < N; ++k) { sum += A[i * N + k] * B[k * P + j]; } C[i * P + j] = sum; } 
    }
    __kernel void add(__global const float* a, __global const float* b, __global float* result) { 
        int gid = get_global_id(0); result[gid] = a[gid] + b[gid]; 
    }
    __kernel void subtract(__global const float* a, __global const float* b, __global float* result) { 
        int gid = get_global_id(0); result[gid] = a[gid] - b[gid]; 
    }
    __kernel void multiply(__global const float* a, __global const float* b, __global float* result) { 
        int gid = get_global_id(0); result[gid] = a[gid] * b[gid]; 
    }
    __kernel void fill(__global float* data, float value, int size) { 
        int gid = get_global_id(0); if(gid < size) data[gid] = value; 
    }
    __kernel void sigmoid(__global const float* a, __global float* result) {
        int gid = get_global_id(0); float input = a[gid];
        result[gid] = 1.0f / (1.0f + exp(-input));
    }
    __kernel void sigmoid_derivative(__global const float* output, __global float* result) {
        int gid = get_global_id(0); float o = output[gid];
        result[gid] = o * (1.0f - o);
    }
    __kernel void tanh_activation(__global const float* a, __global float* result) {
        int gid = get_global_id(0); result[gid] = tanh(a[gid]);
    }
    __kernel void tanh_derivative(__global const float* output, __global float* result) {
        int gid = get_global_id(0); float o = output[gid];
        result[gid] = 1.0f - o * o;
    }
    __kernel void matrix_multiply_transpose_a(__global const float* A, __global const float* B, __global float* C, int M, int K, int P) {
        int i = get_global_id(0); int j = get_global_id(1);
        if (i < M && j < P) { float sum = 0.0f; for (int k = 0; k < K; ++k) { sum += A[k * M + i] * B[k * P + j]; } C[i * P + j] = sum; }
    }
    __kernel void matrix_multiply_transpose_b(__global const float* A, __global const float* B, __global float* C, int M, int K, int P) {
        int i = get_global_id(0); int j = get_global_id(1);
        if (i < M && j < P) { float sum = 0.0f; for (int k = 0; k < K; ++k) { sum += A[i * K + k] * B[j * K + k]; } C[i * P + j] = sum; }
    }
    __kernel void adam_update(__global float* p, __global const float* g, __global float* m, __global float* v, float lr, float beta1, float beta2, float epsilon, int t) {
        int i = get_global_id(0);
        float grad = g[i];
        float m_val = beta1 * m[i] + (1.0f - beta1) * grad;
        float v_val = beta2 * v[i] + (1.0f - beta2) * (grad * grad);
        float m_hat = m_val / (1.0f - pow(beta1, (float)t));
        float v_hat = v_val / (1.0f - pow(beta2, (float)t));
        p[i] -= lr * m_hat / (sqrt(v_hat) + epsilon);
        m[i] = m_val; v[i] = v_val;
    }
    __kernel void softmax(__global const float* input, __global float* output, int size) {
        int row = get_global_id(0); int offset = row * size;
        float maxVal = -1e37; for (int i = 0; i < size; i++) if (input[offset+i] > maxVal) maxVal = input[offset+i];
        float sumExp = 0.0f; for (int i = 0; i < size; i++) { output[offset+i] = exp(input[offset+i] - maxVal); sumExp += output[offset+i]; }
        for (int i = 0; i < size; i++) output[offset+i] /= sumExp;
    }
    """
        self.program = cl.Program(self.context, self.program_source).build()

    def create_buffer(self, data: Optional[np.ndarray] = None, size: Optional[int] = None):
        if data is not None:
            return cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data.astype(np.float32))
        return cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=size)

    def matrix_multiply(self, a, b, c, m, n, p):
        self.program.matrix_multiply(self.queue, (m, p), None, a, b, c, np.int32(m), np.int32(n), np.int32(p))

    def add(self, a, b, r, size):
        self.program.add(self.queue, (size,), None, a, b, r)

    def subtract(self, a, b, r, size):
        self.program.subtract(self.queue, (size,), None, a, b, r)

    def multiply(self, a, b, r, size):
        self.program.multiply(self.queue, (size,), None, a, b, r)

    def fill(self, buf, value, size):
        self.program.fill(self.queue, (size,), None, buf, np.float32(value), np.int32(size))

    def sigmoid(self, a, r, size):
        self.program.sigmoid(self.queue, (size,), None, a, r)

    def sigmoid_derivative(self, o, r, size):
        self.program.sigmoid_derivative(self.queue, (size,), None, o, r)

    def tanh_activation(self, a, r, size):
        self.program.tanh_activation(self.queue, (size,), None, a, r)

    def tanh_derivative(self, o, r, size):
        self.program.tanh_derivative(self.queue, (size,), None, o, r)

    def matrix_multiply_transpose_a(self, a, b, c, m, k, p):
        self.program.matrix_multiply_transpose_a(self.queue, (m, p), None, a, b, c, np.int32(m), np.int32(k), np.int32(p))

    def matrix_multiply_transpose_b(self, a, b, c, m, k, p):
        self.program.matrix_multiply_transpose_b(self.queue, (m, p), None, a, b, c, np.int32(m), np.int32(k), np.int32(p))

    def softmax(self, i, o, batch_size, vocab_size):
        self.program.softmax(self.queue, (batch_size,), None, i, o, np.int32(vocab_size))

    def release_buffer(self, buffer):
        if buffer:
            buffer.release()

    def synchronize(self):
        self.queue.finish()

