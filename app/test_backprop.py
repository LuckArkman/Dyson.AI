import numpy as np
from engine import dense_layer_forward, dense_layer_backward
from tensor_manager import ensure_v0_weights, reset_accumulated_grads
import os

def test_backprop():
    print("--- ZeroRAM-GEN: Testando Backpropagation (Sprint 48) ---")
    ensure_v0_weights()
    reset_accumulated_grads()
    
    # 1. Forward Pass
    x = np.random.randn(1, 128).astype(np.float16)
    out = dense_layer_forward(x, "hidden_01_weights", "hidden_01_bias", activation='relu')
    print(f"[FORWARD] Output shape: {out.shape}")
    
    # 2. Backward Pass (Simulando dout como gradientes aleatórios da próxima camada)
    dout = np.random.randn(1, 256).astype(np.float16)
    din = dense_layer_backward(dout, x, "hidden_01_weights")
    print(f"[BACKWARD] Input Grad (din) shape: {din.shape}")
    
    # 3. Verificar gradientes no disco (Zero RAM Principle)
    grad_dir = os.path.join("weights", "grads")
    if os.path.exists(grad_dir):
        grads = os.listdir(grad_dir)
        print(f"[DISK] Gradientes persistidos em '{grad_dir}': {grads}")
        
        if any("hidden_01_weights_dw" in g for g in grads):
            print("[OK] Gradientes de peso salvos corretamente no disco.")
    else:
        print("[ERRO] Diretório de gradientes não encontrado.")

if __name__ == "__main__":
    test_backprop()
