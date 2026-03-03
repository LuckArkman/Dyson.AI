from tensor_manager import save_svd_weights, ensure_v0_weights, load_tensor_mmap, get_layer_metadata, REGISTRY_PATH, WEIGHTS_DIR
from database_manager import init_db
from engine import dense_layer_forward
import numpy as np
import os
import shutil

def test_svd():
    print("ZeroRAM-GEN: Validando Sprint 33 (Decomposição SVD)")
    # Force reset weights to have clean FP32
    if os.path.exists(WEIGHTS_DIR):
        shutil.rmtree(WEIGHTS_DIR)
    
    init_db()
    ensure_v0_weights()
    
    # 1. Carregar pesos originais da camada oculta (Agora garantido ser FP32/Xavier)
    weights_name = "hidden_01_weights"
    meta = get_layer_metadata(weights_name)
    weights = np.load(meta['path']).astype(np.float32)
    
    # 2. Aplicar SVD (Rank 80% para melhor fidelidade em pesos aleatórios)
    print(f"Aplicando SVD em {weights_name} (Shape: {weights.shape})...")
    save_svd_weights(weights_name, weights, rank_ratio=0.8)
    
    # 3. Testar Forward com SVD
    input_test = np.random.randn(1, 128).astype(np.float32)
    
    print("\nExecutando Forward via Aproximação SVD...")
    output_svd = dense_layer_forward(input_test, weights_name, activation='linear')
    
    # 4. Comparar com Forward original
    print("Executando Forward Original para comparação...")
    output_original = np.dot(input_test, weights)
    
    # Calcular Erro (MSE)
    mse = np.mean((output_svd - output_original)**2)
    print(f"Original Mean: {np.mean(output_original):.4f}, Std: {np.std(output_original):.4f}")
    print(f"SVD Mean: {np.mean(output_svd):.4f}, Std: {np.std(output_svd):.4f}")
    print(f"Erro Quadrático Médio (MSE SVD vs Original): {mse:.8f}")
    
    if mse < 0.1:
        print("[OK] Sprint 33 validada. O sistema suporta aproximação Low-Rank via SVD.")
    else:
        print("[!] Erro de reconstrução SVD pode estar elevado.")

if __name__ == "__main__":
    test_svd()
