import numpy as np
from tensor_manager import save_compressed_tensor, load_compressed_tensor, ensure_v0_weights, get_layer_metadata
import os
import time

def test_lz4_compression():
    print("ZeroRAM-GEN: Validando Sprint 34 (Compressão LZ4)")
    # 1. Preparar pesos originais - Usar dados sintéticos com padrões para garantir compressão
    weights_name = "test_large_tensor"
    # Criar um tensor grande (10MB) com repetições (otimiza para LZ4)
    weights = np.tile(np.arange(100, dtype=np.float32), 1000 * 25) 
    
    # 2. Comprimir
    print(f"Comprimindo tensor de {weights.nbytes // 1024}KB...")
    save_compressed_tensor(weights_name, weights)
    
    # 3. Testar Descompressão Direta
    print("\nExecutando Descompressão LZ4...")
    start_time = time.time()
    restored_weights = load_compressed_tensor(weights_name)
    lz4_time = time.time() - start_time
    print(f"Descompressão concluída em: {lz4_time:.6f}s")
    
    # 4. Comparar com Original (Validação Lossless)
    mse = np.mean((restored_weights - weights)**2)
    print(f"Erro Quadrático Médio (MSE): {mse:.10f}")
    
    # 5. Verificar redução de tamanho real no disco
    lz4_file = os.path.join(os.path.dirname(__file__), 'weights', f"{weights_name}.lz4")
    original_size = weights.nbytes
    compressed_size = os.path.getsize(lz4_file)
    reduction = (1 - compressed_size / original_size) * 100
    
    print(f"Tamanho Original: {original_size // 1024}KB")
    print(f"Tamanho Comprimido: {compressed_size // 1024}KB")
    print(f"Redução Real: {reduction:.2f}%")
    
    if mse == 0 and reduction > 5:
        print("[OK] Sprint 34 validada. Compressão sem perdas (lossless) ativa.")
    else:
        print(f"[!] Falha na validação de compressão. Redução: {reduction:.2f}%")

if __name__ == "__main__":
    test_lz4_compression()
