import numpy as np
from tensor_manager import store_tensor_disk, load_tensor_disk
from database_manager import init_db
import os

def test_activation_quantization():
    print("ZeroRAM-GEN: Validando Sprint 32 (Quantização de Ativações)")
    init_db()
    
    # 1. Criar um tensor de ativação simulado (range normal de ativações)
    activation = np.random.randn(10, 128).astype(np.float32)
    
    # 2. Salvar com quantização
    print("Salvando ativação quantizada no disco...")
    store_tensor_disk("h1_act", activation, folder='temp_test', quantize=True)
    
    # 3. Carregar e comparar (deve dequantizar automaticamente)
    print("Carregando e dequantizando automaticamente...")
    restored = load_tensor_disk("h1_act", folder='temp_test')
    
    # Calcular erro
    mse = np.mean((activation - restored)**2)
    print(f"Erro Quadrático Médio (MSE): {mse:.8f}")
    
    # 4. Verificar se o arquivo quantizado existe e é menor
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights', 'temp_test')
    q_file = os.path.join(weights_dir, "h1_act_q.npy")
    
    if os.path.exists(q_file):
        print(f"[OK] Arquivo quantizado gerado com sucesso.")
        q_size = os.path.getsize(q_file)
        # Comparação teórica com float32 (4 bytes por elemento)
        original_size_est = activation.nbytes
        reduction = (1 - q_size / original_size_est) * 100
        print(f"Redução de tamanho estimada: {reduction:.2f}%")
        
    if mse < 0.001:
        print("[OK] Sprint 32 validada. Ativações agora podem ser compactadas no disco.")
    else:
        print("[!] Erro de quantização muito alto.")

if __name__ == "__main__":
    test_activation_quantization()
