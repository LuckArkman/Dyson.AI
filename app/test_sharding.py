import numpy as np
from tensor_manager import create_tensor_shards, ensure_v0_weights, get_layer_metadata
from engine import embedding_lookup
from database_manager import init_db
import os
import time

def test_sharding():
    print("ZeroRAM-GEN: Validando Sprint 35 (Sharding de Tensores)")
    init_db()
    ensure_v0_weights()
    
    # 1. Carregar Embedding Matrix original
    weights_name = "embedding_matrix"
    meta = get_layer_metadata(weights_name)
    original_weights = np.load(meta['path'])
    
    # 2. Fragmentar em shards de 5000 elementos
    print(f"Fragmentando {weights_name} em shards de 5000...")
    create_tensor_shards(weights_name, original_weights, ids_per_shard=5000)
    
    # 3. Testar Lookup de IDs em diferentes shards
    # IDs: 10 (Shard 0), 6000 (Shard 1), 12000 (Shard 2)
    test_ids = [10, 6000, 12000]
    
    print(f"\nRealizando Lookup Shard-Aware para IDs: {test_ids}...")
    start_time = time.time()
    vectors = embedding_lookup(test_ids)
    latency = time.time() - start_time
    
    print(f"Lookup concluído em {latency:.6f}s")
    
    # 4. Validar integridade dos dados
    for i, tid in enumerate(test_ids):
        mse = np.mean((vectors[i] - original_weights[tid])**2)
        print(f"ID {tid}: MSE vs Original = {mse:.10f}")
        if mse > 0:
            print(f"[!] Erro de integridade no ID {tid}")
            return
            
    # 5. Verificar se os diretórios de shards existem
    shard_dir = os.path.join(os.path.dirname(__file__), 'weights', f"{weights_name}_shards")
    if os.path.exists(shard_dir) and len(os.listdir(shard_dir)) > 1:
        print(f"\n[OK] Diretório de shards verificado: {shard_dir}")
        print(f"Total de fragmentos: {len(os.listdir(shard_dir))}")
    
    print("\n[OK] Sprint 35 validada. O sistema agora suporta segmentação física de tensores.")

if __name__ == "__main__":
    test_sharding()
