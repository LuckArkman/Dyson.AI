import os
import numpy as np
import json
from tensor_manager import (
    load_model_checkpoint, save_model_checkpoint, 
    quantize_to_int4, WEIGHTS_DIR, REGISTRY_PATH
)

def compress_checkpoint_v11():
    print("--- Dyson Optimizer: Compressão INT4 para v1.1_pt ---")
    
    # 1. Carregar checkpoint base
    if not load_model_checkpoint("v1.1_pt"):
        print("[ERR] Checkpoint 'v1.1_pt' não encontrado.")
        return

    # 2. Ler o registro de pesos
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)

    layers = registry['layers']
    
    # 3. Quantizar camadas pesadas (Embedding e Output)
    for name in ["embedding_matrix", "output_weights"]:
        if name in layers:
            meta = layers[name]
            print(f"[COMPRESS] Quantizando '{name}' para 4 bits...")
            
            # Carregar pesos FP
            weights = np.load(meta['path'])
            original_size = weights.nbytes
            
            # Quantizar INT4
            packed, scale, zp = quantize_to_int4(weights)
            
            # Salvar novo arquivo .int4.npy
            int4_path = meta['path'].replace(".npy", ".int4.npy")
            np.save(int4_path, packed)
            
            # Atualizar metadados
            meta['quantized'] = "int4"
            meta['scale'] = float(scale)
            meta['zero_point'] = int(zp)
            meta['path'] = int4_path
            
            # Remover original se existir
            old_path = meta['path'].replace(".int4", ".npy")
            if os.path.exists(old_path):
                # Importante: Garantir que não há handles abertos (o mm_obj do numpy.load)
                del weights
                import gc
                gc.collect()
                try:
                    os.remove(old_path)
                except Exception as e:
                    print(f"[WARN] Não foi possível remover o arquivo original: {e}")
            
            new_size = packed.nbytes
            reduction = (1 - new_size/original_size) * 100
            print(f" -> Tamanho original: {original_size/1024/1024:.2f} MB")
            print(f" -> Novo tamanho: {new_size/1024/1024:.2f} MB")
            print(f" -> Redução: {reduction:.1f}%")

    # 4. Atualizar Registro Global
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=4)

    # 5. Salvar como novo Checkpoint
    print("\n[STRETP] Salvando Checkpoint Otimizado 'v1.2_compressed'...")
    save_model_checkpoint("v1.2_compressed")
    print("--- Compressão Concluída ---")

if __name__ == "__main__":
    compress_checkpoint_v11()
