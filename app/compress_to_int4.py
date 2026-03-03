import os
import json
import numpy as np
from tensor_manager import (
    WEIGHTS_DIR, REGISTRY_PATH, get_layer_metadata, 
    save_quantized_int4_tensor
)

def compress_model_to_int4():
    """
    Varre o registro de pesos e converte todas as camadas densas e de embedding para INT4.
    """
    if not os.path.exists(REGISTRY_PATH):
        print("[ERRO] Registro de pesos não encontrado.")
        return

    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)

    print("========================================")
    print("   ZeroRAM-GEN: Compressão Extrema INT4   ")
    print("========================================")

    for name, meta in registry['layers'].items():
        # Apenas processar se não for bias (bias costuma ser pequeno, manter maior precisão)
        if "_bias" in name:
            continue
            
        print(f"\nProcessando: {name}...")
        path = meta['path']
        if not os.path.exists(path):
            print(f" [!] Arquivo não encontrado: {path}")
            continue

        # Carregar pesos atuais
        weights = np.load(path)
        
        # Se já estiver quantizado, dequantizar primeiro para melhor precisão na conversão
        if meta.get('quantized'):
            from tensor_manager import get_quant_params, dequantize_from_int8, dequantize_from_int4
            q_params = get_quant_params(name)
            if meta['quantized'] == "int8":
                weights = dequantize_from_int8(weights, q_params['scale'], q_params['zero_point'])
            elif meta['quantized'] == "int4":
                print(" [SKIP] Já está em INT4.")
                continue

        # Salvar como INT4
        save_quantized_int4_tensor(name, weights)
        
        # Atualizar metadados no registro
        meta['path'] = os.path.join(WEIGHTS_DIR, f"{name}_int4.npy")
        meta['dtype'] = 'int4'
        meta['quantized'] = 'int4'
        meta['shape'] = list(weights.shape)

    # Salvar registro atualizado
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=4)

    print("\n========================================")
    print("   Compressão INT4 Concluída com Sucesso! ")
    print("========================================")

if __name__ == "__main__":
    compress_model_to_int4()
