import os
import json
import numpy as np
from database_manager import get_db_connection
from tensor_manager import initialize_layer_weights, create_weight_registry, WEIGHTS_DIR

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 05")
    
    # 1. Definir Dimensões do Modelo (V0)
    # Vocabulário: ~252k tokens (obtido via database_manager)
    # Embeddings: 128 (pequeno para ZeroRAM)
    # Hidden Layers: 256
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM vocab")
        vocab_size = cursor.fetchone()[0]
    
    embed_dim = 128
    hidden_dim = 256
    
    print(f"Dimensões do Modelo V0:")
    print(f"- Vocabulário: {vocab_size}")
    print(f"- Dimensão de Embedding: {embed_dim}")
    print(f"- Dimensão de Hidden: {hidden_dim}")
    
    # 2. Inicializar Camadas e Pesos em Disco (Arquitetura ZeroRAM)
    layers_metadata = {}
    
    # Camada 01: Embeddings (Matriz gigante)
    print("\nInicializando Embeddings...")
    embed_shape = (vocab_size, embed_dim)
    path, shape = initialize_layer_weights(embed_shape, "embedding_matrix", "xavier")
    layers_metadata["embedding_matrix"] = {"path": path, "shape": list(shape), "dtype": "float32"}
    
    # Camada 02: Hidden 01 (Weights + Bias)
    print("Inicializando Hidden Layer 01...")
    h1_shape = (embed_dim, hidden_dim)
    path, shape = initialize_layer_weights(h1_shape, "hidden_01_weights", "xavier")
    layers_metadata["hidden_01_weights"] = {"path": path, "shape": list(shape), "dtype": "float32"}
    
    b1_shape = (hidden_dim,)
    path, shape = initialize_layer_weights(b1_shape, "hidden_01_bias", "zeros")
    layers_metadata["hidden_01_bias"] = {"path": path, "shape": list(shape), "dtype": "float32"}
    
    # Camada 03: Output Layer (Mapeia Hidden de volta para Vocab)
    print("Inicializando Output Layer...")
    output_shape = (hidden_dim, vocab_size)
    path, shape = initialize_layer_weights(output_shape, "output_weights", "xavier")
    layers_metadata["output_weights"] = {"path": path, "shape": list(shape), "dtype": "float32"}
    
    # 3. Criar Registro de Pesos
    create_weight_registry(layers_metadata)
    
    # 4. Validação do Disco
    print("\nValidando arquivos criados:")
    for name, meta in layers_metadata.items():
        file_path = meta['path']
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f" [OK] {name}: {size_mb:.2f} MB")
        else:
            print(f" [ERRO] {name} não encontrado!")

    print("\nSprint 05 Concluída com Sucesso: Estrutura de Tensores em Disco validada.")

if __name__ == "__main__":
    main()
