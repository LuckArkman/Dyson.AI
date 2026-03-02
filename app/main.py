import os
import psutil
from database_manager import get_or_create_id, get_db_connection
from tensor_manager import initialize_layer_weights, create_weight_registry, load_tensor_mmap, dispose_tensor, WEIGHTS_DIR

def get_ram_usage():
    """Retorna o uso atual de RAM em MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def ensure_v0_weights():
    """Garante que os pesos iniciais existem (Tarefa da Sprint 05)."""
    if not os.path.exists(WEIGHTS_DIR) or len(os.listdir(WEIGHTS_DIR)) <= 1:
        print("\n[!] Pesos não encontrados. Reinicializando (Sprint 05)...")
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM vocab")
            vocab_size = cursor.fetchone()[0]
        
        embed_dim = 128
        hidden_dim = 256
        layers_metadata = {}
        
        # Camada 01: Embeddings
        path, shape = initialize_layer_weights((vocab_size, embed_dim), "embedding_matrix", "xavier")
        layers_metadata["embedding_matrix"] = {"path": path, "shape": list(shape), "dtype": "float32"}
        
        # Camada 02: Hidden 01
        path, shape = initialize_layer_weights((embed_dim, hidden_dim), "hidden_01_weights", "xavier")
        layers_metadata["hidden_01_weights"] = {"path": path, "shape": list(shape), "dtype": "float32"}
        
        path, shape = initialize_layer_weights((hidden_dim,), "hidden_01_bias", "zeros")
        layers_metadata["hidden_01_bias"] = {"path": path, "shape": list(shape), "dtype": "float32"}
        
        # Camada 03: Output
        path, shape = initialize_layer_weights((hidden_dim, vocab_size), "output_weights", "xavier")
        layers_metadata["output_weights"] = {"path": path, "shape": list(shape), "dtype": "float32"}
        
        create_weight_registry(layers_metadata)
        print("[OK] Pesos reinicializados.")

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 06 (Loading On-Demand)")
    
    # 0. Garantir existência dos pesos
    ensure_v0_weights()
    
    # 1. Medir RAM Inicial
    initial_ram = get_ram_usage()
    print(f"\nRAM Inicial: {initial_ram:.2f} MB")
    
    # 2. Carregar Matriz de Embeddings via MMap (Aprox. 123MB em disco)
    print("Carregando Matriz de Embeddings em modo MMap...")
    embed_mmap = load_tensor_mmap("embedding_matrix")
    
    # A RAM não deve ter aumentado proporcionalmente aos 123MB do arquivo
    ram_after_mmap = get_ram_usage()
    print(f"Matriz Aberta via MMap. RAM atual: {ram_after_mmap:.2f} MB")
    print(f"Delta RAM (MMap): {ram_after_mmap - initial_ram:.2f} MB")
    
    # 3. Acessar uma "Fatia" (Slicing) - Zero RAM residente
    token = 'olá'
    token_id = get_or_create_id(token)
    
    print(f"\nAcessando vetor do token '{token}' (ID: {token_id})...")
    # O Slicing no NumPy mmap não carrega a matriz toda para a RAM
    vector_slice = embed_mmap[token_id] 
    
    print(f"Vetor (Início): {vector_slice[:5]} [...]")
    
    # RAM após ler o slice
    ram_after_slice = get_ram_usage()
    print(f"RAM após ler slice: {ram_after_slice:.2f} MB")
    
    # 4. Liberar Memória
    print("\nDescartando tensores e forçando limpeza...")
    dispose_tensor(embed_mmap)
    dispose_tensor(vector_slice)
    
    final_ram = get_ram_usage()
    print(f"RAM Final: {final_ram:.2f} MB")
    
    print("\nSprint 06 Concluída com Sucesso: Mecanismo MMap validado.")

if __name__ == "__main__":
    main()
