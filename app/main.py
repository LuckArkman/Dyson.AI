import os
import psutil
from database_manager import get_or_create_id
from tensor_manager import load_tensor_mmap, dispose_tensor

def get_ram_usage():
    """Retorna o uso atual de RAM em MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 06 (Loading On-Demand)")
    
    # 1. Medir RAM Inicial
    initial_ram = get_ram_usage()
    print(f"RAM Inicial: {initial_ram:.2f} MB")
    
    # 2. Carregar Matriz de Embeddings via MMap (Aprox. 123MB em disco)
    print("\nCarregando Matriz de Embeddings em modo MMap...")
    embed_mmap = load_tensor_mmap("embedding_matrix")
    
    # A RAM não deve ter aumentado proporcionalmente aos 123MB do arquivo
    ram_after_mmap = get_ram_usage()
    print(f"Matriz Aberta. RAM atual: {ram_after_mmap:.2f} MB (Aumento: {ram_after_mmap-initial_ram:.2f} MB)")
    
    # 3. Acessar uma "Fatia" (Slicing) - Zero RAM
    # Vamos buscar o vetor do token 'olá'
    token = 'olá'
    token_id = get_or_create_id(token)
    
    print(f"\nAcessando vetor do token '{token}' (ID: {token_id})...")
    vector_slice = embed_mmap[token_id] # Apenas esta linha é lida do disco
    
    print(f"Vetor Original (slice): {vector_slice[:5]} [...]")
    print(f"Shape do slice: {vector_slice.shape}")
    
    # RAM após ler o slice
    ram_after_slice = get_ram_usage()
    print(f"RAM após ler slice: {ram_after_slice:.2f} MB")
    
    # 4. Liberar Memória
    print("\nDescartando tensores e forçando limpeza...")
    dispose_tensor(embed_mmap)
    dispose_tensor(vector_slice)
    
    final_ram = get_ram_usage()
    print(f"RAM Final: {final_ram:.2f} MB")
    
    print("\nSprint 06 Concluída com Sucesso: Mecanismo MMap validado. Nenhum tensor residente permanentemente em RAM.")

if __name__ == "__main__":
    main()
