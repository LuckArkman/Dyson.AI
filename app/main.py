import os
import psutil
from database_manager import get_or_create_id
from tensor_manager import ensure_v0_weights, dispose_tensor
from engine import embedding_lookup, dense_layer_forward, apply_activation

def get_ram_usage():
    """Retorna o uso atual de RAM em MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 07 (Motor Matemático Forward)")
    
    # 0. Garantir existência dos pesos
    ensure_v0_weights()
    
    # 1. Preparar entrada (Prompt)
    frase_original = "Olá! Como você está?"
    print(f"\nPrompt: '{frase_original}'")
    
    # Tokenizar e converter para IDs
    tokens = ["olá", "!", "como", "você", "está", "?"]
    token_ids = [get_or_create_id(t) for t in tokens]
    print(f"Token IDs: {token_ids}")
    
    # Medir RAM Inicial
    initial_ram = get_ram_usage()
    print(f"RAM Inicial: {initial_ram:.2f} MB")
    
    # 2. Etapa 01: Embedding Lookup (Mapeamento de IDs para Vetores)
    print("\n[Forward Step 01] Embedding Lookup...")
    embeddings = embedding_lookup(token_ids)
    print(f" Shape Embeddings: {embeddings.shape}")
    print(f" RAM: {get_ram_usage():.2f} MB (Delta: {get_ram_usage()-initial_ram:.2f} MB)")
    
    # 3. Etapa 02: Hidden Layer 01 (Dense Pass)
    print("\n[Forward Step 02] Hidden Layer (Dense + ReLU)...")
    hidden_output = dense_layer_forward(
        embeddings, 
        "hidden_01_weights", 
        "hidden_01_bias", 
        activation='relu'
    )
    print(f" Shape Hidden: {hidden_output.shape}")
    print(f" RAM: {get_ram_usage():.2f} MB")
    
    # 4. Etapa 03: Output Layer (Logits para Vocab)
    print("\n[Forward Step 03] Output Layer (Final Logits)...")
    logits = dense_layer_forward(
        hidden_output, 
        "output_weights", 
        bias_name=None, # Sem bias na output nesta V0
        activation='softmax_ready' # Deixa linear para Softmax posterior
    )
    print(f" Shape Logits: {logits.shape}")
    print(f" RAM Final: {get_ram_usage():.2f} MB")
    
    # 5. Finalização
    print("\nValidando Resultados:")
    print(f" - Primeiro Logit do Primeiro Token: {logits[0][0]:.6f}")
    
    # Limpeza total
    dispose_tensor(embeddings)
    dispose_tensor(hidden_output)
    dispose_tensor(logits)
    
    print(f"RAM após descarte: {get_ram_usage():.2f} MB")
    print("\nSprint 07 Concluída com Sucesso: Motor Matemático Forward operando em disco.")

if __name__ == "__main__":
    main()
