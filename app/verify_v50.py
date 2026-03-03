from tensor_manager import load_model_checkpoint
from engine import (
    embedding_lookup, dense_layer_forward, compute_softmax
)
from database_manager import get_or_create_id, get_text_by_id, init_db
import numpy as np

def verify_sprint_50():
    print("--- Verificação Final da Sprint 50 ---")
    
    # 1. Carregar o checkpoint treinado
    print("[LOAD] Restaurando Checkpoint 'v1.0_trained'...")
    if not load_model_checkpoint("v1.0_trained"):
        print("[ERR] Checkpoint não encontrado.")
        return
    
    # 2. Teste de Inferência Real (Usando palavras conhecidas do vocab < ID 64)
    prompt = "ZeroRAM engine is"
    print(f"[TESTE] Prompt: '{prompt}'")
    
    token_ids = [get_or_create_id(w) for w in prompt.split()]
    
    # Forward Pass
    emb = embedding_lookup(token_ids)
    x = np.mean(emb, axis=0, keepdims=True)
    h1 = dense_layer_forward(x, "hidden_01_weights", "hidden_01_bias", activation='relu')
    logits = dense_layer_forward(h1, "output_weights", activation='linear')
    probs = compute_softmax(logits)
    
    next_id = np.argmax(probs[0])
    next_word = get_text_by_id(next_id)
    
    print(f"\n[RESULTADO]")
    print(f" -> Próximo token: '{next_word}' (ID: {next_id})")
    print("\n[OK] Modelo treinado e restaurado com sucesso.")

if __name__ == "__main__":
    init_db()
    verify_sprint_50()
