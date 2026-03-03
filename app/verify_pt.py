from tensor_manager import load_model_checkpoint
from engine import (
    embedding_lookup, dense_layer_forward, compute_softmax
)
from database_manager import get_or_create_id, get_text_by_id, init_db
import numpy as np

def verify_pt_finetune():
    print("--- Verificação Fine-tuning Português (v1.1_pt) ---")
    
    # 1. Carregar Checkpoint PT
    print("[LOAD] Restaurando Checkpoint 'v1.1_pt'...")
    if not load_model_checkpoint("v1.1_pt"):
        print("[ERR] Checkpoint 'v1.1_pt' não encontrado.")
        return
    
    # 2. Teste de Inferência em PT
    # Prompts em Português
    prompts = [
        "A economia Dyson",
        "O ZeroRAM-GEN permite",
        "O treinamento em português"
    ]
    
    for prompt in prompts:
        print(f"\n[TESTE] Prompt: '{prompt}'")
        from tokenizer import tokenize_line
        tokens = tokenize_line(prompt)
        token_ids = [get_or_create_id(w) for w in tokens]
        
        # Forward Pass
        emb = embedding_lookup(token_ids)
        x = np.mean(emb, axis=0, keepdims=True)
        h1 = dense_layer_forward(x, "hidden_01_weights", "hidden_01_bias", activation='relu')
        logits = dense_layer_forward(h1, "output_weights", activation='linear')
        probs = compute_softmax(logits)
        
        next_id = np.argmax(probs[0])
        next_word = get_text_by_id(next_id)
        
        print(f" -> Próximo token: '{next_word}' (ID: {next_id})")

    print("\n[OK] Modelo adaptado para PT validado.")

if __name__ == "__main__":
    init_db()
    verify_pt_finetune()
