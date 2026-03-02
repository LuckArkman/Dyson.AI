import os
import numpy as np
from database_manager import init_db, get_or_create_id, log_training_metrics, get_db_connection
from tensor_manager import ensure_v0_weights, dispose_tensor
from engine import embedding_lookup, dense_layer_forward, compute_softmax, calculate_loss

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 08 (Funções de Perda e Log)")
    
    # 0. Garantir existência das Tabelas e Pesos
    init_db()
    ensure_v0_weights()
    
    # 1. Definir entrada e Alvo (Target) para treino simulado
    tokens_input = ["olá", "!", "como", "você"]
    target_tokens = ["você", "está", "hoje", "?"] # Próximos tokens esperados
    
    input_ids = [get_or_create_id(t) for t in tokens_input]
    target_ids = [get_or_create_id(t) for t in target_tokens]
    
    print(f"Sequence Input: {tokens_input}")
    print(f"Target IDs: {target_ids}")
    
    # 2. Forward Pass Completo (Zero RAM)
    emb = embedding_lookup(input_ids)
    h1 = dense_layer_forward(emb, "hidden_01_weights", "hidden_01_bias", activation='relu')
    logits = dense_layer_forward(h1, "output_weights", activation='linear')
    
    # 3. Softmax & Loss (Sprint 08)
    print("\nCalculando Softmax e Loss...")
    probs = compute_softmax(logits)
    loss = calculate_loss(probs, target_ids)
    
    print(f"Loss Calculada: {loss:.6f}")
    
    # 4. Logging no SQLite (Sprint 08)
    print("Gravando métricas no SQLite...")
    log_training_metrics(epoch=0, step=1, loss=loss)
    
    # 5. Validação do Log
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM train_log ORDER BY timestamp DESC LIMIT 1")
        last_log = cursor.fetchone()
        
    print("\nValidação do Log no Banco:")
    print(f" - Timestamp: {last_log[0]}")
    print(f" - Epoch: {last_log[1]} | Step: {last_log[2]} | Loss: {last_log[3]:.6f}")

    # Limpeza
    dispose_tensor(emb); dispose_tensor(h1); dispose_tensor(logits)

    print("\nSprint 08 Concluída com Sucesso: Função de Perda e Subsistema de Log validados.")

if __name__ == "__main__":
    main()
