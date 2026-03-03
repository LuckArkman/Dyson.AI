import numpy as np
import time
import os
import shutil
from database_manager import init_db, get_text_by_id, get_or_create_id, log_telemetry, get_db_connection
from tensor_manager import (
    ensure_v0_weights, create_tensor_shards, save_compressed_tensor, 
    load_tensor_mmap, WEIGHTS_DIR, REGISTRY_PATH
)
from engine import embedding_lookup, dense_layer_forward, compute_softmax, apply_behavioral_bias

def run_stable_inference_demo(prompt="The ZeroRAM engine is now stable"):
    print(f"--- ZeroRAM-GEN v1.0 [STABLE] Inference Demo ---")
    
    # 1. Limpeza TOTAL para o Demo v1.0 (Vocab + Pesos)
    print("[CLEAN] Resetando ambiente para estado inicial estável...")
    DB_PATH = os.path.join(os.path.dirname(__file__), 'Dayson', 'vocab.db')
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(WEIGHTS_DIR):
        shutil.rmtree(WEIGHTS_DIR)
    
    init_db()
    ensure_v0_weights()
    
    # Após o reset, o vocab_size será pequeno (apenas tokens especiais + tokens do prompt)
    # 2. Tokenização do Prompt (Isso vai popular o vocab)
    tokens = prompt.split()
    token_ids = [get_or_create_id(word) for word in tokens]
    print(f"[TOKEN] IDs: {token_ids} | Prompt: '{prompt}'")
    
    # 3. Preparar Sharding e Compressão com o tamanho CORRETO
    # Nota: No ZeroRAM Real, o vocab cresce e a matriz de embeddings seria expandida ou teria um buffer.
    # Aqui vamos recriar os pesos uma última vez para bater com o vocab populado.
    if os.path.exists(WEIGHTS_DIR):
        shutil.rmtree(WEIGHTS_DIR)
    ensure_v0_weights()
    
    print("[INIT] Preparando Sharding e LZ4...")
    embed_matrix = load_tensor_mmap("embedding_matrix")
    create_tensor_shards("embedding_matrix", embed_matrix, ids_per_shard=10) # Shards pequenos para o demo
    
    hidden_weights = load_tensor_mmap("hidden_01_weights")
    save_compressed_tensor("hidden_01_weights", hidden_weights)
    
    # 4. Início da Inferência
    start_time = time.time()
    
    # Embedding Lookup (via Shards)
    embeddings = embedding_lookup(token_ids)
    input_vector = np.mean(embeddings, axis=0, keepdims=True)
    print(f"[EMBED] Lookup via Shards concluído (Zero RAM).")
    
    # Forward Pass - Hidden (LZ4)
    hidden_out = dense_layer_forward(input_vector, "hidden_01_weights", "hidden_01_bias", activation='relu')
    
    # Forward Pass - Output (FP16 MMap)
    logits = dense_layer_forward(hidden_out, "output_weights", activation='linear')
    
    # Softmax
    probs = compute_softmax(logits)
    next_token_id = np.argmax(probs[0])
    next_word = get_text_by_id(next_token_id)
    
    latency = time.time() - start_time
    print(f"\n[RESULTADO DA INFERÊNCIA v1.0]")
    print(f" -> Próximo token previsto: '{next_word}'")
    print(f" -> Latência Total Estável: {latency:.4f}s")
    
    log_telemetry('stable_v1_demo', latency, "fresh_install:true")
    print(f"\n--- Demo Finalizado com Sucesso ---")

if __name__ == "__main__":
    run_stable_inference_demo()
