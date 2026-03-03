import os
import numpy as np
from database_manager import init_db, get_or_create_id, get_db_connection
from tensor_manager import (
    ensure_v0_weights, store_tensor_disk, load_tensor_disk, 
    dispose_tensor, reset_accumulated_grads, WEIGHTS_DIR
)
from engine import (
    embedding_lookup, dense_layer_forward, apply_activation, 
    compute_softmax, calculate_loss, compute_output_gradient, 
    backward_layer_step, d_relu, accumulate_embedding_grad
)

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 10 (Conclusão do Backward Pass)")
    
    # 0. Setup e Limpeza
    init_db()
    ensure_v0_weights()
    reset_accumulated_grads()
    
    # 1. Forward Pass (Salvando estados no disco)
    # ----------------------------------------
    tokens_input = ["olá", "!", "como", "você"]
    target_tokens = ["você", "está", "hoje", "?"]
    
    input_ids = [get_or_create_id(t) for t in tokens_input]
    target_ids = [get_or_create_id(t) for t in target_tokens]
    
    print("\n[Forward] Processando sequência...")
    # Camada 1: Embedding
    emb = embedding_lookup(input_ids)
    store_tensor_disk("forward_emb", emb)
    
    # Camada 2: Hidden
    h1_z = dense_layer_forward(emb, "hidden_01_weights", "hidden_01_bias", activation='linear')
    h1_act = apply_activation(h1_z, 'relu')
    store_tensor_disk("forward_h1_act", h1_act)
    store_tensor_disk("forward_h1_z", h1_z)
    
    # Camada 3: Output
    logits = dense_layer_forward(h1_act, "output_weights", activation='linear')
    probs = compute_softmax(logits)
    loss = calculate_loss(probs, target_ids)
    print(f"Loss Inicial: {loss:.6f}")
    
    # 2. Backward Pass (Caminho inverso até o Embedding)
    # ------------------------------------------------
    print("\n[Backward] Iniciando retropropagação total...")
    
    # Passo 1: Gradiente na Saída
    grad_output = compute_output_gradient(probs, target_ids)
    
    # Passo 2: Output Weights -> Hidden Act
    h1_act_from_disk = load_tensor_disk("forward_h1_act")
    print(" -> Calculando gradientes para 'output_weights'...")
    grad_w_out, grad_b_out, grad_h1_act = backward_layer_step(
        grad_output, "output_weights", h1_act_from_disk
    )
    store_tensor_disk("output_weights", grad_w_out, folder='grads')
    
    # Passo 3: Hidden Act -> Hidden Z (ReLU Derivative)
    h1_z_from_disk = load_tensor_disk("forward_h1_z")
    grad_h1_z = grad_h1_act * d_relu(h1_z_from_disk)
    
    # Passo 4: Hidden Weights & Bias -> Embedding
    emb_from_disk = load_tensor_disk("forward_emb")
    print(" -> Calculando gradientes para 'hidden_01_weights' e 'bias'...")
    grad_w_h1, grad_b_h1, grad_emb = backward_layer_step(
        grad_h1_z, "hidden_01_weights", emb_from_disk
    )
    store_tensor_disk("hidden_01_weights", grad_w_h1, folder='grads')
    store_tensor_disk("hidden_01_bias", grad_b_h1, folder='grads')
    
    # Passo 5: Embedding Matrix Gradient
    print(" -> Calculando gradientes para a matriz de Embeddings...")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM vocab")
        vocab_size = cursor.fetchone()[0]
    
    grad_embedding_matrix = accumulate_embedding_grad(
        input_ids, grad_emb, vocab_size, emb.shape[1]
    )
    store_tensor_disk("embedding_matrix", grad_embedding_matrix, folder='grads')
    
    # 3. Validação final
    print("\nValidando arquivos de gradientes gerados:")
    for name in ["output_weights", "hidden_01_weights", "hidden_01_bias", "embedding_matrix"]:
        path = os.path.join(WEIGHTS_DIR, "grads", f"{name}.npy")
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f" [OK] {name}.grad: {size:.2f} KB")
        else:
            print(f" [ERRO] {name}.grad não encontrado!")
            
    print("\nMagnitude dos gradientes de Embedding (Tokens ativos):")
    # Mostrar magnitude média apenas para as linhas que receberam gradiente
    active_grads = grad_embedding_matrix[input_ids]
    print(f" - Magnitude média (Active Embeds): {np.abs(active_grads).mean():.10f}")

    # Limpeza de tensores em RAM
    dispose_tensor(emb); dispose_tensor(h1_z); dispose_tensor(h1_act); dispose_tensor(logits)
    dispose_tensor(grad_output); dispose_tensor(grad_w_out); dispose_tensor(grad_b_out)
    dispose_tensor(grad_h1_act); dispose_tensor(grad_w_h1); dispose_tensor(grad_b_h1)
    dispose_tensor(grad_embedding_matrix)

    print("\nSprint 10 Concluída com Sucesso: Retropropagação concluída até os Embeddings.")

if __name__ == "__main__":
    main()
