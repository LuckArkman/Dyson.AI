import os
import numpy as np
from database_manager import init_db, get_or_create_id
from tensor_manager import ensure_v0_weights, store_tensor_disk, load_tensor_disk, dispose_tensor
from engine import (
    embedding_lookup, dense_layer_forward, compute_softmax, 
    calculate_loss, compute_output_gradient, backward_layer_step, d_relu
)

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 09 (Backward Pass Inicial)")
    
    # 0. Setup
    init_db()
    ensure_v0_weights()
    
    # 1. Forward Pass com Persistência de Ativações (Simulação de Treino)
    tokens_input = ["olá", "!", "como", "você"]
    target_tokens = ["você", "está", "hoje", "?"]
    
    input_ids = [get_or_create_id(t) for t in tokens_input]
    target_ids = [get_or_create_id(t) for t in target_tokens]
    
    print("\nExecutando Forward Pass e salvando ativações...")
    
    # Camada 1: Embedding
    emb = embedding_lookup(input_ids)
    store_tensor_disk("forward_emb", emb) # Salva para o Backward
    
    # Camada 2: Hidden (ReLU)
    # No ZeroRAM, salvamos a entrada (emb) e talvez o estado Z antes da ativação
    h1_z = dense_layer_forward(emb, "hidden_01_weights", "hidden_01_bias", activation='linear')
    h1_act = apply_activation(h1_z, 'relu')
    store_tensor_disk("forward_h1_act", h1_act)
    store_tensor_disk("forward_h1_z", h1_z) # Necessário para d_relu
    
    # Camada 3: Output
    logits = dense_layer_forward(h1_act, "output_weights", activation='linear')
    
    # Loss
    probs = compute_softmax(logits)
    loss = calculate_loss(probs, target_ids)
    print(f"Loss: {loss:.6f}")
    
    # 2. Backward Pass (Lendo do Disco)
    print("\nIniciando Backward Pass...")
    
    # Gradiente na Saída (dL/dz para Softmax+CE)
    grad_output = compute_output_gradient(probs, target_ids)
    
    # Gradientes Camada Output -> Hidden
    # Precisamos da ativação h1_act salva no disco
    h1_act_from_disk = load_tensor_disk("forward_h1_act")
    
    print("Calculando gradientes para 'output_weights'...")
    grad_w_output, grad_h1_act = backward_layer_step(
        grad_output, "output_weights", h1_act_from_disk
    )
    
    # Salvar gradientes de pesos no disco (.grad)
    store_tensor_disk("output_weights", grad_w_output, folder='grads')
    
    # Gradientes Camada Hidden -> Embedding
    # Aplicar derivada da ativação (ReLU)
    h1_z_from_disk = load_tensor_disk("forward_h1_z")
    grad_h1_z = grad_h1_act * d_relu(h1_z_from_disk)
    
    # Precisamos do embedding salvo
    emb_from_disk = load_tensor_disk("forward_emb")
    
    print("Calculando gradientes para 'hidden_01_weights'...")
    grad_w_h1, grad_emb = backward_layer_step(
        grad_h1_z, "hidden_01_weights", emb_from_disk
    )
    
    store_tensor_disk("hidden_01_weights", grad_w_h1, folder='grads')
    
    # 3. Validação final
    print("\nValidando arquivos de gradientes:")
    from tensor_manager import WEIGHTS_DIR
    grad_file = os.path.join(WEIGHTS_DIR, "grads", "output_weights.npy")
    if os.path.exists(grad_file):
        size_kb = os.path.getsize(grad_file) / 1024
        print(f" [OK] output_weights.grad: {size_kb:.2f} KB")
        print(f"      Magnitude média do gradiente: {np.abs(grad_w_output).mean():.8f}")
    
    # Limpeza
    dispose_tensor(emb); dispose_tensor(h1_z); dispose_tensor(h1_act); dispose_tensor(logits)
    dispose_tensor(grad_output); dispose_tensor(grad_w_output); dispose_tensor(grad_h1_act)

    print("\nSprint 09 Concluída com Sucesso: Gradientes calculados e armazenados em disco.")

# Mock apply_activation since it was missing in scope for main.py testing above
from engine import apply_activation

if __name__ == "__main__":
    main()
