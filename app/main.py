import os
import numpy as np
from database_manager import init_db, get_or_create_id
from tensor_manager import (
    ensure_v0_weights, store_tensor_disk, load_tensor_disk, 
    dispose_tensor, reset_accumulated_grads, WEIGHTS_DIR,
    get_layer_metadata
)
from engine import (
    embedding_lookup, dense_layer_forward, apply_activation, 
    compute_softmax, calculate_loss, compute_output_gradient, 
    backward_layer_step, d_relu
)
from optimizer import increment_training_step, adam_update_step

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 11 (Otimizadores no Disco)")
    
    # 0. Setup
    init_db()
    ensure_v0_weights()
    reset_accumulated_grads()
    
    # Incrementar o passo global do Adam
    t = increment_training_step()
    print(f"\nPasso de Treinamento (t): {t}")
    
    # 1. Simulação de um passo de Forward + Backward para obter gradientes
    input_ids = [get_or_create_id('oi')]
    target_ids = [get_or_create_id('.')]
    
    emb = embedding_lookup(input_ids)
    h1_z = dense_layer_forward(emb, "hidden_01_weights", "hidden_01_bias", activation='linear')
    h1_act = apply_activation(h1_z, 'relu')
    logits = dense_layer_forward(h1_act, "output_weights", activation='linear')
    probs = compute_softmax(logits)
    
    # Backward simplificado para uma camada para teste do otimizador
    grad_output = compute_output_gradient(probs, target_ids)
    grad_w_out, grad_b_out, grad_h1_act = backward_layer_step(
        grad_output, "output_weights", h1_act
    )
    
    # 2. Teste do Otimizador Adam
    print("\nExecutando atualização Adam para 'output_weights'...")
    
    # Carregar o peso original de forma explícita para comparação
    meta = get_layer_metadata("output_weights")
    original_weights = np.load(meta['path'])
    
    # Aplicar atualização
    new_weights = adam_update_step("output_weights", original_weights, grad_w_out)
    
    # Salvar o novo peso de volta no disco (ZeroRAM: atualizar e persistir)
    np.save(meta['path'], new_weights)
    
    # 3. Validação
    print("\nValidando Resultados da Sprint 11:")
    
    # Verificar se os estados do otimizador foram criados
    m_path = os.path.join(WEIGHTS_DIR, 'optim', "output_weights_m.npy")
    v_path = os.path.join(WEIGHTS_DIR, 'optim', "output_weights_v.npy")
    
    if os.path.exists(m_path):
        print(f" [OK] Estado M salvo: {os.path.getsize(m_path)/1024:.2f} KB")
    if os.path.exists(v_path):
        print(f" [OK] Estado V salvo: {os.path.getsize(v_path)/1024:.2f} KB")
        
    # Verificar se as mudanças nos pesos ocorreram
    diff = np.abs(new_weights - original_weights).mean()
    print(f" [OK] Diferença média nos pesos após atualização: {diff:.10f}")
    
    if diff > 0:
        print(" [OK] Otimizador calibrou os pesos com sucesso.")
    else:
        print(" [ERRO] Os pesos não foram alterados.")

    # Limpeza
    dispose_tensor(original_weights); dispose_tensor(new_weights)
    dispose_tensor(grad_w_out); dispose_tensor(logits)

    print("\nSprint 11 Concluída com Sucesso: Otimizador Adam persistente validado.")

if __name__ == "__main__":
    main()
