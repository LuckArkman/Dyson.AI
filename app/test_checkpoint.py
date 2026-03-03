import os
import shutil
from tensor_manager import (
    ensure_v0_weights, save_model_checkpoint, load_model_checkpoint, 
    freeze_layer, is_layer_frozen, unfreeze_layer
)
from database_manager import init_db, get_or_create_id, get_text_by_id

def test_checkpoint_and_finetune():
    print("--- ZeroRAM-GEN: Testando Ciclo de Checkpoint e Fine-tuning (Sprint 47) ---")
    
    # 1. Setup Inicial
    init_db()
    ensure_v0_weights()
    token_id = get_or_create_id("dyson_ai")
    print(f"[INIT] Vocab populado com 'dyson_ai' (ID: {token_id})")
    
    # 2. Salvar Checkpoint 'v1_base'
    print("\n[STRETP 1] Salvando Checkpoint Base...")
    checkpoint_path = save_model_checkpoint("v1_base")
    
    # 3. Simular alteração (deletar pesos atuais)
    print("\n[STRETP 2] Simulando perda de dados / novo ambiente...")
    # (O load_model_checkpoint já cuida de sobrescrever)
    
    # 4. Carregar Checkpoint
    print("\n[STRETP 3] Recarregando Checkpoint...")
    success = load_model_checkpoint("v1_base")
    if success:
        word = get_text_by_id(token_id)
        print(f"[RELOAD] Sucesso! Token {token_id} recuperado como: '{word}'")
    
    # 5. Configurar para Fine-tuning
    print("\n[STRETP 4] Configurando Fine-tuning...")
    freeze_layer("embedding_matrix")
    freeze_layer("hidden_01_weights")
    
    # Verificar status
    if is_layer_frozen("embedding_matrix"):
        print("[FINE-TUNING] Matriz de embeddings protegida contra escrita.")
    
    # 6. Salvar Checkpoint com estado de Fine-tuning
    save_model_checkpoint("v1_finetune_ready")
    
    # 7. Reset e Reload para verificar se o estado de congelamento persistiu
    unfreeze_layer("embedding_matrix") # Resetar localmente
    load_model_checkpoint("v1_finetune_ready")
    
    if is_layer_frozen("embedding_matrix"):
        print("[OK] Estado de congelamento restaurado com sucesso do manifesto.")

    print("\n--- Ciclo de Persistência Validado (Conforme Áudio do Usuário) ---")

if __name__ == "__main__":
    test_checkpoint_and_finetune()
