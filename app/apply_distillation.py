import os
from distillation_manager import batch_distillation
from trainer import run_distillation_training
from tensor_manager import load_model_checkpoint, save_model_checkpoint
from inference import generate_text

# Chave API do Usuário
GEMINI_API_KEY = "AIzaSyAEd6BEr9jUvtoLIAhkASOcXmzAcJtZmXE"

def main():
    print("========================================")
    print("   ZeroRAM-GEN: Refinamento por Destilação ")
    print("========================================")
    
    # 1. Carregar Checkpoint v1.1_pt
    print("\n[STEP 1] Carregando modelo base v1.1_pt...")
    if not load_model_checkpoint("v1.1_pt"):
        print("[ERR] v1.1_pt não encontrado.")
        return

    # 2. Gerar Dados Dourados (Destilação)
    from database_manager import get_db_connection
    with get_db_connection() as conn:
        conn.execute("DELETE FROM gold_data") # Limpar treinamentos antigos
        conn.execute("INSERT OR REPLACE INTO train_state (key, value) VALUES ('adam_t', '1')")
        conn.commit()
        
    from tensor_manager import WEIGHTS_DIR
    import shutil
    for folder in ['optim', 'grads']:
        path = os.path.join(WEIGHTS_DIR, folder)
        if os.path.exists(path): shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    
    pt_corpus = r"c:\Users\MPLopes\Documents\Dyson.AI\app\Dayson\pt_0.txt"
    print(f"\n[STEP 2] Refinando amostras de '{pt_corpus}' usando Gemini...")
    batch_distillation(pt_corpus, GEMINI_API_KEY, num_samples=5)
    
    # Conferir se os dados foram realmente salvos
    with get_db_connection() as conn:
        count = conn.execute("SELECT count(*) FROM gold_data").fetchone()[0]
        print(f"\n[INFO] {count} amostras douradas encontradas no banco.")
    
    if count == 0:
        print("[ERR] Falha ao gerar dados dourados. Treinamento abortado.")
        return

    # 3. Treinamento na Gold Data
    print("\n[STEP 3] Iniciando Fine-tuning nos dados refinados...")
    run_distillation_training(batch_size=2, seq_length=8, epochs=3)

    # 4. Salvar modelo refinado
    print("\n[STEP 4] Salvando modelo refinado v1.2_distilled...")
    save_model_checkpoint("v1.2_distilled")

    # 5. Comparar resultados
    print("\n[STEP 5] Testando Generalização e Qualidade...")
    
    prompts = [
        "O aprendizado de máquina é",
        "A tecnologia das redes neurais",
        "O futuro do trabalho com IA"
    ]
    
    print("\nResultados após Destilação:")
    for p in prompts:
        print(f"\nPrompt: {p}")
        result = generate_text(p, max_new_tokens=15, temperature=0.7, top_k=40)
        print(f"Resposta: {result}")

    print("\n========================================")
    print("   Destilação Concluída com Sucesso!    ")
    print("========================================")

if __name__ == "__main__":
    main()
