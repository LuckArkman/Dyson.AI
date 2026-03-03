import os
from trainer import run_training_session
from tensor_manager import load_model_checkpoint, save_model_checkpoint
from inference import generate_text

def main():
    print("========================================")
    print("   ZeroRAM-GEN: Fine-tuning Português   ")
    print("========================================")
    
    # 1. Carregar Checkpoint Estável v1.1 (Português Base)
    print("\n[STEP 1] Carregando modelo base v1.1_pt...")
    if not load_model_checkpoint("v1.1_pt"):
        print("[!] Checkpoint v1.1_pt não encontrado. Tentando v1.0_trained...")
        if not load_model_checkpoint("v1.0_trained"):
             print("[ERR] Nenhum modelo base encontrado.")
             return
             
    # Limpar estados do otimizador para evitar conflitos de forma (Nova Escala)
    from tensor_manager import WEIGHTS_DIR
    import shutil
    from database_manager import get_db_connection
    optim_dir = os.path.join(WEIGHTS_DIR, 'optim')
    if os.path.exists(optim_dir):
        shutil.rmtree(optim_dir)
    os.makedirs(optim_dir, exist_ok=True)
    with get_db_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO train_state (key, value) VALUES ('adam_t', '1')")
        conn.commit()
    print("[OK] Otimizador resetado para novo aprendizado.")

    # 2. Executar Fine-tuning em Português
    # Usando uma parcela controlada para demonstração (50 steps)
    pt_corpus = r"c:\Users\MPLopes\Documents\Dyson.AI\app\Dayson\pt_0.txt"
    print(f"\n[STEP 2] Iniciando Fine-tuning com o corpus: {pt_corpus}")
    print("Isso irá assimilar novos tokens e adaptar os pesos ao idioma português.")
    
    try:
        run_training_session(pt_corpus, batch_size=4, seq_length=8, epochs=1, steps_per_epoch=50)
    except Exception as e:
        print(f"[ERR] Erro durante o treinamento: {e}")
        return

    # 3. Salvar e Serializar o novo modelo
    print("\n[STEP 3] Serializando e salvando novo checkpoint v1.1_pt...")
    save_model_checkpoint("v1.1_pt")

    # 4. Testar Generalização
    print("\n[STEP 4] Testando resposta após Fine-tuning...")
    
    # Simular recarregamento do zero (Deserialização e Carga)
    print("Simulando recarga do disco...")
    load_model_checkpoint("v1.1_pt")
    
    prompts = [
        "O sistema de inteligência artificial",
        "A economia da Dyson é baseada em",
        "A tecnologia ZeroRAM permite que"
    ]
    
    print("\nResultados da Geração:")
    for p in prompts:
        print(f"\nPrompt: {p}")
        result = generate_text(p, max_new_tokens=15, temperature=0.7, top_k=40)
        print(f"Resposta: {result}")

    print("\n========================================")
    print("   Processo Concluído com Sucesso!      ")
    print("========================================")

if __name__ == "__main__":
    main()
