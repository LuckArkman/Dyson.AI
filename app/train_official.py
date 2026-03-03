import os
import shutil
from app_manager import auto_initialize_system
from trainer import run_training_session
from tensor_manager import save_model_checkpoint, load_model_checkpoint
from demo_v1_stable import run_stable_inference_demo

def run_official_training_cycle():
    print("--- Sprint 50: Ciclo de Treinamento Oficial v1.0 ---")
    
    # 1. Garantir que o sistema está inicializado (Vocab e Pesos V0)
    auto_initialize_system()
    
    # Limpar shards antigos para evitar conflitos de I/O no demo
    from database_manager import get_db_connection
    with get_db_connection() as conn:
        conn.execute("DELETE FROM shard_map")
        conn.commit()
    
    # 2. Caminho do Mini-Corpus (Gerado na Sprint 46)
    corpus_file = os.path.join(os.path.dirname(__file__), 'corpus', 'dyson_economics.txt')
    if not os.path.exists(corpus_file):
        print("[ERR] Corpus não encontrado. Rode o data_ingestor.py primeiro.")
        return

    # 3. Executar o Treinamento Real (1 Época para o demo)
    print("\n[TREINO] Iniciando processamento de gradientes em disco...")
    # X, Y serão gerados pelo sequence_generator em trainer.py
    run_training_session(corpus_file, batch_size=2, seq_length=4, epochs=1, steps_per_epoch=20)
    
    # 4. Salvar Checkpoint Oficial Pós-Treino
    print("\n[STRETP] Consolidando Checkpoint 'v1.0_trained'...")
    checkpoint_dir = save_model_checkpoint("v1.0_trained")
    
    # 5. Teste Qualitativo de Inferência
    print("\n[TESTE] Validando inferência com o modelo treinado...")
    run_stable_inference_demo("Dyson Network is")

    print("\n--- Épico de Treinamento Concluído com Sucesso ---")

if __name__ == "__main__":
    run_official_training_cycle()
