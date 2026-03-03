import os
import shutil
from app_manager import auto_initialize_system
from trainer import run_training_session
from tensor_manager import save_model_checkpoint, load_model_checkpoint, freeze_layer
from data_ingestor import ZeroRAMDataIngestor
import numpy as np

def run_fine_tuning_pt():
    print("--- Sprint 51: Iniciando Fine-tuning em Português ---")
    
    # 1. Carregar Checkpoint Base (v1.0_trained)
    print("\n[RELOAD] Carregando modelo base...")
    if not load_model_checkpoint("v1.0_trained"):
        print("[ERR] Checkpoint 'v1.0_trained' não encontrado. Rode o train_official.py primeiro.")
        return

    # 2. Caminho do arquivo de Fine-tuning (confirmado em Dayson/)
    pt_corpus = os.path.join(os.path.dirname(__file__), 'Dayson', 'pt_0.txt')
    if not os.path.exists(pt_corpus):
        print(f"[ERR] Corpus '{pt_corpus}' não encontrado.")
        return

    # 3. Ingestão de novos tokens (Expansão do Vocabulário para PT)
    print("\n[INGEST] Expandindo vocabulário com termos em português...")
    ingestor = ZeroRAMDataIngestor(chunk_size=10000)
    all_tokens = set()
    for line in ingestor.stream_file(pt_corpus):
        tokens = ingestor.tokenize(line)
        all_tokens.update(tokens)
        
        if len(all_tokens) >= 10000:
            from database_manager import bulk_insert_vocab
            bulk_insert_vocab(list(all_tokens))
            all_tokens.clear()
            print(".", end="", flush=True)
    
    if all_tokens:
        from database_manager import bulk_insert_vocab
        bulk_insert_vocab(list(all_tokens))
    print("\n[OK] Vocabulário expandido.")

    # Sincronizar dimensões do modelo com o novo vocabulário (Dyson Sync)
    from tensor_manager import sync_model_to_vocab
    sync_model_to_vocab()
    
    # 4. Configurar Fine-tuning: Congelar camadas se necessário
    # Para adaptação de idioma, geralmente permitimos que todas as camadas se adaptem,
    # mas poderíamos congelar as camadas de base para focar apenas no topo.
    # freeze_layer("hidden_01_weights") # Exemplo: Congelar base e treinar apenas output?
    # Por padrão no ZeroRAM, vamos permitir o treino completo para melhor convergência em PT.

    # 5. Executar Sessão de Treino
    print("\n[TRAIN] Iniciando treinamento no idioma Português...")
    # Usando parâmetros ajustados para fine-tuning (Learning Rate pode ser reduzida no optimizer se necessário)
    run_training_session(pt_corpus, batch_size=4, seq_length=8, epochs=1, steps_per_epoch=50)
    
    # 6. Salvar novo Checkpoint
    print("\n[STRETP] Consolidando Checkpoint 'v1.1_pt'...")
    save_model_checkpoint("v1.1_pt")
    
    print("\n--- Fine-tuning PT Concluído com Sucesso ---")

if __name__ == "__main__":
    run_fine_tuning_pt()
