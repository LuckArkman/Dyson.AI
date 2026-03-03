import os
import time
from database_manager import init_db
from tensor_manager import ensure_v0_weights, REGISTRY_PATH
from data_ingestor import ZeroRAMDataIngestor

def auto_initialize_system():
    """
    Lógica de Inicialização Inteligente (Sprint 48):
    Se não houver modelo, inicia o treinamento.
    Se houver, carrega para uso.
    """
    print("--- ZeroRAM-GEN: Sistema de Inicialização Inteligente ---")
    init_db()
    
    if not os.path.exists(REGISTRY_PATH):
        print("[!] Nenhum modelo pré-treinado encontrado.")
        print("[TRABALHO] Iniciando modo de treinamento de novo modelo...")
        
        # 1. Garantir pesos iniciais (Estrutura V0)
        ensure_v0_weights()
        
        # 2. Ingestão de dados automática se houver corpus
        ingestor = ZeroRAMDataIngestor(chunk_size=1000)
        corpus_dir = os.path.join(os.path.dirname(__file__), 'corpus')
        if os.path.exists(corpus_dir):
            ingestor.process_directory(corpus_dir)
        else:
            print("[WARN] Diretório 'corpus' não encontrado. Vocabulário ficará vazio.")
        
        print(f"[OK] Treinamento inicial (Cold Start) configurado.")
    else:
        print("[OK] Modelo pré-treinado detectado.")
        print("[LOAD] Carregando pesos registrados em: " + REGISTRY_PATH)
        
    print("--- Sistema Pronto para Operação ---")

if __name__ == "__main__":
    auto_initialize_system()
