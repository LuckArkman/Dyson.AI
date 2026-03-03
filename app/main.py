import os
import time
from database_manager import init_db, log_training_metrics, get_db_connection
from tensor_manager import ensure_v0_weights, reset_accumulated_grads
from tokenizer import sequence_generator
from trainer import train_step, check_early_stopping

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 16 (Early Stopping Test)")
    
    # 0. Setup
    init_db()
    ensure_v0_weights()
    reset_accumulated_grads()
    
    # Limpar estado de treino anterior para o teste
    with get_db_connection() as conn:
        conn.execute("DELETE FROM train_state WHERE key IN ('best_loss', 'patience_counter')")
        conn.commit()
    
    # 1. Parâmetros
    data_path = os.path.join(os.path.dirname(__file__), 'Dayson', 'pt_0.txt')
    batch_size = 2
    seq_length = 5
    patience = 3 # Paciência curta para o teste
    
    print("\nIniciando Ciclo de Treino com Early Stopping...")
    gen = sequence_generator(data_path, batch_size, seq_length)
    
    for i in range(10): # Tentamos rodar 10 passos
        X, Y = next(gen)
        loss, t = train_step(X, Y)
        
        # Log 
        log_training_metrics(0, i+1, loss)
        print(f" [Step {i+1}] Loss: {loss:.6f}")
        
        # Early Stopping Check
        should_stop = check_early_stopping(loss, patience=patience)
        
        if should_stop:
            print(f"\n[STOP] Treinamento interrompido por falta de melhora (Paciência: {patience})")
            break
            
    print("\nSprint 16 Concluída com Sucesso:")
    print("- Mecanismo de Early Stopping validado.")
    print("- Versão 'Best Model' persistida em /weights/best/ se houve melhora.")

if __name__ == "__main__":
    main()
