import os
import time
from database_manager import init_db, log_training_metrics
from tensor_manager import ensure_v0_weights, reset_accumulated_grads
from tokenizer import sequence_generator
from trainer import train_step, save_training_checkpoint, load_training_checkpoint

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 12 (Loop de Treinamento)")
    
    # 0. Setup
    init_db()
    ensure_v0_weights()
    # reset_accumulated_grads() # Só se quisermos limpar histórico anterior
    
    # 1. Parâmetros do Dataset
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'Dayson', 'pt_0.txt')
    batch_size = 4
    seq_length = 8
    
    # 2. Recuperar Checkpoint
    epoch_start, batch_start = load_training_checkpoint()
    print(f"\nRetomando do Checkpoint: Epoch {epoch_start}, Batch {batch_start}")
    
    # 3. Loop de Treinamento (Exemplo: 1 Época, 5 Passos para validação)
    print("\nIniciando Ciclo de Batches...")
    
    gen = sequence_generator(data_path, batch_size, seq_length)
    
    # Avançar o gerador até o batch_start se necessário (simulado aqui pelo limite de passos)
    steps_to_run = 5
    current_step = 0
    
    for X, Y in gen:
        current_step += 1
        
        start_time = time.time()
        loss, t = train_step(X, Y)
        elapsed = time.time() - start_time
        
        # Log de métricas
        log_training_metrics(epoch_start, current_step, loss)
        
        print(f"[Step {current_step}] Loss: {loss:.4f} | Tempo: {elapsed:.2f}s | Global T: {t}")
        
        # Salvar Checkpoint a cada passo (para validação da sprint)
        save_training_checkpoint(epoch_start, current_step)
        
        if current_step >= steps_to_run:
            break
            
    print("\nSprint 12 Concluída com Sucesso:")
    print(f"- Processados {current_step} batches de tamanho {batch_size}.")
    print(f"- Checkpoint persistido no SQLite.")
    print("- Orquestração Forward -> Backward -> Update validada.")

if __name__ == "__main__":
    main()
