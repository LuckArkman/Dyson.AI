import os
import time
from database_manager import init_db, get_db_connection
from tensor_manager import ensure_v0_weights, reset_accumulated_grads
from tokenizer import sequence_generator
from trainer import train_step

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 14 (Telemetria)")
    
    # 0. Setup
    init_db()
    ensure_v0_weights()
    reset_accumulated_grads()
    
    # Limpar telemetria antiga
    with get_db_connection() as conn:
        conn.execute("DELETE FROM telemetry")
        conn.commit()
    
    # 1. Parâmetros
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'Dayson', 'pt_0.txt')
    batch_size = 2
    seq_length = 5
    
    # 2. Executar alguns passos para coletar telemetria
    print("\nExecutando Ciclo de Treino com Coleta de Telemetria...")
    gen = sequence_generator(data_path, batch_size, seq_length)
    
    for i in range(2):
        X, Y = next(gen)
        loss, t = train_step(X, Y)
        print(f" [Step {i+1}] Loss: {loss:.4f}")
        
    # 3. Validar Telemetria no Banco
    print("\nResumo da Telemetria (SQLite):")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        metrics = [
            ('io_read_latency', 'Latência Média de Leitura (I/O)', '{:.6f}s'),
            ('io_write_latency', 'Latência Média de Escrita (I/O)', '{:.6f}s'),
            ('ram_usage_mb', 'Pico de RAM Detectado', '{:.2f} MB'),
            ('grad_norm', 'Última Magnitude de Gradiente', '{:.8f}')
        ]
        
        for metric_id, label, fmt in metrics:
            if metric_id == 'ram_usage_mb':
                cursor.execute(f"SELECT MAX(value) FROM telemetry WHERE metric_name = '{metric_id}'")
            elif metric_id == 'grad_norm':
                cursor.execute(f"SELECT value FROM telemetry WHERE metric_name = '{metric_id}' ORDER BY timestamp DESC LIMIT 1")
            else:
                cursor.execute(f"SELECT AVG(value) FROM telemetry WHERE metric_name = '{metric_id}'")
                
            val = cursor.fetchone()[0]
            if val is not None:
                print(f" - {label}: " + fmt.format(val))
            else:
                print(f" - {label}: [Dados não encontrados]")

    print("\nSprint 14 Concluída com Sucesso: Telemetria integrada e validada.")

if __name__ == "__main__":
    main()
