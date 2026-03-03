import os
import time
from database_manager import init_db, get_db_connection
from tensor_manager import ensure_v0_weights, convert_weights_to_fp16, WEIGHTS_DIR
from tokenizer import sequence_generator
from trainer import train_step

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 15 (Otimização FP16)")
    
    # 0. Setup e Inicialização Normal (FP32)
    init_db()
    ensure_v0_weights()
    
    # 1. Converter para FP16
    convert_weights_to_fp16()
    
    # 2. Validar Tamanhos de Arquivo
    print("\nValidando tamanhos de arquivos (FP16):")
    output_w_path = os.path.join(WEIGHTS_DIR, "output_weights.npy")
    if os.path.exists(output_w_path):
        # Para um vocabulário de ~251k e hidden de 256, original (FP32) era ~246MB.
        # FP16 deve ser ~123MB.
        size_mb = os.path.getsize(output_w_path) / (1024 * 1024)
        print(f" - output_weights.npy: {size_mb:.2f} MB")
        
    # 3. Executar treino para medir nova latência de I/O
    print("\nColetando nova telemetria (FP16)...")
    data_path = os.path.join(os.path.dirname(__file__), 'Dayson', 'pt_0.txt')
    gen = sequence_generator(data_path, batch_size=2, seq_length=5)
    
    # Limpar telemetria para o teste
    with get_db_connection() as conn:
        conn.execute("DELETE FROM telemetry")
    
    X, Y = next(gen)
    train_step(X, Y)
    
    # 4. Resultados
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT AVG(value) FROM telemetry WHERE metric_name = 'io_write_latency'")
        avg_write = cursor.fetchone()[0]
        cursor.execute("SELECT MAX(value) FROM telemetry WHERE metric_name = 'ram_usage_mb'")
        max_ram = cursor.fetchone()[0]

    print(f"\nResultados da Otimização:")
    print(f" - Latência Média de Escrita (FP16): {avg_write:.6f}s")
    print(f" - Pico de RAM: {max_ram:.2f} MB")
    
    print("\nSprint 15 Concluída com Sucesso: Modelo otimizado para FP16.")

if __name__ == "__main__":
    main()
