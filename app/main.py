import os
from database_manager import init_db, get_db_connection, bulk_insert_vocab, create_index_on_text
from tokenizer import load_raw_data, tokenize_line

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 03")
    
    # 1. Caminhos de arquivos
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'Dayson', 'pt_0.txt')
    
    # Validação do arquivo de entrada
    if not os.path.exists(data_path):
        print(f"ERRO: Dataset não encontrado em {data_path}")
        return

    # 2. Inicializar banco e estrutura de índices
    init_db()
    create_index_on_text()
    
    # 3. Processamento de Ingestão em Massa
    print(f"\nAlimentando vocabulário SQLite a partir de: {data_path}")
    count_lines = 0
    batch_size = 5000
    unique_tokens_batch = set()
    total_unique_inserted = 0
    
    # Vamos processar as primeiras 20.000 linhas para popular bem o vocabulário real
    max_lines = 20000 
    
    for line in load_raw_data(data_path):
        count_lines += 1
        tokens = tokenize_line(line)
        for t in tokens:
            unique_tokens_batch.add(t)
            
        # Quando atingir o tamanho do lote, insere no banco
        if len(unique_tokens_batch) >= batch_size:
            bulk_insert_vocab(list(unique_tokens_batch))
            total_unique_inserted += len(unique_tokens_batch)
            unique_tokens_batch.clear()
            print(f"Lote Processado: {count_lines} linhas lidas...")
            
        if count_lines >= max_lines:
            break
            
    # Último lote residual
    if unique_tokens_batch:
        bulk_insert_vocab(list(unique_tokens_batch))
        total_unique_inserted += len(unique_tokens_batch)

    # 4. Verificação Final do Banco
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM vocab")
        total_db_count = cursor.fetchone()[0]
            
    print(f"\nSprint 03 Concluída:")
    print(f"- Linhas lidas do dataset: {count_lines}")
    print(f"- Total de tokens únicos no banco de dados agora: {total_db_count}")
    print("\nCheckpoint: Vocabulário pronto para as próximas etapas de serialização de IDs.")

if __name__ == "__main__":
    main()
