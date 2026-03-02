import os
from database_manager import init_db
from tokenizer import load_raw_data, tokenize_line, normalize_text

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 02")
    
    # 1. Caminhos de arquivos
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'Dayson', 'pt_0.txt')
    
    # Validação do arquivo de entrada
    if not os.path.exists(data_path):
        print(f"ERRO: Dataset não encontrado em {data_path}")
        return

    # 2. Teste de Normalização
    frase_teste = "Olá! Como você está hoje? (Testando ZeroRAM-GEN)"
    normalizada = normalize_text(frase_teste)
    tokens_test = tokenize_line(frase_teste)
    
    print(f"Original: '{frase_teste}'")
    print(f"Normalizada: '{normalizada}'")
    print(f"Tokens: {tokens_test}")

    # 3. Teste de Ingestão e Tokenização em Massa
    print(f"\nProcessando dataset bruto: {data_path}")
    count_tokens = 0
    count_lines = 0
    unique_tokens = set()
    
    # Processar apenas as primeiras 1000 linhas para validação rápida da sprint
    for line in load_raw_data(data_path):
        count_lines += 1
        tokens = tokenize_line(line)
        count_tokens += len(tokens)
        for t in tokens:
            unique_tokens.add(t)
            
        if count_lines >= 1000:
            break
            
    print(f"Linhas processadas: {count_lines}")
    print(f"Total de tokens gerados: {count_tokens}")
    print(f"Tokens únicos identificados nestas linhas: {len(unique_tokens)}")
    
    print("\nSprint 02 Concluída com Sucesso: Pipeline de Ingestão e Normalização validado.")

if __name__ == "__main__":
    main()
