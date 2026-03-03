import os
import re
from typing import Generator, List, Set
from database_manager import bulk_insert_vocab, create_index_on_text, get_db_connection

class ZeroRAMDataIngestor:
    """
    Motor de ingestão de dados para o ZeroRAM-GEN.
    Focado em ler arquivos massivos e povoar o vocabulário SQLite de forma eficiente.
    """
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        self.tokenizer_regex = re.compile(r"\w+|[^\w\s]")

    def stream_file(self, file_path: str) -> Generator[str, None, None]:
        """Lê um arquivo linha por linha para economizar RAM."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                yield line

    def tokenize(self, text: str) -> List[str]:
        """Tokenização simples baseada em regex."""
        return self.tokenizer_regex.findall(text.lower())

    def process_directory(self, directory_path: str) -> None:
        """Processa todos os arquivos .txt em um diretório."""
        print(f"--- ZeroRAM Ingestor: Iniciando processamento de {directory_path} ---")
        
        unique_tokens: Set[str] = set()
        total_tokens_processed = 0
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    print(f" -> Lendo: {file}")
                    
                    for line in self.stream_file(file_path):
                        tokens = self.tokenize(line)
                        unique_tokens.update(tokens)
                        total_tokens_processed += len(tokens)
                        
                        # Inserir em lotes se o set ficar muito grande
                        if len(unique_tokens) >= self.chunk_size:
                            print(f" [BULK] Inserindo {len(unique_tokens)} tokens únicos no banco...")
                            bulk_insert_vocab(list(unique_tokens))
                            unique_tokens.clear()
        
        # Inserção final
        if unique_tokens:
            bulk_insert_vocab(list(unique_tokens))
            
        print(f"\n[OK] Ingestão concluída.")
        print(f" -> Tokens totais processados: {total_tokens_processed}")
        
        # Garantir indexação
        create_index_on_text()

def create_sample_corpus():
    """Cria um arquivo de texto de exemplo para teste de ingestão."""
    corpus_dir = os.path.join(os.path.dirname(__file__), 'corpus')
    os.makedirs(corpus_dir, exist_ok=True)
    
    sample_path = os.path.join(corpus_dir, 'dyson_economics.txt')
    content = """
    A economia Dyson baseia-se na eficiência absoluta do uso de energia e dados. 
    Diferente de sistemas centralizados, o ZeroRAM-GEN permite que cada nó da rede 
    seja um produtor e consumidor de inteligência. O custo marginal de inferência 
    tende a zero conforme os shards são distribuídos por discos redundantes. 
    Este é o início da singularidade de baixo custo.
    """ * 100 # Multiplicar para simular volume
    
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return corpus_dir

if __name__ == "__main__":
    from database_manager import init_db
    init_db()
    
    ingestor = ZeroRAMDataIngestor(chunk_size=5000)
    corpus_path = create_sample_corpus()
    ingestor.process_directory(corpus_path)
