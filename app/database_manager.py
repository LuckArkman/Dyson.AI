import sqlite3
import os
from contextlib import contextmanager

# Caminho para o banco de dados de vocabulário
DB_PATH = os.path.join(os.path.dirname(__file__), 'Dayson', 'vocab.db')

@contextmanager
def get_db_connection():
    """Context manager para conexão segura com o SQLite."""
    conn = None
    try:
        # Garante que o diretório Dayson existe
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        yield conn
    except sqlite3.Error as e:
        print(f"Erro de banco de dados: {e}")
        raise
    finally:
        if conn:
            conn.close()

def init_db():
    """Inicializa a tabela VOCAB conforme especificado no White Paper."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vocab (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL UNIQUE
            )
        ''')
        # Inserir token <PAD> se não existir
        cursor.execute("INSERT OR IGNORE INTO vocab (id, text) VALUES (0, '<PAD>')")
        conn.commit()
    print(f"Banco de Dados Inicializado em: {DB_PATH}")

if __name__ == "__main__":
    init_db()
