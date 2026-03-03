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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS train_log (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                epoch INTEGER,
                step INTEGER,
                loss REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS train_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS telemetry (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                value REAL,
                context TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conv_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                role TEXT,
                message TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bias_templates (
                name TEXT PRIMARY KEY,
                description TEXT,
                vector_path TEXT
            )
        ''')
        # Inserir tokens especiais se não existirem
        cursor.execute("INSERT OR IGNORE INTO vocab (id, text) VALUES (0, '<PAD>')")
        special_tokens = [
            (1, '<|system|>'),
            (2, '<|user|>'),
            (3, '<|assistant|>'),
            (4, '<|endoftext|>')
        ]
        cursor.executemany("INSERT OR IGNORE INTO vocab (id, text) VALUES (?, ?)", special_tokens)
        conn.commit()
    print(f"Banco de Dados Inicializado em: {DB_PATH}")

def get_or_create_id(word):
    """Retorna o ID de uma palavra ou cria um novo registro se não existir."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM vocab WHERE text = ?", (word,))
        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            cursor.execute("INSERT INTO vocab (text) VALUES (?)", (word,))
            conn.commit()
            return cursor.lastrowid

def bulk_insert_vocab(words):
    """Insere uma lista de palavras únicas no banco de forma eficiente."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Usamos INSERT OR IGNORE para evitar erros de duplicidade
        cursor.executemany("INSERT OR IGNORE INTO vocab (text) VALUES (?)", [(w,) for w in words])
        conn.commit()

def create_index_on_text():
    """Cria índice na coluna 'text' para acelerar buscas durante o treino."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vocab_text ON vocab (text)")
        conn.commit()
    print("Índice idx_vocab_text criado/validado.")

# Cache simples em memória para os IDs mais frequentes (Otimização da Sprint 4)
ID_TO_TEXT_CACHE = {}
MAX_CACHE_SIZE = 5000

def get_text_by_id(token_id):
    """Retorna o texto original correspondente a um ID no banco."""
    if token_id in ID_TO_TEXT_CACHE:
        return ID_TO_TEXT_CACHE[token_id]
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM vocab WHERE id = ?", (token_id,))
        row = cursor.fetchone()
        if row:
            text = row[0]
            # Gerenciamento simples de cache
            if len(ID_TO_TEXT_CACHE) < MAX_CACHE_SIZE:
                ID_TO_TEXT_CACHE[token_id] = text
            return text
        return "<UNK>" # Token desconhecido

def log_training_metrics(epoch, step, loss):
    """Grava métricas de treinamento no SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO train_log (epoch, step, loss) VALUES (?, ?, ?)",
            (epoch, int(step), float(loss))
        )
        conn.commit()

def log_telemetry(metric_name, value, context=None):
    """Grava métricas de telemetria no SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO telemetry (metric_name, value, context) VALUES (?, ?, ?)",
            (metric_name, float(value), context)
        )
        conn.commit()

if __name__ == "__main__":
    init_db()
