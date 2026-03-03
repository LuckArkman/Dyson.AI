import sqlite3
import os
from contextlib import contextmanager
from typing import List, Optional, Tuple, Any, Dict, Generator

# Caminho para o banco de dados de vocabulário
DB_PATH: str = os.path.join(os.path.dirname(__file__), 'Dayson', 'vocab.db')

@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager para conexão segura com o SQLite.
    
    Yields:
        sqlite3.Connection: Conexão ativa com o banco de dados especificado em DB_PATH.
    """
    conn: Optional[sqlite3.Connection] = None
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

def init_db() -> None:
    """
    Inicializa as tabelas do banco de dados conforme as especificações do ZeroRAM-GEN.
    Cria as tabelas de vocabulário, logs de treino, telemetria, histórico, biased templates,
    sessão, dados sintéticos, estatísticas de vocabulário, sharding e rede.
    """
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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT PRIMARY KEY,
                persona_name TEXT,
                bias_name TEXT,
                temperature REAL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gold_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                completion TEXT,
                source TEXT,
                similarity_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vocab_stats (
                id INTEGER PRIMARY KEY,
                usage_count INTEGER DEFAULT 0,
                FOREIGN KEY(id) REFERENCES vocab(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shard_map (
                tensor_name TEXT,
                shard_id INTEGER,
                start_index INTEGER,
                end_index INTEGER,
                file_path TEXT,
                PRIMARY KEY (tensor_name, shard_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS network_nodes (
                node_id TEXT PRIMARY KEY,
                base_url TEXT,
                status TEXT,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS swarm_stats (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        # Inserir tokens especiais se não existirem
        cursor.execute("INSERT OR IGNORE INTO vocab (id, text) VALUES (0, '<PAD>')")
        special_tokens: List[Tuple[int, str]] = [
            (1, '<|system|>'),
            (2, '<|user|>'),
            (3, '<|assistant|>'),
            (4, '<|endoftext|>')
        ]
        cursor.executemany("INSERT OR IGNORE INTO vocab (id, text) VALUES (?, ?)", special_tokens)
        conn.commit()
    print(f"Banco de Dados Inicializado em: {DB_PATH}")

def bulk_insert_vocab(words: List[str]) -> None:
    """
    Insere uma lista de palavras únicas no banco de forma eficiente.
    
    Args:
        words: Lista de strings a serem inseridas no vocabulário.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany("INSERT OR IGNORE INTO vocab (text) VALUES (?)", [(w,) for w in words])
        conn.commit()

def create_index_on_text() -> None:
    """Cria índice na coluna 'text' da tabela vocab para acelerar buscas."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vocab_text ON vocab (text)")
        conn.commit()
    print("Índice idx_vocab_text criado/validado.")

# Cache simples em memória para os IDs mais frequentes (Otimização da Sprint 4/27)
ID_TO_TEXT_CACHE: Dict[int, str] = {}
TEXT_TO_ID_CACHE: Dict[str, int] = {}
MAX_CACHE_SIZE: int = 10000

def update_vocab_usage(token_id: int) -> None:
    """
    Incrementa o contador de uso de um token para otimização de cache.
    
    Args:
        token_id: ID do token a ser incrementado.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO vocab_stats (id, usage_count) VALUES (?, 1) "
            "ON CONFLICT(id) DO UPDATE SET usage_count = usage_count + 1",
            (token_id,)
        )
        conn.commit()

def build_hot_token_cache(size: int = 5000) -> None:
    """
    Carrega os tokens mais usados do banco para a RAM (Cache Hot-Tokens).
    
    Args:
        size: Quantidade de tokens frequentes a carregar.
    """
    global ID_TO_TEXT_CACHE, TEXT_TO_ID_CACHE
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT v.id, v.text FROM vocab v
            JOIN vocab_stats s ON v.id = s.id
            ORDER BY s.usage_count DESC LIMIT ?
        ''', (size,))
        rows: List[Tuple[int, str]] = cursor.fetchall()
        for tid, text in rows:
            ID_TO_TEXT_CACHE[tid] = text
            TEXT_TO_ID_CACHE[text] = tid
    print(f"[CACHE] {len(rows)} Hot-Tokens carregados na RAM.")

def get_text_by_id(token_id: int) -> str:
    """
    Retorna o texto original correspondente a um ID, usando cache se possível.
    
    Args:
        token_id: ID do token.
        
    Returns:
        str: Texto do token ou '<UNK>' se não encontrado.
    """
    if token_id in ID_TO_TEXT_CACHE:
        return ID_TO_TEXT_CACHE[token_id]
        
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM vocab WHERE id = ?", (token_id,))
        row: Optional[Tuple[str]] = cursor.fetchone()
        
    if row:
        text = row[0]
        if len(ID_TO_TEXT_CACHE) < MAX_CACHE_SIZE:
            ID_TO_TEXT_CACHE[token_id] = text
            TEXT_TO_ID_CACHE[text] = token_id
        return text
    return "<UNK>"

def get_or_create_id(text: str) -> int:
    """
    Retorna o ID de um token, buscando no cache ou criando no banco se necessário.
    
    Args:
        text: Texto do token.
        
    Returns:
        int: ID único do token.
    """
    if text in TEXT_TO_ID_CACHE:
        return TEXT_TO_ID_CACHE[text]
        
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM vocab WHERE text = ?", (text,))
        row: Optional[Tuple[int]] = cursor.fetchone()
        
        if row:
            token_id = row[0]
        else:
            cursor.execute("INSERT INTO vocab (text) VALUES (?)", (text,))
            token_id = cursor.lastrowid
            conn.commit()
    
    if len(ID_TO_TEXT_CACHE) < MAX_CACHE_SIZE:
        ID_TO_TEXT_CACHE[token_id] = text
        TEXT_TO_ID_CACHE[text] = token_id
        
    return token_id

def log_training_metrics(epoch: int, step: int, loss: float) -> None:
    """
    Grava métricas de treinamento (Loss por step) no banco de dados.
    
    Args:
        epoch: Época atual.
        step: Passo global de treino.
        loss: Valor do erro (Loss).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO train_log (epoch, step, loss) VALUES (?, ?, ?)",
            (epoch, int(step), float(loss))
        )
        conn.commit()

def log_telemetry(metric_name: str, value: float, context: Optional[str] = None) -> None:
    """
    Grava o log de telemetria para análise de performance de I/O e RAM.
    
    Args:
        metric_name: Nome da métrica (ex: 'io_read_latency').
        value: Valor numérico da métrica.
        context: Contexto adicional opcional.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO telemetry (metric_name, value, context) VALUES (?, ?, ?)",
            (metric_name, float(value), context)
        )
        conn.commit()

if __name__ == "__main__":
    init_db()
