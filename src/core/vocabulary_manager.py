import sqlite3
import re
import os
from typing import List, Dict, Optional

class VocabularyManager:
    def __init__(self, db_path: str = "Dayson/vocab.db"):
        self.db_path = os.path.abspath(db_path)
        print(f"[Vocabulary] Conectando ao banco de dados: {self.db_path}")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)

        self.hot_cache_word_to_id = {}
        self.hot_cache_id_to_word = {}
        self.cache_size = 1000
        self.create_schema()

    def create_schema(self):
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vocab (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL UNIQUE
            );
        """)
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_vocab_text ON vocab(text);")
        self.connection.commit()

    def build_vocabulary(self, dataset_path: str) -> int:
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM vocab")
        if cursor.fetchone()[0] > 10:
            print("[Vocabulary] Vocabulário já existente. Carregando hot cache...")
            self.load_hot_cache()
            return self.get_vocab_count()

        print(f"[Vocabulary] Construindo vocabulário a partir de: {dataset_path}")
        temp_counts = {}
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = re.split(r'(\w+|[.,!?;:\'\"/\-])', line.lower())
                tokens = [t for t in tokens if t and not t.isspace()]
                for token in tokens:
                    temp_counts[token] = temp_counts.get(token, 0) + 1

        sorted_tokens = sorted(temp_counts.keys(), key=lambda x: temp_counts[x], reverse=True)
        
        # Início da Transação
        cursor.execute("BEGIN TRANSACTION;")
        cursor.execute("INSERT OR IGNORE INTO vocab (id, text) VALUES (0, '<PAD>')")
        cursor.execute("INSERT OR IGNORE INTO vocab (id, text) VALUES (1, '<UNK>')")

        for token in sorted_tokens:
            if token in ["<PAD>", "<UNK>"]: continue
            cursor.execute("INSERT OR IGNORE INTO vocab (text) VALUES (?)", (token,))
        
        self.connection.commit() # Fim da Transação
        print(f"[Vocabulary] Vocabulário construído com {len(temp_counts)} tokens.")
        self.load_hot_cache()
        return self.get_vocab_count()


    def load_hot_cache(self):
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT id, text FROM vocab ORDER BY id ASC LIMIT {self.cache_size}")
        for row in cursor.fetchall():
            tid, text = row
            self.hot_cache_word_to_id[text] = tid
            self.hot_cache_id_to_word[tid] = text

    def get_token_index(self, token: str) -> int:
        token = token.lower()
        if token in self.hot_cache_word_to_id:
            return self.hot_cache_word_to_id[token]
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM vocab WHERE text = ?", (token,))
        result = cursor.fetchone()
        if result:
            return result[0]
        return 1  # <UNK>

    def get_token(self, tid: int) -> str:
        if tid in self.hot_cache_id_to_word:
            return self.hot_cache_id_to_word[tid]
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM vocab WHERE id = ?", (tid,))
        result = cursor.fetchone()
        return result[0] if result else "<UNK>"

    def get_vocab_count(self) -> int:
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM vocab")
        return cursor.fetchone()[0]

    def close(self):
        self.connection.close()
