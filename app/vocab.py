import re
import os
import numpy as np
from database_manager import get_or_create_id, get_text_by_id, get_db_connection
from tensor_manager import sync_model_to_vocab

def load_raw_data(path):
    """Lê o arquivo bruto linha por linha para economia de memória."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

def normalize_text(text):
    """Limpa e normaliza o texto (lowercase e remoção de caracteres especiais)."""
    text = text.lower()
    # Adicionar espaços ao redor da pontuação, mas ignorar o que estiver dentro de <| |>
    text = re.sub(r"(?<![\|<])([\.\!\?\-\,])(?![\|>])", r" \1 ", text)
    # Adicionar espaços ao redor das tags de sistema
    text = re.sub(r"(<\|.*?\|>)", r" \1 ", text)
    # Remover espaços extras
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    """Converte texto bruto em uma lista de strings de tokens."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")

def serialize(text):
    """
    Converte texto bruto em uma lista de IDs de tokens (Serialização).
    Se um novo token for encontrado, ele é assimilado no vocabulário SQLite 
    e as matrizes de pesos do modelo são expandidas automaticamente.
    """
    tokens = tokenize(text)
    
    # Obter tamanho atual do vocabulário para detectar novos tokens
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM vocab")
        max_id_before = cursor.fetchone()[0] or 0
    
    token_ids = []
    new_token_added = False
    
    for token in tokens:
        tid = get_or_create_id(token)
        token_ids.append(tid)
        if tid > max_id_before:
            new_token_added = True
            
    # Se novos tokens foram adicionados, sincronizar o modelo (Assimilação Total)
    if new_token_added:
        print(f"[VOCAB] Novos tokens detectados. Iniciando expansão de pesos...")
        sync_model_to_vocab()
        
    return token_ids

def deserialize(token_ids):
    """Converte uma lista de IDs de volta para texto humano (Desserialização)."""
    decoded_tokens = [get_text_by_id(tid) for tid in token_ids]
    # Reconstruir frase, removendo espaços extras ao redor da pontuação
    text = " ".join(decoded_tokens)
    text = re.sub(r"\s+([\.\!\?\-\,])", r"\1", text)
    return text

def token_iterator(data_path):
    """Lê o arquivo bruto e gera tokens um a um (Zero RAM)."""
    for line in load_raw_data(data_path):
        tokens = tokenize(line)
        for token in tokens:
            yield token

def sequence_generator(data_path, batch_size, seq_length):
    """Gera batches de (X, Y) para treinamento com assimilação automática."""
    tokens = []
    # Acumula tokens até ter o suficiente para um batch
    for tid in serialize_iterator(data_path):
        tokens.append(tid)
        
        # Precisamos de (seq_length + 1) tokens para criar X e Y (Target é o próximo token)
        required_tokens = batch_size * seq_length + 1
        
        if len(tokens) >= required_tokens:
            X = []
            Y = []
            for i in range(batch_size):
                start = i * seq_length
                end = start + seq_length
                X.append(tokens[start:end])
                Y.append(tokens[end])
            
            yield np.array(X), np.array(Y)
            tokens = tokens[batch_size * seq_length:] 

def serialize_iterator(data_path):
    """Iterador que serializa tokens e assimila novos termos."""
    # Obter tamanho atual do vocabulário
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM vocab")
        max_id_seen = cursor.fetchone()[0] or 0
        
    for token in token_iterator(data_path):
        tid = get_or_create_id(token)
        if tid > max_id_seen:
            # Novo token! Precisamos expandir o modelo AGORA para o treino não quebrar
            print(f"[VOCAB] Novo token '{token}' (ID {tid}) detectado. Expandindo weights...")
            sync_model_to_vocab()
            max_id_seen = tid
            
        yield tid

def gold_data_generator(batch_size, seq_length):
    """Gera batches a partir da tabela gold_data no SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT completion FROM gold_data")
        rows = cursor.fetchall()
    
    tokens = []
    for row in rows:
        text = row[0]
        # Serializar o texto (com assimilação se necessário)
        tids = serialize(text)
        tokens.extend(tids)
        
        while len(tokens) >= batch_size * seq_length + 1:
            X = []
            Y = []
            for i in range(batch_size):
                start = i * seq_length
                end = start + seq_length
                X.append(tokens[start:end])
                Y.append(tokens[end])
            
            yield np.array(X), np.array(Y)
            tokens = tokens[batch_size * seq_length:]

if __name__ == "__main__":
    # Teste de Assimilação
    test_text = "DysonAI é um sistema inovador de ZeroRAM."
    print(f"Original: {test_text}")
    ids = serialize(test_text)
    print(f"IDs Serializados: {ids}")
    decoded = deserialize(ids)
    print(f"Desserializado: {decoded}")
