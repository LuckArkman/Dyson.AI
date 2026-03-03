import re
import os
import numpy as np

def load_raw_data(path):
    """Lê o arquivo bruto linha por linha para economia de memória."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

def normalize_text(text):
    """Limpa e normaliza o texto (lowercase e remoção de caracteres especiais)."""
    # Converter para minúsculas
    text = text.lower()
    # Remover caracteres especiais mantendo pontuação básica e tags de sistema <|...|>
    text = re.sub(r"[^a-zA-Z0-9áéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ\s\.\!\?\-\,\<\| \>]", "", text)
    # Proteger as tags especiais (endoftext, etc) para não serem separadas por espaços
    # text = re.sub(r"([\.\!\?\-\,])", r" \1 ", text) # Antigo
    # Nova lógica: Adicionar espaços ao redor da pontuação, mas ignorar o que estiver dentro de <| |>
    text = re.sub(r"(?<![\|<])([\.\!\?\-\,])(?![\|>])", r" \1 ", text)
    # Adicionar espaços ao redor das tags de sistema para garantir que sejam tokens únicos
    text = re.sub(r"(<\|.*?\|>)", r" \1 ", text)
    # Remover espaços extras
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_line(line):
    """Divide a linha normalizada em tokens unitários."""
    normalized = normalize_text(line)
    if not normalized:
        return []
    return normalized.split(" ")

def decode_sequence(token_ids):
    """Converte uma lista de IDs em uma frase humana legível."""
    from database_manager import get_text_by_id
    decoded_tokens = [get_text_by_id(tid) for tid in token_ids]
    # Reconstruir frase, removendo espaços extras ao redor da pontuação
    text = " ".join(decoded_tokens)
    text = re.sub(r"\s+([\.\!\?\-\,])", r"\1", text)
    return text

def token_iterator(data_path):
    """Lê o arquivo bruto e gera tokens um a um (Zero RAM)."""
    for line in load_raw_data(data_path):
        tokens = tokenize_line(line)
        for token in tokens:
            yield token

def sequence_generator(data_path, batch_size, seq_length):
    """Gera batches de (X, Y) para treinamento."""
    from database_manager import get_or_create_id
    
    tokens = []
    # Acumula tokens até ter o suficiente para um batch
    for token in token_iterator(data_path):
        tokens.append(get_or_create_id(token))
        
        # Precisamos de (seq_length + 1) tokens para criar X e Y (Target é o próximo token)
        required_tokens = batch_size * seq_length + 1
        
        if len(tokens) >= required_tokens:
            # Simplificação: gera um batch e limpa o buffer
            # Para treinamento real, usaríamos janelas deslizantes
            X = []
            Y = []
            for i in range(batch_size):
                start = i * seq_length
                end = start + seq_length
                X.append(tokens[start:end])
                Y.append(tokens[end])
            
            yield np.array(X), np.array(Y)
            tokens = tokens[batch_size * seq_length:] # Mantém o último para o próximo X

if __name__ == "__main__":
    # Teste rápido
    sample = "Oi! Tudo bem? Vamos testar o ZeroRAM-GEN."
    tokens = tokenize_line(sample)
    print(f"Original: {sample}")
    print(f"Tokens: {tokens}")
