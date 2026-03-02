import re
import os

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
    # Remover caracteres especiais mantendo pontuação básica
    text = re.sub(r"[^a-zA-Z0-9áéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ\s\.\!\?\-\,]", "", text)
    # Adicionar espaços ao redor da pontuação para correta tokenização
    text = re.sub(r"([\.\!\?\-\,])", r" \1 ", text)
    # Remover espaços extras
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_line(line):
    """Divide a linha normalizada em tokens unitários."""
    normalized = normalize_text(line)
    if not normalized:
        return []
    return normalized.split(" ")

if __name__ == "__main__":
    # Teste rápido
    sample = "Oi! Tudo bem? Vamos testar o ZeroRAM-GEN."
    tokens = tokenize_line(sample)
    print(f"Original: {sample}")
    print(f"Tokens: {tokens}")
