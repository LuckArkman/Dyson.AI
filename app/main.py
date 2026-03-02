import os
from database_manager import get_or_create_id, get_text_by_id
from tokenizer import tokenize_line, decode_sequence

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 04")
    
    # 1. Testar Serialização (Texto -> ID)
    frase_original = "Oi! Como você está?"
    tokens = tokenize_line(frase_original)
    
    # Converter cada token em ID (Muitos já existem no banco da Sprint 03)
    ids = []
    print(f"Serializando: '{frase_original}'")
    for t in tokens:
        tid = get_or_create_id(t)
        ids.append(tid)
        print(f" Token: '{t}' -> ID: {tid}")

    print(f"Sequência de IDs: {ids}")

    # 2. Testar Desserialização (ID -> Texto)
    print("\nDesserializando a sequência:")
    frase_reconstruida = decode_sequence(ids)
    
    print(f"Frase Reconstruída: '{frase_reconstruida}'")
    
    # 3. Validar consistência
    # (Note que a normalização da Sprint 2 torna tudo lowercase)
    original_norm = " ".join(tokenize_line(frase_original))
    reconstruida_norm = " ".join(tokenize_line(frase_reconstruida))
    
    if original_norm == reconstruida_norm:
        print("\nSucesso: O processo de ida e volta é consistente.")
    else:
        print("\nErro: A frase reconstruída não condiz com a original normalizada.")
        
    # 4. Testar Token Desconhecido (UNK)
    unk_id = 999999999
    print(f"\nTestando ID desconhecido ({unk_id}):")
    print(f" Resultado: '{get_text_by_id(unk_id)}'")

    print("\nSprint 04 Concluída com Sucesso: Serialização/Desserialização Bidirecional validada.")

if __name__ == "__main__":
    main()
