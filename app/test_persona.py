from database_manager import init_db
from tokenizer import tokenize_line, normalize_text
from chat_utils import apply_chat_format, map_special_tokens
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def test_persona():
    print("ZeroRAM-GEN: Validando Sprint 23 (Modulação Comportamental)")
    init_db() # Garante tokens especiais
    
    # 1. Testar Normalização com Tags
    text_with_tags = "<|system|> Você é amigável. <|user|> Oi!"
    normalized = normalize_text(text_with_tags)
    print(f"\nOriginal: {text_with_tags}")
    print(f"Normalizado: {normalized}")
    
    tokens = tokenize_line(text_with_tags)
    print(f"Tokens: {tokens}")
    
    # 2. Testar Formatação de Chat
    messages = [
        SystemMessage(content="Você é um assistente socrático."),
        HumanMessage(content="O que é o bem?"),
        AIMessage(content="O que você acha que é?")
    ]
    formatted = apply_chat_format(messages)
    print(f"\nChat Formatted:\n{formatted}")
    
    # 3. Mapear IDs
    ids = map_special_tokens(formatted)
    print(f"Token IDs: {ids}")
    
    # Verificar se as tags mapearam para IDs válidos (não <UNK>)
    # <UNK> retornaria o texto original ou 0 dependendo da implementação, 
    # mas aqui o database_manager.get_text_by_id retorna <UNK> para IDs não encontrados.
    # O get_or_create_id sempre cria se não existir.
    
    from database_manager import get_db_connection
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM vocab WHERE text = '<|system|>'")
        sys_id = cursor.fetchone()[0]
        cursor.execute("SELECT id FROM vocab WHERE text = '<|user|>'")
        user_id = cursor.fetchone()[0]
        cursor.execute("SELECT id FROM vocab WHERE text = '<|assistant|>'")
        ast_id = cursor.fetchone()[0]
        
    print(f"IDs detectados: system={sys_id}, user={user_id}, assistant={ast_id}")
    
    if sys_id in ids and user_id in ids and ast_id in ids:
        print("\n[OK] Tokens especiais detectados e mapeados corretamente.")
    else:
        print("\n[ERRO] Falha ao mapear tokens especiais.")

if __name__ == "__main__":
    test_persona()
