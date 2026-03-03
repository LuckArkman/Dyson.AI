from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

def apply_chat_format(messages: List[BaseMessage]) -> str:
    """
    Converte uma lista de mensagens do LangChain em uma string formatada com tags especiais.
    Formato: <|system|>...<|user|>...<|assistant|>...
    """
    formatted_text = ""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            formatted_text += f"<|system|>{msg.content}"
        elif isinstance(msg, HumanMessage):
            formatted_text += f"<|user|>{msg.content}"
        elif isinstance(msg, AIMessage):
            formatted_text += f"<|assistant|>{msg.content}"
    
    # Adicionar o marcador final de assistente para induzir a resposta
    if not formatted_text.endswith("<|assistant|>"):
        formatted_text += "<|assistant|>"
        
    return formatted_text

def map_special_tokens(text: str) -> List[int]:
    """
    Traduz o texto formatado (com tags) em uma lista de IDs de tokens.
    """
    from tokenizer import tokenize_line
    from database_manager import get_or_create_id
    
    tokens = tokenize_line(text)
    token_ids = []
    for t in tokens:
        # get_or_create_id lidará com os tokens especiais pois já inserimos no banco com IDs fixos
        token_ids.append(get_or_create_id(t))
    return token_ids
