from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_wrapper import ZeroRAMLLM
import re

def create_cot_chain(llm: ZeroRAMLLM):
    """
    Cria uma chain que força o modelo a pensar antes de responder.
    """
    cot_template = """Pense passo a passo sobre o seguinte: {task}
Raciocínio:
1."""
    
    prompt = PromptTemplate.from_template(cot_template)
    chain = prompt | llm | StrOutputParser()
    return chain

def parse_cot_result(raw_text: str):
    """
    Tenta separar o raciocínio da resposta final.
    """
    # Procura por palavras chave como "Portanto", "Resposta", etc.
    # Como o modelo ainda é simples, vamos apenas limpar o excesso.
    thoughts = raw_text.strip()
    
    # Simulação de extração de resposta final (se existisse um separador padrão)
    if "final:" in raw_text.lower():
        parts = re.split(r"final:", raw_text, flags=re.IGNORECASE)
        return parts[0].strip(), parts[1].strip()
        
    return thoughts, thoughts # Se não achar, retorna tudo como pensamento e resposta
