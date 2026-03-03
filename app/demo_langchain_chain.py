from langchain_core.prompts import PromptTemplate
from langchain_wrapper import ZeroRAMLLM
from database_manager import init_db
from tensor_manager import ensure_v0_weights

def main():
    print("ZeroRAM-GEN: Demonstração de PromptTemplates e Chains")
    
    # 1. Setup
    init_db()
    ensure_v0_weights()
    
    # 2. Definir Template
    template = "Você é um robô. O usuário disse: {user_input}. Sua resposta é "
    prompt = PromptTemplate.from_template(template)
    
    # 3. Instanciar LLM
    llm = ZeroRAMLLM(max_new_tokens=5, temperature=0.9)
    
    # 4. Criar Chain (Padrão LCEL - LangChain Expression Language)
    chain = prompt | llm
    
    # 5. Executar
    user_msg = "Olá, quem é você?"
    print(f"\nPrompt Template: '{template}'")
    print(f"Input do Usuário: '{user_msg}'")
    
    res = chain.invoke({"user_input": user_msg})
    
    print(f"\nResposta da Chain: '{res}'")
    print("\n[OK] Sprint 21 validada com padrões modernos do LangChain.")

if __name__ == "__main__":
    main()
