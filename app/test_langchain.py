from langchain_wrapper import ZeroRAMLLM
from database_manager import init_db
from tensor_manager import ensure_v0_weights

def main():
    print("ZeroRAM-GEN: Validando Sprint 20 (Integração LangChain)")
    
    # Setup
    init_db()
    ensure_v0_weights()
    
    # Instanciar o LLM
    llm = ZeroRAMLLM(max_new_tokens=10, temperature=0.8)
    
    # Teste de Chamada Direta
    prompt = "O robô falou"
    print(f"\nPrompt LangChain: '{prompt}'")
    
    try:
        response = llm.invoke(prompt)
        print(f"Resposta LangChain: '{response}'")
        print("\n[OK] Integração com LangChain validada com sucesso.")
    except Exception as e:
        print(f"\n[ERRO] Falha no Wrapper LangChain: {e}")

if __name__ == "__main__":
    main()
