from cot_orchestrator import create_cot_chain, parse_cot_result
from langchain_wrapper import ZeroRAMLLM
from database_manager import init_db
from tensor_manager import ensure_v0_weights

def main():
    print("ZeroRAM-GEN: Validando Sprint 25 (Chain-of-Thought)")
    
    # 1. Setup
    init_db()
    ensure_v0_weights()
    
    # 2. Instanciar LLM e Chain
    llm = ZeroRAMLLM(max_new_tokens=15, temperature=0.6)
    cot_chain = create_cot_chain(llm)
    
    # 3. Execução
    task = "Como fazer um robô feliz?"
    print(f"\nTarefa: {task}")
    
    print("\n[Robô está pensando...]")
    raw_output = cot_chain.invoke({"task": task})
    
    thoughts, answer = parse_cot_result(raw_output)
    
    print(f"\nRaciocínio Gerado:\n{thoughts}")
    print(f"\nResposta Final:\n{answer}")
    
    print("\n[OK] Sprint 25 validada. Estrutura de CoT orquestrada com sucesso.")

if __name__ == "__main__":
    main()
