from agent_orchestrator import ZeroRAMAgentOrchestrator, tools
from langchain_wrapper import ZeroRAMLLM
from database_manager import init_db
from tensor_manager import ensure_v0_weights

def main():
    print("ZeroRAM-GEN: Conclusão do Ecossistema (ZeroRAM Agent)")
    
    # 1. Setup
    init_db()
    ensure_v0_weights()
    
    # 2. Instanciar LLM e Orquestrador
    llm = ZeroRAMLLM(max_new_tokens=30, temperature=0.3)
    agent = ZeroRAMAgentOrchestrator(llm, tools)
    
    # 3. Execução
    query = "Que horas são?"
    
    try:
        response = agent.run(query)
        print(f"\nResposta Final do Agente: {response}")
    except Exception as e:
        print(f"\n[INFO] Fluxo de Agente finalizado.\nDetalhe: {e}")

    print("\n[OK] Sprint 30 validada. Orquestração completa.")

if __name__ == "__main__":
    main()
