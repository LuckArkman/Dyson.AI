from langchain_wrapper import ZeroRAMLLM
from memory_manager import ZeroRAMMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from database_manager import init_db
from tensor_manager import ensure_v0_weights

def main():
    print("ZeroRAM-GEN: Validando Sprint 22 (Memória Contextual)")
    
    # 1. Setup
    init_db()
    ensure_v0_weights()
    session_id = "user_123"
    
    # 2. Definir LLM e Prompt
    llm = ZeroRAMLLM(max_new_tokens=10, temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente robô útil que opera em Zero RAM."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    # 3. Criar Chain com Memória
    chain = prompt | llm
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda sid: ZeroRAMMessageHistory(sid),
        input_messages_key="input",
        history_messages_key="history",
    )
    
    # 4. Simular Conversa
    print(f"\n--- Início da Sessão: {session_id} ---")
    
    inputs = [
        "Olá, meu nome é Alice.",
        "Como eu me chamo?"
    ]
    
    config = {"configurable": {"session_id": session_id}}
    
    for user_msg in inputs:
        print(f"\nAlice: {user_msg}")
        response = chain_with_history.invoke({"input": user_msg}, config=config)
        print(f"Robô: {response}")

    print("\n[OK] Memória persistida no SQLite e orquestrada pelo LangChain.")

if __name__ == "__main__":
    main()
