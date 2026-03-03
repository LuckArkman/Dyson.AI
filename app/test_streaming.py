from langchain_wrapper import ZeroRAMLLM
from database_manager import init_db
from tensor_manager import ensure_v0_weights
import sys
import time

def main():
    print("ZeroRAM-GEN: Validando Sprint 26 (Streaming Nativo LangChain)")
    
    # Setup
    init_db()
    ensure_v0_weights()
    
    llm = ZeroRAMLLM(max_new_tokens=10, temperature=0.7)
    prompt = "Era uma vez um"
    
    print(f"\nPrompt: '{prompt}'")
    print("Resposta (Streaming): ", end="")
    
    start_time = time.time()
    tokens_count = 0
    
    for chunk in llm.stream(prompt):
        text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        sys.stdout.write(text)
        sys.stdout.flush()
        tokens_count += 1
        
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n\nEstatísticas:")
    print(f"- Tokens gerados: {tokens_count}")
    print(f"- Tempo total: {duration:.2f}s")
    print(f"- Velocidade: {tokens_count/duration:.2f} tokens/s")
    
    print("\n[OK] Sprint 26 validada com sucesso.")

if __name__ == "__main__":
    main()
