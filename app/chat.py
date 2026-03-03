import os
import sys
import time
from database_manager import init_db, get_text_by_id, get_or_create_id
from tensor_manager import ensure_v0_weights, REGISTRY_PATH
from inference import generate_text, predict_next_token
from vocab import serialize, deserialize

def print_slow(text, delay=0.05):
    """Simula digitação para a resposta do robô."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def chat_loop():
    print("========================================")
    print("   ZeroRAM-GEN: Chat Interativo v1.0   ")
    print("========================================")
    print("Comandos: /exit (Sair), /clear (Limpar Contexto)")
    
    # Setup
    init_db()
    ensure_v0_weights()
    
    context_ids = []
    
    while True:
        try:
            user_input = input("\nVocê > ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == '/exit':
                print("Encerrando chat...")
                break
                
            if user_input.lower() == '/clear':
                context_ids = []
                print("[Contexto Resetado]")
                continue
            
            # 1. Processar Input (Serialização com Assimilação)
            input_ids = serialize(user_input)
            context_ids.extend(input_ids)
            
            # Limitar contexto (últimos 16 tokens para estabilidade inicial)
            context_ids = context_ids[-16:]
            
            print("Robô > ", end="")
            
            # 2. Geração Token a Token (Streaming)
            new_tokens = []
            for _ in range(10): # Gerar até 10 tokens por resposta
                next_id = predict_next_token(context_ids, temperature=0.7, top_k=40)
                
                if next_id == 0: # <PAD> ou Stop
                    break
                    
                token_text = get_text_by_id(next_id)
                if token_text is None: break
                
                # Exibir
                sys.stdout.write(token_text + " ")
                sys.stdout.flush()
                time.sleep(0.1)
                
                context_ids.append(next_id)
                new_tokens.append(next_id)
                
                # Curto delay entre tokens
                
            print() # Nova linha ao fim da resposta
            
        except KeyboardInterrupt:
            print("\nSaindo...")
            break
        except Exception as e:
            print(f"\n[ERRO] Ocorreu um problema: {e}")

if __name__ == "__main__":
    chat_loop()
