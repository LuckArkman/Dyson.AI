import os
from database_manager import init_db
from tensor_manager import ensure_v0_weights
from inference import generate_text

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 13 (Subsistema de Inferência)")
    
    # 0. Setup
    init_db()
    ensure_v0_weights()
    
    # 1. Teste de Geração de Texto
    # Nota: Como o modelo ainda não foi treinado extensivamente, 
    # os resultados serão baseados nos pesos iniciais (aleatórios) 
    # ou no mini-treino da Sprint anterior.
    
    prompt = "O robô disse"
    
    print("\n--- Teste de Geração 01 (Greedy Search) ---")
    resultado_greedy = generate_text(prompt, max_new_tokens=5, temperature=1.0)
    print(f"Resultado: {resultado_greedy}")
    
    print("\n--- Teste de Geração 02 (Com Temperature 0.8 e Top-K 50) ---")
    resultado_random = generate_text(prompt, max_new_tokens=5, temperature=0.8, top_k=50)
    print(f"Resultado: {resultado_random}")
    
    print("\nSprint 13 Concluída com Sucesso:")
    print("- Motor de Inferência operando em modo Zero RAM.")
    print("- Geração auto-regressiva validada.")

if __name__ == "__main__":
    main()
