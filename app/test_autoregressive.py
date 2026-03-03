from database_manager import init_db
from tensor_manager import ensure_v0_weights
from inference import generate_text

def test_autoregressive():
    print("ZeroRAM-GEN: Validando Sprint 19 (Geração Recursiva)")
    init_db()
    ensure_v0_weights()
    
    # Testar geração com interrupção por pontuação
    prompt = "Hoje o robô"
    print(f"\nPrompt: {prompt}")
    
    # Forçamos uma geração longa para ver se ele para na pontuação
    result = generate_text(prompt, max_new_tokens=20, temperature=1.0, stop_on_punctuation=True)
    print(f"Resultado Final: {result}")
    
    if any(p in result for p in ['.', '!', '?']):
        print("\n[OK] O modelo interrompeu a geração ao encontrar pontuação.")
    else:
        print("\n[INFO] O modelo gerou até o limite ou não encontrou pontuação.")

if __name__ == "__main__":
    test_autoregressive()
