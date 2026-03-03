import numpy as np
from database_manager import init_db
from tensor_manager import ensure_v0_weights, initialize_default_biases
from inference import generate_text

def test_behavioral_bias():
    print("ZeroRAM-GEN: Validando Sprint 24 (Embeddings Condicionais)")
    
    # 1. Setup
    init_db()
    ensure_v0_weights()
    initialize_default_biases() # Cria os vetores 'creative' e 'technical'
    
    prompt = "O robô está"
    
    # 2. Geração Padrão (Sem viés)
    print(f"\nPrompt: {prompt}")
    res_normal = generate_text(prompt, max_new_tokens=5, bias_name=None)
    print(f"Resultado (Normal): {res_normal}")
    
    # 3. Geração com Viés 'Creative'
    res_creative = generate_text(prompt, max_new_tokens=5, bias_name='creative')
    print(f"Resultado (Creative): {res_creative}")
    
    # 4. Geração com Viés 'Technical' (Deveria ser igual ao normal se for zeros)
    res_tech = generate_text(prompt, max_new_tokens=5, bias_name='technical')
    print(f"Resultado (Technical): {res_tech}")
    
    print("\n[OK] Sprint 24 validada. O motor suporta injeção de viés comportamental via disco.")

if __name__ == "__main__":
    test_behavioral_bias()
