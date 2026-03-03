from inference import generate_text
from tensor_manager import load_model_checkpoint

def test():
    print(">>> Testando Modelo Refinado (v1.2_distilled) <<<")
    if not load_model_checkpoint("v1.2_distilled"):
        print("Erro ao carregar modelo.")
        return
        
    prompts = [
        "O aprendizado de máquina é",
        "A tecnologia das redes neurais",
        "O futuro do trabalho com IA"
    ]
    
    for p in prompts:
        print(f"\nPrompt: {p}")
        # Usando temperatura mais baixa para ver a solidez do aprendizado
        result = generate_text(p, max_new_tokens=40, temperature=0.5)
        print(f"Resposta: {result}")

if __name__ == "__main__":
    test()
