import os
import google.generativeai as genai
from database_manager import get_db_connection

def request_golden_data(prompt, api_key=None):
    """
    Solicita uma completude de alta qualidade ao Gemini (Modelo Professor).
    """
    if not api_key:
        # Mock para fins de demonstração se não houver chave
        return f"Esta é uma resposta de alta qualidade gerada sinteticamente para: {prompt}"

    try:
        genai.configure(api_key=api_key)
        # Utilizando a versão sugerida pelo usuário (2.5 Flash Lite)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[ERRO] Falha na API Gemini (Verifique a KEY): {e}")
        return None

def store_gold_pair(prompt, completion, source="Gemini", score=1.0):
    """Grava o par prompt-resposta 'Dourado' no banco."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO gold_data (prompt, completion, source, similarity_score) VALUES (?, ?, ?, ?)",
            (prompt, completion, source, score)
        )
        conn.commit()

def batch_distillation(data_path, api_key, num_samples=10):
    """
    Lê o dataset, pede ao Gemini para refinar exemplos e salva no gold_data.
    """
    from vocab import load_raw_data
    
    print(f"\n[DISTILL] Iniciando refinamento de {num_samples} amostras...")
    samples = []
    
    # Pegar amostras do arquivo
    gen = load_raw_data(data_path)
    for i, line in enumerate(gen):
        if i >= num_samples: break
        if len(line.strip()) < 20: continue # Ignorar linhas curtas
        samples.append(line)
        
    for original in samples:
        # Prompt de refinamento
        prompt = (
            "Refine este texto para que seja mais robusto, profundo, elaborado e gramaticalmente perfeito em português. "
            "Mantenha o sentido original, mas use um tom mais profissional e técnico se apropriado.\n\n"
            f"Texto original: {original}\n\n"
            "Texto refinado:"
        )
        
        print(f"Refinando: {original[:50]}...")
        refined = request_golden_data(prompt, api_key=api_key)
        
        if refined:
            # Limpar a resposta do Gemini (remover labels se houver)
            refined = refined.replace("Texto refinado:", "").strip()
            score = calculate_similarity_score(original, refined)
            store_gold_pair(original, refined, score=score)
            print(f" [OK] Score {score:.4f} | Salvo no gold_data.")
            
    print("\n[OK] Refinamento (Destilação) concluída.")

def calculate_similarity_score(text1, text2):
    """
    Calcula a similaridade textual básica (Jaccard).
    """
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union: return 0.0
    return len(intersection) / len(union)
