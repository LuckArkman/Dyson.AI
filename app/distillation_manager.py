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
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[ERRO] Falha na API Gemini: {e}")
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
