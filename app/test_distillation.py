from distillation_manager import request_golden_data, store_gold_pair, calculate_similarity_score
from inference import generate_text
from database_manager import init_db

def main():
    print("ZeroRAM-GEN: Validando Sprint 29 (Teacher-Student Distillation)")
    init_db()
    
    prompt = "A inteligência artificial é"
    
    # 1. Gerar resposta com o Pequeno ZeroRAM-GEN (Estudante)
    print(f"\nPrompt: {prompt}")
    student_res = generate_text(prompt, max_new_tokens=10)
    print(f"Estudante (ZeroRAM-GEN): {student_res}")
    
    # 2. Gerar resposta com o Modelo Professor (Gemini - Mockado/Real)
    # Nota: Sem API_KEY ele usará o Mock interno
    print("\nSolicitando 'Golden Data' ao Professor...")
    teacher_res = request_golden_data(prompt)
    print(f"Professor (Gemini): {teacher_res}")
    
    # 3. Comparar e Armazenar
    score = calculate_similarity_score(student_res, teacher_res)
    print(f"\nSemantic Similarity Score: {score:.4f}")
    
    store_gold_pair(prompt, teacher_res, score=score)
    print("[OK] Par 'Dourado' armazenado para futuro retreinamento.")
    
    # 4. Verificar no Banco
    from database_manager import get_db_connection
    with get_db_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM gold_data").fetchone()[0]
        print(f"Total de registros Gold Data: {count}")

    print("\n[OK] Sprint 29 validada. Sistema de destilação preparado.")

if __name__ == "__main__":
    main()
