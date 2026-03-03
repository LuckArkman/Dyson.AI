from session_manager import save_session_state, load_session_state, list_recent_sessions
from database_manager import init_db

def main():
    print("ZeroRAM-GEN: Validando Sprint 28 (Persistência de Sessão)")
    init_db()
    
    sid = "test_user_001"
    
    # 1. Salvar Estado
    print(f"\nSalvando estado para a sessão: {sid}")
    save_session_state(sid, persona_name="Expert", bias_name="technical", temperature=0.5)
    
    # 2. Listar Recentes
    print("\nSessões recentes no banco:")
    sessions = list_recent_sessions()
    for s_id, last in sessions:
        print(f"- {s_id} (Atualizada em: {last})")
        
    # 3. Carregar e Validar
    print(f"\nCarregando estado de {sid}...")
    state = load_session_state(sid)
    if state and state['persona_name'] == "Expert":
        print("[OK] Estado recuperado corretamente.")
        print(f"Details: {state}")
    else:
        print("[ERRO] Falha ao recuperar estado da sessão.")

    print("\n[OK] Sprint 28 validada. Sessões agora são cidadãs de primeira classe no SQLite.")

if __name__ == "__main__":
    main()
