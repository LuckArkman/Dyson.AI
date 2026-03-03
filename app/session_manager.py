from database_manager import get_db_connection

def save_session_state(session_id, persona_name=None, bias_name=None, temperature=0.7):
    """Salva os parâmetros da sessão no banco de dados."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO session_state (session_id, persona_name, bias_name, temperature, last_updated)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (session_id, persona_name, bias_name, float(temperature)))
        conn.commit()
    print(f"[SESSION] Estado da sessão '{session_id}' salvo.")

def load_session_state(session_id):
    """Recupera os parâmetros da sessão do banco de dados."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT persona_name, bias_name, temperature FROM session_state WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
    
    if row:
        return {
            "persona_name": row[0],
            "bias_name": row[1],
            "temperature": row[2]
        }
    return None

def list_recent_sessions(limit=5):
    """Lista as sessões mais recentes."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, last_updated FROM session_state ORDER BY last_updated DESC LIMIT ?",
            (limit,)
        )
        return cursor.fetchall()
