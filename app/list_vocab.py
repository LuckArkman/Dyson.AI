from database_manager import get_db_connection

def list_vocab():
    with get_db_connection() as conn:
        rows = conn.execute("SELECT id, text FROM vocab").fetchall()
        print(f"Vocab Size: {len(rows)}")
        for r in rows:
            print(f" [{r[0]}] {r[1]}")

if __name__ == "__main__":
    list_vocab()
