from database_manager import get_db_connection

def verify():
    with get_db_connection() as conn:
        count = conn.execute("SELECT count(*) FROM vocab").fetchone()[0]
        print(f"Total de Tokens no Vocabulário: {count}")
        
        print("\nTop 10 tokens recentemente adicionados:")
        rows = conn.execute("SELECT text FROM vocab ORDER BY id DESC LIMIT 10").fetchall()
        for r in rows:
            print(f" - {r[0]}")

if __name__ == "__main__":
    verify()
