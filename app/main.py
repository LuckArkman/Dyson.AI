import os
import sqlite3
from database_manager import init_db, get_db_connection

def main():
    print("ZeroRAM-GEN: Iniciando Sprint 01")
    
    # 1. Inicializar o Banco de Dados
    init_db()
    
    # 2. Validar Estrutura do Vocabulário
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM vocab")
        count = cursor.fetchone()[0]
        
        cursor.execute("SELECT * FROM vocab LIMIT 5")
        rows = cursor.fetchall()

    print(f"Total de tokens no vocabulário atual: {count}")
    print("Primeiros 5 tokens:")
    for row in rows:
        print(f" ID: {row[0]} | TEXT: '{row[1]}'")
        
    print("\nSprint 01 Concluída com Sucesso: Ambiente Validado e Vocabulário SQLite Pronto.")

if __name__ == "__main__":
    main()
