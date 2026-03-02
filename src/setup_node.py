import os
import subprocess
import sys

def setup():
    print("=== Configuração Automatizada: Galileu.Node (Python) ===")
    
    # 1. Criar estrutura de diretórios
    directories = [
        "Dayson",
        "Dayson/TensorCache",
        "Dayson/logs",
        "Dayson/batches"
    ]
    
    for folder in directories:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[+] Criado: {folder}")
        else:
            print(f"[.] Já existe: {folder}")

    # 2. Criar um dataset inicial se não existir
    dataset_path = "Dayson/pt_0.txt"
    if not os.path.exists(dataset_path):
        with open(dataset_path, "w", encoding="utf-8") as f:
            f.write("olá mundo isto é um teste do sistema galileu node em python para treinamento distribuído.")
        print("[+] Dataset de teste criado.")

    # 3. Instalar dependências
    print("\n[!] Instalando dependências do requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[+] Dependências instaladas com sucesso.")
    except Exception as e:
        print(f"[-] Erro ao instalar dependências: {e}")

    print("\n=== Configuração Concluída! ===")
    print("Para iniciar o nó, execute: python main.py")

if __name__ == "__main__":
    setup()
