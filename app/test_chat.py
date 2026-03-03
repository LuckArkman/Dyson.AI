import os
from chat import chat_loop
from unittest.mock import patch

def test_chat():
    print("ZeroRAM-GEN: Validando Sprint 17 (Interface de Chat)")
    
    # Simular entrada "/exit" para validar o loop
    with patch('builtins.input', side_effect=['Oi', '/exit']):
        try:
            chat_loop()
            print("\n[OK] Loop de chat iniciado e fechado corretamente.")
        except Exception as e:
            print(f"\n[ERRO] Falha no motor de chat: {e}")

if __name__ == "__main__":
    test_chat()
