import time
import random
from database_manager import init_db, log_training_metrics

def simulate_training_loop(epochs=2, steps_per_epoch=10):
    print("ZeroRAM-GEN: Simulando Treinamento para o Dashboard...")
    init_db()
    
    current_loss = 2.5
    for epoch in range(epochs):
        print(f"Época {epoch} iniciada.")
        for step in range(steps_per_epoch):
            # Simular queda da loss (convergência)
            current_loss *= 0.95
            current_loss += random.uniform(-0.05, 0.05)
            if current_loss < 0.1: current_loss = 0.1
            
            print(f" -> Época {epoch}, Step {step}: Loss {current_loss:.4f}")
            log_training_metrics(epoch, step, current_loss)
            time.sleep(1) # Aguarda 1s entre passos para percepção humana
    
    print("[OK] Simulação de treino concluída.")

if __name__ == "__main__":
    simulate_training_loop()
