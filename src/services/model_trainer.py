import os
import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from services.dataset_service import DatasetService
from brain.neural_network_lstm import GenerativeNeuralNetworkLSTM
import json

class ModelTrainer:
    def __init__(self, math_engine: Any):
        self.math_engine = math_engine
        self.log_path = "Dayson/training_log.txt"
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def train_model(self, initial_model: GenerativeNeuralNetworkLSTM, dataset_path: str, final_model_path: str,
                    learning_rate: float, epochs: int, batch_size: int, context_window: int, validation_split: float):
        
        swap_file_path = os.path.join(os.getcwd(), "Dayson/batches.bts")
        
        with DatasetService(swap_file_path) as dataset_service:
            # 1. Preparação do Dataset
            dataset_service.initialize_and_split(
                dataset_path, 
                context_window, 
                initial_model.vocab_manager, 
                batch_size, 
                validation_split
            )

            train_offsets = dataset_service.train_batch_offsets
            val_offsets = dataset_service.validation_batch_offsets

            print(f"\n[Trainer] Ciclo de Treinamento Híbrido (Python) Iniciado.")
            print(f"[Trainer] Duração estimativa baseada em lotes: {len(train_offsets)} por época.")

            total_elapsed_time = 0
            for epoch in range(epochs):
                start_time = time.time()
                print(f"\n{'═'*60}\nÉPOCA {epoch + 1}/{epochs} >> LR: {learning_rate} >> {time.strftime('%H:%M:%S')}\n{'═'*60}")

                total_epoch_loss = 0
                batch_count = 0

                # Carrega pesos para a VRAM/RAM (Modo Zero-RAM ativo)
                weights = initial_model.get_model_weights()


                # Loop de Batches (Limitado para teste rápido)
                for offset in train_offsets[:100]:
                    batch_data = dataset_service.load_batch_from_disk(offset)
                    if not batch_data: continue


                    # Processamento REAL via LSTM com Gradientes e Adam
                    loss = initial_model.train_batch(batch_data, learning_rate, weights)

                    
                    total_epoch_loss += loss
                    batch_count += 1
                    
                    # Log a cada lote para feedback imediato
                    print(f"Lotes: {batch_count}/{len(train_offsets)} | Perda: {loss:.4f}")


                epoch_duration = time.time() - start_time
                avg_loss = total_epoch_loss / batch_count if batch_count > 0 else float('inf')
                
                with open(self.log_path, "a") as f:
                    f.write(f"Época {epoch + 1}/{epochs}: Perda={avg_loss:.4f}, Tempo={epoch_duration:.2f}s\n")
                
                print(f"\nÉpoca {epoch + 1} concluída em {epoch_duration:.0f}s. Perda Média: {avg_loss:.4f}")

                # Checkpoint
                initial_model.save_model(f"Dayson/dayson_{epoch + 1}.json")
            
            initial_model.save_model(final_model_path)
            print(f"[Trainer] Treinamento Finalizado. Modelo salvo em {final_model_path}")
            return initial_model
