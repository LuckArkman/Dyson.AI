import os
import sys
from gpu.gpu_math_engine import GpuMathEngine
from brain.neural_network_lstm import GenerativeNeuralNetworkLSTM, ModelWeights
from services.model_trainer import ModelTrainer

def run_test_training():
    print(" === Iniciando Teste de Treinamento (Galileu.Node Python) ===")
    
    # Configurações
    dataset_path = "Dayson/pt_0.txt"
    if not os.path.exists(dataset_path):
        # Cria um mini dataset para teste se não existir
        os.makedirs("Dayson", exist_ok=True)
        with open(dataset_path, "w", encoding="utf-8") as f:
            f.write("olá mundo isto é um teste do sistema galileu node em python para treinamento distribuído.")
    
    math_engine = GpuMathEngine()
    
    # Hiperparâmetros
    vocab_size = 1000 
    embedding_size = 64
    hidden_size = 128
    context_window = 4

    
    print(f"[Setup] Iniciando modelo: {vocab_size} tokens, {hidden_size} hidden units.")
    
    model = GenerativeNeuralNetworkLSTM(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        dataset_path=dataset_path,
        math_engine=math_engine
    )
    
    trainer = ModelTrainer(math_engine)
    
    try:
        trainer.train_model(
            initial_model=model,
            dataset_path=dataset_path,
            final_model_path="Dayson/Dayson.json",
            learning_rate=0.01,
            epochs=2, # Reduzido para teste rápido
            batch_size=4,
            context_window=context_window,
            validation_split=0.1
        )
        print("\n[Sucesso] Treinamento de demonstração concluído!")
    except Exception as e:
        print(f"\n[Erro] Falha no treinamento: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test_training()
