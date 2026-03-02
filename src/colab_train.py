# Script para execução no Google Colab - Galileu.Node LSTM
# Este script automatiza a instalação de drivers OpenCL e dependências Python no Colab.

import os
import sys

def setup_colab_env():
    print(" === Configurando Ambiente Google Colab para Galileu.Node === ")
    
    # 1. Instalar dependências de sistema para OpenCL
    print("[Colab] Instalando drivers OpenCL e headers...")
    os.system("apt-get update -q")
    os.system("apt-get install -y -q empty_file opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev pocl-opencl-icd")
    
    # 2. Instalar dependências Python
    print("[Colab] Instalando pacotes Python...")
    os.system("pip install pyopencl numpy fastapi uvicorn cryptography aiohttp websockets")
    
    # 3. Criar diretórios necessários
    os.makedirs("Dayson", exist_ok=True)
    
    print("[Colab] Ambiente Pronto!")

def run_training():
    # Adicionar o diretório atual ao path para importar os módulos
    sys.path.append(os.getcwd())
    
    try:
        from gpu.gpu_math_engine import GpuMathEngine
        from brain.neural_network_lstm import GenerativeNeuralNetworkLSTM
        from services.model_trainer import ModelTrainer
        import numpy as np
    except ImportError as e:
        print(f"[Erro] Falha ao importar módulos. Certifique-se de que a pasta 'src' foi carregada corretamente. Erro: {e}")
        return

    print(" === Iniciando Treinamento no Colab === ")
    
    dataset_path = "Dayson/pt_0.txt"
    if not os.path.exists(dataset_path):
        print(f"[Aviso] Dataset {dataset_path} não encontrado. Criando um dataset de exemplo...")
        with open(dataset_path, "w", encoding="utf-8") as f:
            f.write("exemplo de texto para treinamento inicial no ambiente colab do projeto galileu node.")

    math_engine = GpuMathEngine()
    
    # Hiperparâmetros recomendados para Colab
    vocab_size = 5000 
    embedding_size = 64
    hidden_size = 128
    
    model = GenerativeNeuralNetworkLSTM(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        dataset_path=dataset_path,
        math_engine=math_engine
    )
    
    trainer = ModelTrainer(math_engine)
    
    print("[Trainer] Iniciando ciclo de treinamento...")
    trainer.train_model(
        initial_model=model,
        dataset_path=dataset_path,
        final_model_path="Dayson/Dayson_Colab.json",
        learning_rate=0.001, # Recomendado para estabilidade
        epochs=5,
        batch_size=32,
        context_window=12,
        validation_split=0.1
    )
    print("[Sucesso] Treinamento concluído no Google Colab!")

if __name__ == "__main__":
    # Verifica se estamos no Colab
    if 'google.colab' in sys.modules:
        setup_colab_env()
    
    run_training()
