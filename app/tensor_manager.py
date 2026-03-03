import numpy as np
import os
import json
import time
from database_manager import log_telemetry

# Diretório base para os pesos do modelo
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
REGISTRY_PATH = os.path.join(WEIGHTS_DIR, 'weight_registry.json')

def ensure_weights_dir():
    """Garante que o diretório de pesos existe."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

def initialize_layer_weights(shape, name, init_type='xavier'):
    """Inicializa um tensor de peso e salva no disco."""
    ensure_weights_dir()
    
    if init_type == 'xavier':
        # Xavier Initialization (Glorot)
        limit = np.sqrt(6 / sum(shape))
        weights = np.random.uniform(-limit, limit, shape).astype(np.float32)
    elif init_type == 'zeros':
        weights = np.zeros(shape, dtype=np.float32)
    else:
        weights = np.random.randn(*shape).astype(np.float32) * 0.01

    file_path = os.path.join(WEIGHTS_DIR, f"{name}.npy")
    start_time = time.time()
    np.save(file_path, weights)
    latency = time.time() - start_time
    log_telemetry('io_write_latency', latency, f"layer:{name}")
    return file_path, weights.shape

def save_tensor_logged(path, tensor, name="unknown"):
    """Salva um tensor no disco registrando a latência de escrita."""
    start_time = time.time()
    np.save(path, tensor)
    latency = time.time() - start_time
    log_telemetry('io_write_latency', latency, f"file:{name}")

def create_weight_registry(layers_info):
    """Cria um registro JSON com os metadados de todos os tensores."""
    ensure_weights_dir()
    registry = {
        "model_name": "ZeroRAM-GEN-V0",
        "layers": layers_info
    }
    with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=4)
    print(f"Registro de pesos criado em: {REGISTRY_PATH}")

def get_layer_metadata(name):
    """Recupera metadados de uma camada específica do registro."""
    if not os.path.exists(REGISTRY_PATH):
        return None
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)
    return registry['layers'].get(name)

def load_tensor_mmap(name):
    """Carrega um tensor do disco no modo memory-mapped (Zero RAM)."""
    meta = get_layer_metadata(name)
    if not meta:
        raise ValueError(f"Dados da camada '{name}' não encontrados.")
    
    start_time = time.time()
    tensor = np.load(meta['path'], mmap_mode='r')
    latency = time.time() - start_time
    log_telemetry('io_read_latency', latency, f"layer:{name}")
    
    return tensor

def dispose_tensor(tensor_obj):
    """Remove o objeto da RAM e força a coleta de lixo."""
    import gc
    # Nota: Em Python, del remove a referência. O GC limpa o espaço.
    del tensor_obj
    gc.collect()

def store_tensor_disk(name, tensor, folder='temp'):
    """Salva um tensor temporário (gradiente ou ativação) no disco."""
    path = os.path.join(WEIGHTS_DIR, folder)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{name}.npy")
    start_time = time.time()
    np.save(file_path, tensor)
    latency = time.time() - start_time
    log_telemetry('io_write_latency', latency, f"temp:{name}")
    return file_path

def load_tensor_disk(name, folder='temp'):
    """Carrega um tensor temporário do disco."""
    file_path = os.path.join(WEIGHTS_DIR, folder, f"{name}.npy")
    if not os.path.exists(file_path):
        return None
    start_time = time.time()
    tensor = np.load(file_path)
    latency = time.time() - start_time
    log_telemetry('io_read_latency', latency, f"temp_load:{name}")
    return tensor

def reset_accumulated_grads():
    """Remove todos os gradientes temporários do disco."""
    path = os.path.join(WEIGHTS_DIR, 'grads')
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
        os.makedirs(path)
    # Também limpar pasta temp (ativações)
    temp_path = os.path.join(WEIGHTS_DIR, 'temp')
    if os.path.exists(temp_path):
        import shutil
        shutil.rmtree(temp_path)
        os.makedirs(temp_path)
    print("Gradientes e ativações temporárias removidos.")

def ensure_v0_weights():
    """Garante que os pesos iniciais existem (Tarefa da Sprint 05)."""
    if not os.path.exists(WEIGHTS_DIR) or len(os.listdir(WEIGHTS_DIR)) <= 1:
        print("\n[!] Pesos não encontrados. Reinicializando (Sprint 05)...")
        from database_manager import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM vocab")
            vocab_size = cursor.fetchone()[0]
        
        embed_dim = 128
        hidden_dim = 256
        layers_metadata = {}
        
        # Camada 01: Embeddings
        path, shape = initialize_layer_weights((vocab_size, embed_dim), "embedding_matrix", "xavier")
        layers_metadata["embedding_matrix"] = {"path": path, "shape": list(shape), "dtype": "float32"}
        
        # Camada 02: Hidden 01
        path, shape = initialize_layer_weights((embed_dim, hidden_dim), "hidden_01_weights", "xavier")
        layers_metadata["hidden_01_weights"] = {"path": path, "shape": list(shape), "dtype": "float32"}
        
        path, shape = initialize_layer_weights((hidden_dim,), "hidden_01_bias", "zeros")
        layers_metadata["hidden_01_bias"] = {"path": path, "shape": list(shape), "dtype": "float32"}
        
        # Camada 03: Output
        path, shape = initialize_layer_weights((hidden_dim, vocab_size), "output_weights", "xavier")
        layers_metadata["output_weights"] = {"path": path, "shape": list(shape), "dtype": "float32"}
        
        create_weight_registry(layers_metadata)
        print("[OK] Pesos reinicializados.")
