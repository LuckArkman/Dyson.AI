import numpy as np
import os
import json

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
    np.save(file_path, weights)
    return file_path, weights.shape

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
