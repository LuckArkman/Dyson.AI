import numpy as np
import os
from database_manager import get_db_connection
from tensor_manager import WEIGHTS_DIR, store_tensor_disk, load_tensor_disk, dispose_tensor, get_layer_metadata

# Hiperparâmetros padrão do Adam
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
LEARNING_RATE = 0.001

def get_training_step():
    """Recupera o passo global t do SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM train_state WHERE key = 'adam_t'")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

def increment_training_step():
    """Incrementa o passo global t no SQLite."""
    t = get_training_step() + 1
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO train_state (key, value) VALUES ('adam_t', ?)", (str(t),))
        conn.commit()
    return t

def get_optimizer_state(layer_name, shape):
    """Carrega ou inicializa os momentos M e V para uma camada."""
    m = load_tensor_disk(f"{layer_name}_m", folder='optim')
    v = load_tensor_disk(f"{layer_name}_v", folder='optim')
    
    if m is None:
        m = np.zeros(shape, dtype=np.float32)
    if v is None:
        v = np.zeros(shape, dtype=np.float32)
        
    return m, v

def save_optimizer_state(layer_name, m, v):
    """Salva os momentos M e V no disco."""
    store_tensor_disk(f"{layer_name}_m", m, folder='optim')
    store_tensor_disk(f"{layer_name}_v", v, folder='optim')

def adam_update_step(layer_name, weights, grad, lr=LEARNING_RATE):
    """Aplica uma atualização Adam em uma camada de pesos."""
    t = get_training_step()
    if t == 0: t = 1 # Garantir t >= 1 para as fórmulas
    
    m, v = get_optimizer_state(layer_name, weights.shape)
    
    # Atualização dos momentos
    m = BETA1 * m + (1 - BETA1) * grad
    v = BETA2 * v + (1 - BETA2) * np.square(grad)
    
    # Correção de viés
    m_hat = m / (1 - BETA1**t)
    v_hat = v / (1 - BETA2**t)
    
    # Atualização dos pesos
    new_weights = weights - lr * m_hat / (np.sqrt(v_hat) + EPSILON)
    
    # Persistência do estado do otimizador e pesos
    save_optimizer_state(layer_name, m, v)
    
    # Limpeza
    dispose_tensor(m)
    dispose_tensor(v)
    dispose_tensor(m_hat)
    dispose_tensor(v_hat)
    
    return new_weights
