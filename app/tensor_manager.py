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

def initialize_layer_weights(shape, name, init_type='xavier', dtype=np.float32):
    """Inicializa um tensor de peso e salva no disco com o dtype especificado."""
    ensure_weights_dir()
    
    if init_type == 'xavier':
        # Xavier Initialization (Glorot)
        limit = np.sqrt(6 / sum(shape))
        weights = np.random.uniform(-limit, limit, shape).astype(dtype)
    elif init_type == 'zeros':
        weights = np.zeros(shape, dtype=dtype)
    else:
        weights = np.random.randn(*shape).astype(dtype) * 0.01

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
    """Remove o objeto da RAM. O GC fará a limpeza automática conforme necessário."""
    # Nota: gc.collect() em cada descarte é extremamente lento (Sprint 22 optimization)
    if tensor_obj is not None:
        del tensor_obj

def store_bias_vector(name, vector, description=""):
    """Salva um vetor de viés e registra no banco de dados."""
    from database_manager import get_db_connection
    
    path = os.path.join(WEIGHTS_DIR, "bias")
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{name}.npy")
    
    np.save(file_path, vector.astype(np.float16))
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO bias_templates (name, description, vector_path) VALUES (?, ?, ?)",
            (name, description, file_path)
        )
        conn.commit()
    print(f"[OK] Viés '{name}' persistido em {file_path}")

def initialize_default_biases(embed_dim=128):
    """Cria templates de viés padrão (Creativity vs Technical)."""
    # Viés Criativo: Pequeno offset positivo em dimensões aleatórias
    creative = np.random.randn(embed_dim).astype(np.float16) * 0.1
    store_bias_vector("creative", creative, "Aumenta a variabilidade das predições.")
    
    # Viés Técnico: Reduz magnitude para ser mais determinístico (simulado)
    technical = np.zeros(embed_dim).astype(np.float16)
    store_bias_vector("technical", technical, "Mantém o comportamento padrão estável.")

def quantize_to_int8(tensor):
    """Converte um tensor FP32/FP16 para INT8 com escala e zero_point."""
    tensor = tensor.astype(np.float32)
    t_min, t_max = np.min(tensor), np.max(tensor)
    
    # Scale: mapear o range original para [-128, 127]
    # Range INT8 = 255
    scale = (t_max - t_min) / 255.0 if t_max != t_min else 1.0
    zero_point = -t_min / scale - 128 if scale != 0 else 0
    
    # Mapear e clipar
    q_tensor = np.round(tensor / scale + zero_point).clip(-128, 127).astype(np.int8)
    
    return q_tensor, float(scale), float(zero_point)

def dequantize_from_int8(q_tensor, scale, zero_point):
    """Restaura o tensor para float usando escala e zero_point."""
    return ((q_tensor.astype(np.float32) - zero_point) * scale).astype(DEFAULT_DTYPE)

def save_quantized_tensor(name, tensor):
    """Quantiza e salva um tensor no disco com metadados de escala."""
    q_tensor, scale, zp = quantize_to_int8(tensor)
    
    # Salvar tensor INT8
    path = os.path.join(WEIGHTS_DIR, f"{name}_int8.npy")
    np.save(path, q_tensor)
    
    # Salvar metadados de escala (em um arquivo JSON pequeno ou via banco)
    # Para simplicidade agora, vamos usar um arquivo .meta
    meta_path = os.path.join(WEIGHTS_DIR, f"{name}.meta")
    import json
    with open(meta_path, 'w') as f:
        json.dump({"scale": scale, "zero_point": zp, "quantized": True}, f)
        
    print(f"[QUANT] '{name}' salvo como INT8 (Scale: {scale:.6f})")

def get_quant_params(name):
    """Recupera metadados de quantização de uma camada."""
    meta_path = os.path.join(WEIGHTS_DIR, f"{name}.meta")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None

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

# Configuração de Precisão (Sprint 12: FP16, Sprint 31: INT8)
DEFAULT_DTYPE = np.float16 
USE_INT8 = True # Habilitar quantização experimental

def convert_weights_to_fp16():
    """Converte todos os pesos do modelo e estados do otimizador para FP16 para economizar I/O."""
    print("\nIniciando conversão para FP16 (Otimização da Sprint 15)...")
    if not os.path.exists(REGISTRY_PATH):
        print("Registro não encontrado.")
        return

    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)

    for name, meta in registry['layers'].items():
        path = meta['path']
        if os.path.exists(path):
            tensor = np.load(path)
            if tensor.dtype != np.float16:
                print(f" -> Convertendo {name}: {tensor.dtype} -> float16")
                tensor_fp16 = tensor.astype(np.float16)
                np.save(path, tensor_fp16)
                meta['dtype'] = 'float16'
                
                # Aproveitar para converter estados do Adam se existirem
                for suffix in ['_m', '_v']:
                    optim_path = os.path.join(WEIGHTS_DIR, 'optim', f"{name}{suffix}.npy")
                    if os.path.exists(optim_path):
                        o_tensor = np.load(optim_path)
                        np.save(optim_path, o_tensor.astype(np.float16))
            dispose_tensor(tensor)
    
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=4)
    print("[OK] Conversão para FP16 concluída.")

def convert_weights_to_int8():
    """Converte todos os pesos do modelo para INT8 (Quantização da Sprint 31)."""
    print("\nIniciando Quantização INT8 (Sprint 31)...")
    if not os.path.exists(REGISTRY_PATH):
        print("Registro não encontrado.")
        return

    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)

    for name, meta in registry['layers'].items():
        path = meta['path']
        if os.path.exists(path):
            tensor = np.load(path)
            print(f" -> Quantizando {name} ({tensor.dtype} -> int8)")
            
            # Salvar como quantized
            save_quantized_tensor(name, tensor)
            
            # Atualizar registro para apontar para o arquivo int8
            meta['path'] = os.path.join(WEIGHTS_DIR, f"{name}_int8.npy")
            meta['dtype'] = 'int8'
            meta['quantized'] = True
            
            dispose_tensor(tensor)
    
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=4)
    print("[OK] Quantização INT8 concluída. Model Size reduced by ~50% (base FP16).")
