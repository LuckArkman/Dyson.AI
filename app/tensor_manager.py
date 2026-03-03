import numpy as np
import os
import json
import time
import lz4.frame
import shutil
from typing import List, Optional, Tuple, Dict, Any, Union
from database_manager import log_telemetry, get_db_connection

# Diretório base para os pesos do modelo
WEIGHTS_DIR: str = os.path.join(os.path.dirname(__file__), 'weights')
REGISTRY_PATH: str = os.path.join(WEIGHTS_DIR, 'weight_registry.json')

# Configuração de Precisão (Sprint 12: FP16, Sprint 31: INT8)
DEFAULT_DTYPE = np.float16 
USE_INT8: bool = True # Habilitar quantização experimental

# Dicionário para controlar camadas congeladas (Fine-tuning)
FROZEN_LAYERS: Dict[str, bool] = {}

def freeze_layer(name: str) -> None:
    """Impede que uma camada seja atualizada durante o treinamento (Fine-tuning)."""
    FROZEN_LAYERS[name] = True
    print(f"[FINE-TUNING] Camada '{name}' CONGELADA.")

def unfreeze_layer(name: str) -> None:
    """Permite a atualização de uma camada específica."""
    FROZEN_LAYERS.pop(name, None)
    print(f"[FINE-TUNING] Camada '{name}' DESCONGELADA.")

def is_layer_frozen(name: str) -> bool:
    """Verifica se a camada está no modo Read-Only."""
    return FROZEN_LAYERS.get(name, False)

def ensure_weights_dir() -> None:
    """Garante que o diretório de pesos exista no sistema de arquivos."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

def initialize_layer_weights(shape: Tuple[int, ...], name: str, init_type: str = 'xavier', dtype: Any = np.float32) -> Tuple[str, Tuple[int, ...]]:
    """
    Inicializa um tensor de peso e salva no disco.
    
    Args:
        shape: Dimensões do tensor.
        name: Nome identificador da camada.
        init_type: Tipo de inicialização ('xavier', 'zeros', 'normal').
        dtype: Tipo de dado numérico.
        
    Returns:
        Tuple[str, Tuple[int, ...]]: Caminho do arquivo salvo e o shape do tensor.
    """
    ensure_weights_dir()
    
    if init_type == 'xavier':
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

def save_tensor_logged(path: str, tensor: np.ndarray, name: str = "unknown") -> None:
    """Salva um tensor registrando a latência de escrita na telemetria."""
    start_time = time.time()
    np.save(path, tensor)
    latency = time.time() - start_time
    log_telemetry('io_write_latency', latency, f"file:{name}")

def create_weight_registry(layers_info: Dict[str, Any]) -> None:
    """
    Cria um registro JSON com os metadados de todos os tensores do modelo.
    
    Args:
        layers_info: Dicionário contendo metadados das camadas.
    """
    ensure_weights_dir()
    registry = {
        "model_name": "ZeroRAM-GEN-V0",
        "layers": layers_info
    }
    with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=4)
    print(f"Registro de pesos criado em: {REGISTRY_PATH}")

def get_layer_metadata(name: str) -> Optional[Dict[str, Any]]:
    """Recupera metadados de uma camada específica do registro global."""
    if not os.path.exists(REGISTRY_PATH):
        return None
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)
    return registry['layers'].get(name)

def load_tensor_mmap(name: str) -> np.ndarray:
    """Carrega um tensor do disco no modo memory-mapped (Zero RAM)."""
    meta = get_layer_metadata(name)
    if not meta:
        raise ValueError(f"Dados da camada '{name}' não encontrados.")
    
    start_time = time.time()
    tensor = np.load(meta['path'], mmap_mode='r')
    latency = time.time() - start_time
    log_telemetry('io_read_latency', latency, f"layer:{name}")
    
    return tensor

def dispose_tensor(tensor_obj: Any) -> None:
    """Remove a referência do objeto para auxiliar o Garbage Collector."""
    if tensor_obj is not None:
        del tensor_obj

def store_bias_vector(name: str, vector: np.ndarray, description: str = "") -> None:
    """Salva um vetor de viés comportamental e registra no banco de dados."""
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

def quantize_to_int8(tensor: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Converte um tensor para INT8 usando escala linear e ponto zero."""
    tensor_f32 = tensor.astype(np.float32)
    t_min, t_max = np.min(tensor_f32), np.max(tensor_f32)
    
    scale = (t_max - t_min) / 255.0 if t_max != t_min else 1.0
    zero_point = -t_min / scale - 128 if scale != 0 else 0
    
    q_tensor = np.round(tensor_f32 / scale + zero_point).clip(-128, 127).astype(np.int8)
    return q_tensor, float(scale), float(zero_point)

def dequantize_from_int8(q_tensor: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
    """Restaura um tensor INT8 para ponto flutuante."""
    return ((q_tensor.astype(np.float32) - zero_point) * scale).astype(DEFAULT_DTYPE)

def save_quantized_tensor(name: str, tensor: np.ndarray) -> None:
    """Quantiza e salva um tensor no disco acompanhado de seu arquivo .meta."""
    q_tensor, scale, zp = quantize_to_int8(tensor)
    
    path = os.path.join(WEIGHTS_DIR, f"{name}_int8.npy")
    np.save(path, q_tensor)
    
    meta_path = os.path.join(WEIGHTS_DIR, f"{name}.meta")
    with open(meta_path, 'w') as f:
        json.dump({"scale": scale, "zero_point": zp, "quantized": True}, f)
        
    print(f"[QUANT] '{name}' salvo como INT8 (Scale: {scale:.6f})")

def get_quant_params(name: str) -> Optional[Dict[str, Any]]:
    """Recupera metadados de quantização (.meta) de uma camada."""
    meta_path = os.path.join(WEIGHTS_DIR, f"{name}.meta")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None

def store_tensor_disk(name: str, tensor: np.ndarray, folder: str = 'temp', quantize: bool = False) -> str:
    """Salva um tensor (gradiente ou ativação) no disco, opcionalmente quantizado."""
    path = os.path.join(WEIGHTS_DIR, folder)
    os.makedirs(path, exist_ok=True)
    
    start_time = time.time()
    
    if quantize:
        q_tensor, scale, zp = quantize_to_int8(tensor)
        file_path = os.path.join(path, f"{name}_q.npy")
        np.save(file_path, q_tensor)
        meta_path = os.path.join(path, f"{name}_q.meta")
        with open(meta_path, 'w') as f:
            json.dump({"scale": scale, "zero_point": zp}, f)
    else:
        file_path = os.path.join(path, f"{name}.npy")
        np.save(file_path, tensor)
        
    latency = time.time() - start_time
    log_telemetry('io_write_latency', latency, f"{folder}:{name}")
    return file_path

def load_tensor_disk(name: str, folder: str = 'temp') -> Optional[np.ndarray]:
    """Carrega um tensor temporário do disco, suportando desquantização automática."""
    path = os.path.join(WEIGHTS_DIR, folder)
    q_file_path = os.path.join(path, f"{name}_q.npy")
    file_path = os.path.join(path, f"{name}.npy")
    
    start_time = time.time()
    
    if os.path.exists(q_file_path):
        q_tensor = np.load(q_file_path)
        meta_path = os.path.join(path, f"{name}_q.meta")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        tensor = dequantize_from_int8(q_tensor, meta['scale'], meta['zero_point'])
    elif os.path.exists(file_path):
        tensor = np.load(file_path)
    else:
        return None
        
    latency = time.time() - start_time
    log_telemetry('io_read_latency', latency, f"{folder}_load:{name}")
    return tensor

def reset_accumulated_grads() -> None:
    """Remove diretórios de gradientes e ativações temporárias do disco para limpeza."""
    for folder in ['grads', 'temp']:
        path = os.path.join(WEIGHTS_DIR, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
    print("Dados temporários limpos.")

def ensure_v0_weights() -> None:
    """Inicializa pesos padrão Xavier/Zeros se o diretório de pesos estiver vazio."""
    if not os.path.exists(WEIGHTS_DIR) or len(os.listdir(WEIGHTS_DIR)) <= 1:
        print("\n[!] Pesos não encontrados. Reinicializando...")
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

def decompose_weights_svd(weights: np.ndarray, rank_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompõe pesos via SVD para baixa aproximação (Low-Rank)."""
    w_f32 = weights.astype(np.float32)
    u, s, vh = np.linalg.svd(w_f32, full_matrices=False)
    
    k = max(1, int(len(s) * rank_ratio))
    print(f"[SVD] Rank: {len(s)} -> {k}")
    
    return u[:, :k].astype(DEFAULT_DTYPE), s[:k].astype(DEFAULT_DTYPE), vh[:k, :].astype(DEFAULT_DTYPE)

def save_svd_weights(name: str, weights: np.ndarray, rank_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompõe e salva componentes SVD (U, S, V) no disco."""
    u, s, v = decompose_weights_svd(weights, rank_ratio)
    
    base_path = os.path.join(WEIGHTS_DIR, f"{name}_svd")
    os.makedirs(base_path, exist_ok=True)
    
    np.save(os.path.join(base_path, "u.npy"), u)
    np.save(os.path.join(base_path, "s.npy"), s)
    np.save(os.path.join(base_path, "v.npy"), v)
    
    meta_path = os.path.join(WEIGHTS_DIR, f"{name}.svd_meta")
    with open(meta_path, 'w') as f:
        json.dump({"svd": True, "rank": u.shape[1], "original_shape": list(weights.shape)}, f)
        
    print(f"[SVD] '{name}' salvo como Low-Rank.")
    return u, s, v

def get_svd_params(name: str) -> Optional[Dict[str, Any]]:
    """Recupera metadados de SVD (.svd_meta)."""
    meta_path = os.path.join(WEIGHTS_DIR, f"{name}.svd_meta")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None

def save_compressed_tensor(name: str, tensor: np.ndarray) -> None:
    """Salva tensor comprimido com LZ4 para otimização de I/O de disco."""
    tensor_bytes = tensor.tobytes()
    compressed_data = lz4.frame.compress(tensor_bytes)
    
    path = os.path.join(WEIGHTS_DIR, f"{name}.lz4")
    with open(path, 'wb') as f:
        f.write(compressed_data)
    
    meta_path = os.path.join(WEIGHTS_DIR, f"{name}.lz4_meta")
    with open(meta_path, 'w') as f:
        json.dump({
            "compressed": True, 
            "original_shape": list(tensor.shape), 
            "dtype": str(tensor.dtype)
        }, f)
        
    print(f"[LZ4] '{name}' comprimido: {len(tensor_bytes)//1024}KB -> {len(compressed_data)//1024}KB")

def load_compressed_tensor(name: str) -> Optional[np.ndarray]:
    """Carrega e descompacta tensor LZ4 do disco."""
    path = os.path.join(WEIGHTS_DIR, f"{name}.lz4")
    meta_path = os.path.join(WEIGHTS_DIR, f"{name}.lz4_meta")
    
    if not os.path.exists(path) or not os.path.exists(meta_path):
        return None
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    start_time = time.time()
    with open(path, 'rb') as f:
        compressed_data = f.read()
    
    decompressed_data = lz4.frame.decompress(compressed_data)
    tensor = np.frombuffer(decompressed_data, dtype=meta['dtype']).reshape(meta['original_shape']).copy()
    
    latency = time.time() - start_time
    log_telemetry('io_read_latency', latency, f"lz4_load:{name}")
    return tensor

def create_tensor_shards(name: str, tensor: np.ndarray, ids_per_shard: int = 1000) -> None:
    """Divide um tensor gigante em fragmentos físicos (Shards) no disco."""
    num_elements = tensor.shape[0]
    num_shards = (num_elements + ids_per_shard - 1) // ids_per_shard
    
    shard_dir = os.path.join(WEIGHTS_DIR, f"{name}_shards")
    os.makedirs(shard_dir, exist_ok=True)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM shard_map WHERE tensor_name = ?", (name,))
        
        for i in range(num_shards):
            start_idx = i * ids_per_shard
            end_idx = min((i + 1) * ids_per_shard, num_elements)
            
            shard_data = tensor[start_idx:end_idx]
            shard_path = os.path.join(shard_dir, f"shard_{i}.npy")
            np.save(shard_path, shard_data)
            
            cursor.execute(
                "INSERT INTO shard_map (tensor_name, shard_id, start_index, end_index, file_path) VALUES (?, ?, ?, ?, ?)",
                (name, i, start_idx, end_idx, shard_path)
            )
        conn.commit()
    print(f"[SHARD] '{name}' fragmentado em {num_shards} unidades.")

def lookup_shard_for_id(tensor_name: str, original_id: int) -> Optional[Tuple[str, int]]:
    """Localiza o fragmento e o deslocamento interno para um ID global."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT shard_id, start_index, file_path FROM shard_map WHERE tensor_name = ? AND ? >= start_index AND ? < end_index",
            (tensor_name, original_id, original_id)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return row[2], original_id - row[1]

def calculate_weight_hash(path: str) -> Optional[str]:
    """Calcula o hash SHA256 de um arquivo de pesos."""
    import hashlib
    if not os.path.exists(path):
        return None
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(8192), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def verify_tensor_integrity(name: str) -> bool:
    """Verifica se a integridade do tensor no disco confere com o hash esperado."""
    meta = get_layer_metadata(name)
    if not meta or 'path' not in meta:
        return False
        
    current_hash = calculate_weight_hash(meta['path'])
    if 'hash' not in meta:
        return True # N/A
        
    return current_hash == meta['hash']

def convert_weights_to_fp16() -> None:
    """Converte pesos e otimizador para FP16 globalmente."""
    if not os.path.exists(REGISTRY_PATH): return
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)

    for name, meta in registry['layers'].items():
        path = meta['path']
        if os.path.exists(path):
            tensor = np.load(path)
            if tensor.dtype != np.float16:
                np.save(path, tensor.astype(np.float16))
                meta['dtype'] = 'float16'
    
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=4)
    print("[OK] Global FP16.")

def convert_weights_to_int8() -> None:
    """Quantiza pesos para INT8 globalmente."""
    if not os.path.exists(REGISTRY_PATH): return
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)

    for name, meta in registry['layers'].items():
        path = meta['path']
        if os.path.exists(path):
            tensor = np.load(path)
            save_quantized_tensor(name, tensor)
            meta['path'] = os.path.join(WEIGHTS_DIR, f"{name}_int8.npy")
            meta['dtype'] = 'int8'
            meta['quantized'] = True
    
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=4)
    print("[OK] Global INT8.")

def save_model_checkpoint(checkpoint_name: str) -> str:
    """
    Cria um snapshot completo do modelo (Pesos, Vocab e Registro) para uso posterior ou fine-tuning.
    """
    checkpoint_dir = os.path.join(os.path.dirname(WEIGHTS_DIR), 'checkpoints', checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 1. Copiar Pesos
    dest_weights = os.path.join(checkpoint_dir, 'weights')
    if os.path.exists(WEIGHTS_DIR):
        if os.path.exists(dest_weights):
            shutil.rmtree(dest_weights)
        shutil.copytree(WEIGHTS_DIR, dest_weights, ignore=shutil.ignore_patterns('temp', 'grads'))
    
    # 2. Copiar Banco de Dados (Vocab)
    from database_manager import DB_PATH
    shutil.copy2(DB_PATH, os.path.join(checkpoint_dir, 'vocab.db'))
    
    # 3. Gerar Manifesto
    manifest = {
        "checkpoint_id": checkpoint_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frozen_layers": list(FROZEN_LAYERS.keys())
    }
    with open(os.path.join(checkpoint_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"[CHECKPOINT] Modelo salvo com sucesso em: {checkpoint_dir}")
    return checkpoint_dir

def load_model_checkpoint(checkpoint_name: str) -> bool:
    """
    Restaura um checkpoint anterior, substituindo os pesos e o vocab atual.
    """
    checkpoint_dir = os.path.join(os.path.dirname(WEIGHTS_DIR), 'checkpoints', checkpoint_name)
    if not os.path.exists(checkpoint_dir):
        print(f"[ERRO] Checkpoint '{checkpoint_name}' não encontrado.")
        return False
    
    # Restaura Pesos
    src_weights = os.path.join(checkpoint_dir, 'weights')
    if os.path.exists(src_weights):
        if os.path.exists(WEIGHTS_DIR):
            shutil.rmtree(WEIGHTS_DIR)
        shutil.copytree(src_weights, WEIGHTS_DIR)
        
    # Restaura Vocab
    from database_manager import DB_PATH
    shutil.copy2(os.path.join(checkpoint_dir, 'vocab.db'), DB_PATH)
    
    # Restaura estado de Fine-tuning se houver no manifesto
    manifest_path = os.path.join(checkpoint_dir, 'manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            global FROZEN_LAYERS
            FROZEN_LAYERS = {layer: True for layer in manifest.get('frozen_layers', [])}
            
    print(f"[RELOAD] Modelo '{checkpoint_name}' carregado e pronto para uso/fine-tuning.")
    return True
