from tensor_manager import (
    load_tensor_mmap, dispose_tensor, get_quant_params, 
    dequantize_from_int8, get_svd_params, WEIGHTS_DIR,
    load_compressed_tensor, lookup_shard_for_id
)
import numpy as np
import os
from typing import List, Optional, Any, Union

def relu(x: np.ndarray) -> np.ndarray:
    """Função de Ativação ReLU (Rectified Linear Unit)."""
    return np.maximum(0, x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Função de Ativação Sigmoid."""
    return 1 / (1 + np.exp(-x))

def relu6(x: np.ndarray) -> np.ndarray:
    """Função de Ativação ReLU6 (limitada a 6)."""
    return np.minimum(np.maximum(0, x), 6)

def d_relu(x: np.ndarray) -> np.ndarray:
    """Derivada da função ReLU para backpropagation."""
    return (x > 0).astype(np.float32)

def apply_activation(tensor: np.ndarray, act_type: str = 'relu') -> np.ndarray:
    """
    Aplica a função de ativação especificada a um tensor.
    
    Args:
        tensor: Tensor de entrada.
        act_type: Tipo de ativação ('relu', 'sigmoid', 'relu6', 'linear').
        
    Returns:
        np.ndarray: Tensor ativado.
    """
    if act_type == 'relu':
        return relu(tensor)
    elif act_type == 'sigmoid':
        return sigmoid(tensor)
    elif act_type == 'relu6':
        return relu6(tensor)
    else:
        return tensor # Linear / No activation

def embedding_lookup(token_ids: Union[List[int], np.ndarray]) -> np.ndarray:
    """
    Busca os vetores de embedding para uma lista de IDs de tokens.
    Suporta Sharding (Segmentação física) se a matriz estiver fragmentada.
    
    Args:
        token_ids: Lista ou array de IDs de tokens.
        
    Returns:
        np.ndarray: Matriz de vetores brutos (ou dequantizados).
    """
    # 1. Tentar buscar via Sharding
    first_shard_info = lookup_shard_for_id("embedding_matrix", token_ids[0])
    
    if first_shard_info:
        all_vectors = []
        for tid in token_ids:
            shard_path, internal_idx = lookup_shard_for_id("embedding_matrix", tid)
            shard_data = np.load(shard_path, mmap_mode='r')
            all_vectors.append(shard_data[internal_idx].copy())
            dispose_tensor(shard_data)
        vectors = np.array(all_vectors)
    else:
        embed_matrix = load_tensor_mmap("embedding_matrix")
        vectors = embed_matrix[token_ids]
        dispose_tensor(embed_matrix)
    
    # 2. Dequantizar se o arquivo .meta existir
    q_params = get_quant_params("embedding_matrix")
    if q_params and q_params.get('quantized'):
        vectors = dequantize_from_int8(vectors, q_params['scale'], q_params['zero_point'])
    
    return vectors

def apply_behavioral_bias(embeddings: np.ndarray, bias_name: str) -> np.ndarray:
    """
    Aplica um vetor de viés (Bias) comportamental aos embeddings.
    
    Args:
        embeddings: Matriz de embeddings original.
        bias_name: Nome do template de viés (ex: 'creative').
        
    Returns:
        np.ndarray: Embeddings modificados.
    """
    from tensor_manager import load_tensor_disk
    try:
        bias_vector = load_tensor_disk(bias_name, folder='bias')
        if bias_vector is not None:
            embeddings += bias_vector
            dispose_tensor(bias_vector)
    except Exception as e:
        print(f"[BIAS_ERROR] {e}")
    return embeddings

def dense_layer_forward(input_tensor: np.ndarray, weights_name: str, bias_name: Optional[str] = None, activation: str = 'relu') -> np.ndarray:
    """
    Executa o Forward de uma camada densa carregando componentes sob demanda do disco.
    Prioridades de I/O: LZ4 (Comprimido) > SVD (Low-Rank) > MMap (Quantizado ou FP).
    
    Args:
        input_tensor: Tensor de entrada.
        weights_name: Nome da matriz de pesos.
        bias_name: Nome do vetor de bias opcional.
        activation: Nome da função de ativação.
        
    Returns:
        np.ndarray: Resultado do cálculo da camada.
    """
    # 1. Checar compressão LZ4
    lz4_meta_path = os.path.join(WEIGHTS_DIR, f"{weights_name}.lz4_meta")
    is_lz4 = os.path.exists(lz4_meta_path)

    if is_lz4:
        weights = load_compressed_tensor(weights_name)
        output = np.dot(input_tensor, weights)
        dispose_tensor(weights)
    else:
        # 2. Checar SVD (Low-Rank)
        svd_meta = get_svd_params(weights_name)
        if svd_meta and svd_meta.get('svd'):
            u = np.load(os.path.join(WEIGHTS_DIR, f"{weights_name}_svd", "u.npy"), mmap_mode='r')
            s = np.load(os.path.join(WEIGHTS_DIR, f"{weights_name}_svd", "s.npy"), mmap_mode='r')
            v = np.load(os.path.join(WEIGHTS_DIR, f"{weights_name}_svd", "v.npy"), mmap_mode='r')
            
            inter = np.dot(input_tensor, u)
            output = np.dot(inter, np.diag(s) @ v)
            
            dispose_tensor(u); dispose_tensor(s); dispose_tensor(v)
        else:
            # 3. MMap Normal ou Quantizado
            weights_mmap = load_tensor_mmap(weights_name)
            q_params = get_quant_params(weights_name)
            if q_params and q_params.get('quantized'):
                weights = dequantize_from_int8(weights_mmap, q_params['scale'], q_params['zero_point'])
            else:
                weights = weights_mmap
            output = np.dot(input_tensor, weights)
            dispose_tensor(weights)
    
    # 4. Bias
    if bias_name:
        bias = load_tensor_mmap(bias_name)
        output += bias
        dispose_tensor(bias)
        
    # 5. Ativação
    output = apply_activation(output, activation)
    return output

def compute_softmax(logits: np.ndarray) -> np.ndarray:
    """Transforma logits em probabilidades usando Softmax estável."""
    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return np.clip(probs, 1e-7, 1.0)

def compute_loss(probs: np.ndarray, target_ids: np.ndarray) -> float:
    """Calcula a Categorical Cross-Entropy (Perda)."""
    batch_size = probs.shape[0]
    target_probs = probs[np.arange(batch_size), target_ids]
    loss: float = -np.mean(np.log(target_probs))
    return loss

def dense_layer_backward(dout: np.ndarray, input_tensor: np.ndarray, weights_name: str) -> np.ndarray:
    """
    Backpropagation para uma camada densa (Zero RAM).
    Calcula os gradientes e os armazena no disco para posterior atualização.
    
    Args:
        dout: Gradiente vindo da camada posterior.
        input_tensor: Ativação salva do passo Forward.
        weights_name: Nome da camada para identificação de pesos.
        
    Returns:
        np.ndarray: Gradiente (din) para retropropagar para a camada anterior.
    """
    from tensor_manager import store_tensor_disk, is_layer_frozen
    
    # 0. Verificar se a camada está congelada (Fine-tuning)
    if is_layer_frozen(weights_name):
        weights = load_tensor_mmap(weights_name)
        din = np.dot(dout, weights.T)
        dispose_tensor(weights)
        return din

    # 1. Carregar pesos para cálculo de din
    weights = load_tensor_mmap(weights_name)
    
    # 2. Calcular Gradientes (dw = x^T * dout, db = sum(dout))
    # Para batch size > 1, usamos dot product. x deve ter shape (batch, in), dout (batch, out)
    dw = np.dot(input_tensor.T, dout)
    db = np.sum(dout, axis=0)
    
    # 3. Calcular Gradiente de Entrada (din = dout * w^T)
    din = np.dot(dout, weights.T)
    
    # 4. Persistir Gradientes (Zero RAM - Escrita Direta no Disco)
    store_tensor_disk(f"{weights_name}_dw", dw, folder='grads')
    store_tensor_disk(f"{weights_name}_db", db, folder='grads')
    
    # 5. Liberar RAM
    dispose_tensor(weights)
    
    return din
