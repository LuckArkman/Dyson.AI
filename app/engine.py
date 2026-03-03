from tensor_manager import (
    load_tensor_mmap, dispose_tensor, get_quant_params, 
    dequantize_from_int8, get_svd_params, WEIGHTS_DIR,
    load_compressed_tensor, lookup_shard_for_id
)
import numpy as np
import os

def relu(x):
    """Função de Ativação ReLU."""
    return np.maximum(0, x)

def sigmoid(x):
    """Função de Ativação Sigmoid."""
    return 1 / (1 + np.exp(-x))

def relu6(x):
    """Função de Ativação ReLU6 (limitada a 6)."""
    return np.minimum(np.maximum(0, x), 6)

def d_relu(x):
    """Derivada da ReLU."""
    return (x > 0).astype(np.float32)

def apply_activation(tensor, act_type='relu'):
    """Aplica a função de ativação especificada a um tensor."""
    if act_type == 'relu':
        return relu(tensor)
    elif act_type == 'sigmoid':
        return sigmoid(tensor)
    elif act_type == 'relu6':
        return relu6(tensor)
    else:
        return tensor # Linear / No activation

def embedding_lookup(token_ids):
    """
    Busca os vetores de embedding para uma lista de IDs.
    Suporta Sharding (Segmentação) se a matriz estiver fragmentada.
    """
    # 1. Tentar buscar primeiro via Sharding
    # Para simplificar, verificamos o primeiro ID. Se tiver shard, assumimos modo segmentado.
    first_shard_info = lookup_shard_for_id("embedding_matrix", token_ids[0])
    
    if first_shard_info:
        # Modo Segmentado (Sharded)
        all_vectors = []
        # Nota: Em produção, agruparíamos por shard_path para otimizar I/O
        for tid in token_ids:
            shard_path, internal_idx = lookup_shard_for_id("embedding_matrix", tid)
            shard_data = np.load(shard_path, mmap_mode='r')
            all_vectors.append(shard_data[internal_idx].copy())
            dispose_tensor(shard_data)
        vectors = np.array(all_vectors)
    else:
        # Modo Tradicional (Arquivo Único via MMap)
        embed_matrix = load_tensor_mmap("embedding_matrix")
        vectors = embed_matrix[token_ids]
        dispose_tensor(embed_matrix)
    
    # 3. Dequantizar se necessário (Sprint 31)
    q_params = get_quant_params("embedding_matrix")
    if q_params and q_params.get('quantized'):
        vectors = dequantize_from_int8(vectors, q_params['scale'], q_params['zero_point'])
    
    return vectors

def apply_behavioral_bias(embeddings, bias_name):
    """
    Aplica um vetor de viés (Bias) comportamental aos embeddings.
    Zero RAM: O viés é carregado sob demanda.
    """
    from tensor_manager import load_tensor_disk
    
    # Carrega o vetor de viés do disco
    try:
        bias_vector = load_tensor_disk(bias_name, folder='bias')
        embeddings = embeddings + bias_vector
        dispose_tensor(bias_vector)
    except:
        pass
        
    return embeddings

def dense_layer_forward(input_tensor, weights_name, bias_name=None, activation='relu'):
    """
    Executa o Forward de uma camada densa carregando pesos sob demanda.
    Suporta SVD (Aproximação de Baixo Rank) e Quantização INT8.
    """
    # 0. Checar se existe SVD para esta camada (Prioridade de I/O)
    svd_meta = get_svd_params(weights_name)
    
    # Checar se existe compressão LZ4
    lz4_meta_path = os.path.join(WEIGHTS_DIR, f"{weights_name}.lz4_meta")
    is_lz4 = os.path.exists(lz4_meta_path)

    if is_lz4:
        # Prioridade para descompressão LZ4 (CPU é rápida, Disco é lento)
        weights = load_compressed_tensor(weights_name)
        output = np.dot(input_tensor, weights)
        dispose_tensor(weights)
    elif svd_meta and svd_meta.get('svd'):
        # Forward via Low-Rank: x * (U * S * V) -> (x * U) * (S * V) ou similar
        # Aqui fazemos: (input * U) * (diag(S) * V)
        u = np.load(os.path.join(WEIGHTS_DIR, f"{weights_name}_svd", "u.npy"), mmap_mode='r')
        s = np.load(os.path.join(WEIGHTS_DIR, f"{weights_name}_svd", "s.npy"), mmap_mode='r')
        v = np.load(os.path.join(WEIGHTS_DIR, f"{weights_name}_svd", "v.npy"), mmap_mode='r')
        
        # Produto aproximado
        # Passo 1: input * U
        inter = np.dot(input_tensor, u)
        # Passo 2: inter * (diag(S) * V)
        output = np.dot(inter, np.diag(s) @ v)
        
        dispose_tensor(u); dispose_tensor(s); dispose_tensor(v)
    else:
        # 1. Carregar pesos e bias via MMap (Fluxo Normal ou Quantizado)
        weights_mmap = load_tensor_mmap(weights_name)
        
        # 2. Dequantizar se necessário
        q_params = get_quant_params(weights_name)
        if q_params and q_params.get('quantized'):
            weights = dequantize_from_int8(weights_mmap, q_params['scale'], q_params['zero_point'])
        else:
            weights = weights_mmap

        # 3. Operação de Produto Escalar (Dot Product)
        output = np.dot(input_tensor, weights)
        dispose_tensor(weights)
    
    # 4. Aplicar Bias se existir
    if bias_name:
        bias = load_tensor_mmap(bias_name)
        output += bias
        dispose_tensor(bias)
        
    # 5. Aplicar Ativação
    output = apply_activation(output, activation)
    
    return output

def compute_softmax(logits):
    """Transforma logits em probabilidades com estabilidade para FP16."""
    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return np.clip(probs, 1e-7, 1.0)

def compute_loss(probs, target_ids):
    """Calcula a Categorical Cross-Entropy (Zero RAM)."""
    batch_size = probs.shape[0]
    # Pegar apenas as probabilidades dos índices alvo
    target_probs = probs[np.arange(batch_size), target_ids]
    loss = -np.mean(np.log(target_probs))
    return loss
