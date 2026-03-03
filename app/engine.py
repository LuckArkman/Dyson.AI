import numpy as np
from tensor_manager import load_tensor_mmap, dispose_tensor

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
    Mantém o princípio Zero RAM ao carregar apenas o que for fatiado.
    """
    # Carregar matriz de embeddings em modo mmap (Zero RAM residente)
    embed_matrix = load_tensor_mmap("embedding_matrix")
    
    # Slicing múltiplo retorna uma cópia em RAM apenas dos vetores solicitados
    vectors = embed_matrix[token_ids]
    
    # Liberar o mapeamento da matriz gigante imediatamente
    dispose_tensor(embed_matrix)
    
    return vectors

def apply_behavioral_bias(embeddings, bias_name):
    """
    Aplica um vetor de viés (Bias) comportamental aos embeddings.
    Zero RAM: O viés é carregado sob demanda.
    """
    from tensor_manager import load_tensor_disk, get_layer_metadata
    
    # Carrega o vetor de viés do disco
    # Assumindo que o bias tem o mesmo shape da dimensão de embedding (128)
    try:
        bias_vector = load_tensor_disk(bias_name, folder='bias')
        # Soma o viés a todos os embeddings na sequência/batch
        # Broadcast de numpy resolve o shape (batch, seq, dim) + (dim)
        embeddings = embeddings + bias_vector
        dispose_tensor(bias_vector)
    except:
        # Se o viés não existir, retorna os embeddings originais silenciosamente
        pass
        
    return embeddings

def dense_layer_forward(input_tensor, weights_name, bias_name=None, activation='relu'):
    """
    Executa o Forward de uma camada densa carregando pesos sob demanda.
    """
    # 1. Carregar pesos e bias via MMap
    weights = load_tensor_mmap(weights_name)
    
    # 2. Operação de Produto Escalar (Dot Product)
    # Nota: O cálculo traz temporariamente o resultado para a RAM
    output = np.dot(input_tensor, weights)
    
    # 3. Aplicar Bias se existir
    if bias_name:
        bias = load_tensor_mmap(bias_name)
        output += bias
        dispose_tensor(bias)
        
    # 4. Aplicar Ativação
    output = apply_activation(output, activation)
    
    # 5. Limpeza de memória dos pesos gigantes
    dispose_tensor(weights)
    
    return output

def compute_softmax(logits):
    """Transforma logits em probabilidades com estabilidade para FP16."""
    # Subtrair o máximo para evitar overflow no exp
    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    # Clip para evitar zeros absolutos que quebram o log na Loss
    return np.clip(probs, 1e-7, 1.0)

def calculate_loss(probabilities, target_ids):
    """Calcula Cross Entropy Loss com proteção contra log(0)."""
    batch_size = len(target_ids)
    selected_probs = probabilities[range(batch_size), target_ids]
    # Garantir que não temos zero no log
    return -np.mean(np.log(selected_probs + 1e-7))

def compute_output_gradient(probabilities, target_ids):
    """
    Calcula o gradiente inicial (dL/dz) para Softmax + CrossEntropy.
    Retorno: Shape (seq_len, vocab_size)
    """
    grad = probabilities.copy()
    batch_size = len(target_ids)
    # dL/dz = P - Y (onde Y é one-hot)
    grad[range(batch_size), target_ids] -= 1
    grad = grad / batch_size
    return grad

def backward_layer_step(grad_out, weights_name, forward_input):
    """
    Calcula gradientes para pesos, bias e para a entrada da camada.
    """
    from tensor_manager import load_tensor_mmap, dispose_tensor
    
    weights = load_tensor_mmap(weights_name)
    
    # dL/dW = X.T @ grad_out
    grad_w = np.dot(forward_input.T, grad_out)
    
    # dL/db = sum(grad_out) ao longo do eixo do batch
    grad_b = np.sum(grad_out, axis=0) if grad_out.ndim > 1 else grad_out
    
    # dL/dX = grad_out @ W.T
    grad_x = np.dot(grad_out, weights.T)
    
    dispose_tensor(weights)
    
    return grad_w, grad_b, grad_x

def accumulate_embedding_grad(token_ids, grad_emb_input, vocab_size, embed_dim):
    """
    Calcula o gradiente esparso para a matriz de embeddings.
    token_ids: IDs usados no forward pass.
    grad_emb_input: Gradiente vindo da camada superior (dL/dEmbed).
    """
    # No ZeroRAM, o gradiente do embedding é uma matriz (vocab_size, embed_dim)
    # mas quase toda é zero. Aqui criamos a versão esparsa (apenas fatias).
    grad_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    
    # Para cada token ID no batch, acumulamos o gradiente correspondente
    for i, tid in enumerate(token_ids):
        grad_matrix[tid] += grad_emb_input[i]
        
    return grad_matrix
