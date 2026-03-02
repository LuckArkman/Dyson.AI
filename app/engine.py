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
    """Transforma logits (pesos brutos) em probabilidades (0 a 1)."""
    # Subtrair o máximo para estabilidade numérica (evita overflow)
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def calculate_loss(probabilities, target_ids):
    """
    Calcula Cross Entropy Loss.
    probabilities: Shape (seq_len, vocab_size)
    target_ids: Lista de IDs reais que deveriam ocorrer.
    """
    batch_size = len(target_ids)
    # Selecionar apenas a probabilidade atribuída ao token correto
    # probabilities[range(batch_size), target_ids] pega prob[i][target_id[i]]
    correct_log_probs = -np.log(probabilities[range(batch_size), target_ids] + 1e-10)
    return np.mean(correct_log_probs)
