import numpy as np
from database_manager import get_or_create_id, get_text_by_id
from engine import embedding_lookup, dense_layer_forward, compute_softmax, apply_behavioral_bias
from tensor_manager import dispose_tensor
from tokenizer import normalize_text, tokenize_line, decode_sequence

def apply_temperature(logits, temperature):
    """Ajusta os logits com base na temperatura."""
    if temperature <= 0: return logits
    return logits / temperature

def top_k_sampling(probs, k):
    """Seleciona um token a partir dos K mais prováveis."""
    top_indices = np.argpartition(probs, -k)[-k:]
    top_probs = probs[top_indices]
    top_probs = top_probs / np.sum(top_probs) # Re-normaliza
    return int(np.random.choice(top_indices, p=top_probs))

def greedy_sampling(probs):
    """Seleciona o token mais provável (determinístico)."""
    return int(np.argmax(probs))

def predict_next_token(token_ids, temperature=1.0, top_k=None, bias_name=None):
    """
    Prediz o próximo token a partir de uma sequência de IDs.
    """
    # 1. Forward Pass Zero RAM
    emb = embedding_lookup(token_ids)
    
    # Sprint 24: Aplicação de Viés Comportamental
    if bias_name:
        emb = apply_behavioral_bias(emb, bias_name)
    
    # Hidden Layer
    h1_z = dense_layer_forward(emb, "hidden_01_weights", "hidden_01_bias", activation='relu')
    
    # Output Layer
    logits = dense_layer_forward(h1_z, "output_weights", activation='linear')
    
    # Pegar apenas o logit do último token da sequência
    last_step_logits = logits[-1, :]
    
    # 2. Amostragem (Sampling)
    last_step_logits = apply_temperature(last_step_logits, temperature)
    probs = compute_softmax(last_step_logits)
    
    if top_k is not None:
        next_id = top_k_sampling(probs, top_k)
    else:
        next_id = greedy_sampling(probs)
        
    # Limpeza
    dispose_tensor(emb); dispose_tensor(h1_z); dispose_tensor(logits)
    
    return next_id

def generate_text(prompt, max_new_tokens=10, temperature=1.0, top_k=None, stop_on_punctuation=True, bias_name=None):
    """
    Gera uma sequência de texto auto-regressiva com detecção de Stop Tokens e Viés.
    """
    # Tokenização do prompt
    tokens = tokenize_line(normalize_text(prompt))
    token_ids = [get_or_create_id(t) for t in tokens]
    
    # IDs de Stop Tokens comuns
    stop_words = {'.', '!', '?', '<PAD>'}
    
    generated_ids = list(token_ids)
    
    for i in range(max_new_tokens):
        # Janela de contexto deslizante (últimos 8 tokens)
        context_ids = generated_ids[-8:] 
        
        next_id = predict_next_token(context_ids, temperature, top_k, bias_name=bias_name)
        
        if next_id == 0: # <PAD>
            break
            
        generated_ids.append(next_id)
        
        # Verificar Stop Token
        if stop_on_punctuation:
            token_text = get_text_by_id(next_id)
            if token_text in stop_words:
                break
        
    return decode_sequence(generated_ids)

def stream_generate_text(prompt, max_new_tokens=10, temperature=1.0, top_k=None, stop_on_punctuation=True, bias_name=None):
    """
    Gerador que retorna tokens um a um em tempo real (Streaming).
    """
    # Tokenização do prompt
    tokens = tokenize_line(normalize_text(prompt))
    token_ids = [get_or_create_id(t) for t in tokens]
    
    stop_words = {'.', '!', '?', '<PAD>'}
    generated_ids = list(token_ids)
    
    for i in range(max_new_tokens):
        context_ids = generated_ids[-8:] 
        
        next_id = predict_next_token(context_ids, temperature, top_k, bias_name=bias_name)
        
        if next_id == 0: break
            
        generated_ids.append(next_id)
        token_text = get_text_by_id(next_id)
        
        yield token_text
        
        if stop_on_punctuation and token_text in stop_words:
            break
