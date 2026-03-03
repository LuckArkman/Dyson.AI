import numpy as np
from database_manager import get_or_create_id, get_text_by_id
from engine import embedding_lookup, dense_layer_forward, compute_softmax
from tensor_manager import dispose_tensor
from tokenizer import normalize_text, tokenize_line, decode_sequence

def predict_next_token(token_ids, temperature=1.0, top_k=None):
    """
    Prediz o próximo token a partir de uma sequência de IDs.
    """
    # 1. Forward Pass Zero RAM
    emb = embedding_lookup(token_ids)
    
    # Hidden Layer
    h1_z = dense_layer_forward(emb, "hidden_01_weights", "hidden_01_bias", activation='relu')
    
    # Output Layer
    logits = dense_layer_forward(h1_z, "output_weights", activation='linear')
    
    # Pegar apenas o logit do último token da sequência
    last_step_logits = logits[-1, :]
    
    # 2. Amostragem (Sampling)
    if temperature != 1.0:
        last_step_logits = last_step_logits / temperature
        
    probs = compute_softmax(last_step_logits)
    
    if top_k is not None:
        # Pega os índices dos top_k maiores valores
        top_indices = np.argpartition(probs, -top_k)[-top_k:]
        top_probs = probs[top_indices]
        top_probs = top_probs / np.sum(top_probs) # Re-normaliza
        next_id = np.random.choice(top_indices, p=top_probs)
    else:
        # Amostragem simples (Multinomial) ou Greedy (argmax)
        # Vamos usar Greedy para consistência no teste inicial
        next_id = np.argmax(probs)
        
    # Limpeza
    dispose_tensor(emb); dispose_tensor(h1_z); dispose_tensor(logits)
    
    return int(next_id)

def generate_text(prompt, max_new_tokens=10, temperature=1.0, top_k=None):
    """
    Gera uma sequência de texto a partir de um prompt.
    """
    print(f"Gerando texto para: '{prompt}'...")
    
    # Tokenização do prompt
    tokens = tokenize_line(normalize_text(prompt))
    token_ids = [get_or_create_id(t) for t in tokens]
    
    generated_ids = list(token_ids)
    
    for _ in range(max_new_tokens):
        # Para modelos simples, podemos precisar limitar o contexto (ex: últimos 8 tokens)
        context_ids = generated_ids[-8:] 
        
        next_id = predict_next_token(context_ids, temperature, top_k)
        generated_ids.append(next_id)
        
        # Token <PAD> ou STOP? Se o modelo prever o fim da frase.
        # Aqui geramos até o limite.
        
    return decode_sequence(generated_ids)
