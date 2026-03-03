import os
import numpy as np
from database_manager import get_db_connection, log_training_metrics
from tensor_manager import (
    load_tensor_disk, store_tensor_disk, dispose_tensor, 
    get_layer_metadata, load_tensor_mmap, save_tensor_logged
)
from engine import (
    embedding_lookup, dense_layer_forward, apply_activation,
    compute_softmax, calculate_loss, compute_output_gradient,
    backward_layer_step, d_relu, accumulate_embedding_grad
)
from optimizer import increment_training_step, adam_update_step
import psutil
from database_manager import log_telemetry

def train_step(X_batch, Y_batch):
    """Executa um único passo de treinamento (Forward -> Backward -> Update)."""
    
    # --- FORWARD PASS ---
    # Camada 1: Embedding
    emb = embedding_lookup(X_batch) # Shape: (batch, seq, dim)
    # No ZeroRAM, para camadas densas, achatamos o batch*seq se necessário
    # ou processamos a sequência. Aqui assumimos batch de sequências.
    
    # Camada 2: Hidden
    h1_z = dense_layer_forward(emb, "hidden_01_weights", "hidden_01_bias", activation='linear')
    h1_act = apply_activation(h1_z, 'relu')
    
    # Camada 3: Output (Logits)
    logits = dense_layer_forward(h1_act, "output_weights", activation='linear')
    
    # Tomar apenas o último logit da sequência se for Many-to-One
    # ou tratar como sequência. Para o ZeroRAM-GEN Text Gen, comparamos o final da seq com Y.
    # Se X tem seq_len, Y é o token seq_len + 1.
    last_logits = logits[:, -1, :] # Pegar o último token de cada sequência no batch
    
    probs = compute_softmax(last_logits)
    loss = calculate_loss(probs, Y_batch)
    
    # --- BACKWARD PASS ---
    grad_output = compute_output_gradient(probs, Y_batch)
    
    # Gradientes Output -> Hidden
    grad_w_out, grad_b_out, grad_h1_act = backward_layer_step(
        grad_output, "output_weights", h1_act[:, -1, :] # Apenas o gradiente do último passo propaga
    )
    
    # Gradiente Hidden -> Embedding (Simplificado para o último step da seq)
    # d_relu espera Z. 
    grad_h1_z = grad_h1_act * d_relu(h1_z[:, -1, :])
    grad_w_h1, grad_b_h1, grad_emb = backward_layer_step(
        grad_h1_z, "hidden_01_weights", emb[:, -1, :]
    )
    
    # --- UPDATE (ADAM) ---
    t = increment_training_step()
    
    # Telemetria: Gradiente Norm (Suficiente olhar para um exemplo)
    grad_norm = np.linalg.norm(grad_w_out)
    log_telemetry('grad_norm', grad_norm, 'layer:output_weights')
    
    # Telemetria: RAM Usage
    ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    log_telemetry('ram_usage_mb', ram_mb)
    
    # Atualizar Output Weights
    meta_out = get_layer_metadata("output_weights")
    w_out = np.load(meta_out['path'])
    new_w_out = adam_update_step("output_weights", w_out, grad_w_out)
    save_tensor_logged(meta_out['path'], new_w_out, "output_weights")
    
    # Atualizar Hidden Weights
    meta_h1 = get_layer_metadata("hidden_01_weights")
    w_h1 = np.load(meta_h1['path'])
    new_w_h1 = adam_update_step("hidden_01_weights", w_h1, grad_w_h1)
    save_tensor_logged(meta_h1['path'], new_w_h1, "hidden_01_weights")
    
    # Atualizar Bias
    meta_b1 = get_layer_metadata("hidden_01_bias")
    b1 = np.load(meta_b1['path'])
    new_b1 = adam_update_step("hidden_01_bias", b1, grad_b_h1)
    save_tensor_logged(meta_b1['path'], new_b1, "hidden_01_bias")
    
    # Limpeza
    dispose_tensor(emb); dispose_tensor(h1_z); dispose_tensor(h1_act)
    dispose_tensor(logits); dispose_tensor(probs); dispose_tensor(grad_output)
    
    return loss, t

def save_training_checkpoint(epoch, batch_id):
    """Salva o progresso do treinamento no SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO train_state (key, value) VALUES ('last_epoch', ?)", (str(epoch),))
        cursor.execute("INSERT OR REPLACE INTO train_state (key, value) VALUES ('last_batch_id', ?)", (str(batch_id),))
        conn.commit()

def load_training_checkpoint():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM train_state WHERE key = 'last_epoch'")
        epoch = cursor.fetchone()
        cursor.execute("SELECT value FROM train_state WHERE key = 'last_batch_id'")
        batch = cursor.fetchone()
        return (int(epoch[0]) if epoch else 0, int(batch[0]) if batch else 0)
