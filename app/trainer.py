import os
import psutil
import shutil
import numpy as np
from database_manager import get_db_connection, log_training_metrics, log_telemetry
from tensor_manager import (
    load_tensor_disk, store_tensor_disk, dispose_tensor, 
    get_layer_metadata, load_tensor_mmap, save_tensor_logged,
    load_trained_weight, save_trained_weight
)
from engine import (
    embedding_lookup, dense_layer_forward, apply_activation,
    compute_softmax, calculate_loss, compute_output_gradient,
    backward_layer_step, d_relu, accumulate_embedding_grad
)
from optimizer import increment_training_step, adam_update_step
from vocab import serialize, deserialize

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
    
    # Acumular gradiente de embeddings (especial para Zero RAM)
    accumulate_embedding_grad(grad_emb, X_batch[:, -1])
    
    # --- UPDATE (ADAM) ---
    t = increment_training_step()
    
    # Telemetria: Gradiente Norm (Suficiente olhar para um exemplo)
    grad_norm = np.linalg.norm(grad_w_out)
    log_telemetry('grad_norm', grad_norm, 'layer:output_weights')
    
    # Telemetria: RAM Usage
    ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    log_telemetry('ram_usage_mb', ram_mb)
    
    # Atualizar Output Weights
    w_out = load_trained_weight("output_weights")
    new_w_out = adam_update_step("output_weights", w_out, grad_w_out)
    save_trained_weight("output_weights", new_w_out)
    
    # Atualizar Hidden Weights
    w_h1 = load_trained_weight("hidden_01_weights")
    new_w_h1 = adam_update_step("hidden_01_weights", w_h1, grad_w_h1)
    save_trained_weight("hidden_01_weights", new_w_h1)
    
    # Atualizar Bias
    b1 = load_trained_weight("hidden_01_bias")
    new_b1 = adam_update_step("hidden_01_bias", b1, grad_b_h1)
    save_trained_weight("hidden_01_bias", new_b1)
    
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

def check_early_stopping(current_loss, patience=50, min_delta=0.001):
    """
    Verifica o critério de Early Stopping e gerencia o 'Best Model'.
    Retorna True se o treinamento deve ser interrompido.
    """
    from tensor_manager import WEIGHTS_DIR
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Obter Melhor Loss anterior
        cursor.execute("SELECT value FROM train_state WHERE key = 'best_loss'")
        row = cursor.fetchone()
        best_loss = float(row[0]) if row else float('inf')
        
        # 2. Obter Contador de Paciência
        cursor.execute("SELECT value FROM train_state WHERE key = 'patience_counter'")
        row_p = cursor.fetchone()
        counter = int(row_p[0]) if row_p else 0
        
        # 3. Comparar
        if current_loss < (best_loss - min_delta):
            # Melhoria detectada
            print(f" [NEW BEST] Loss melhorou de {best_loss:.6f} para {current_loss:.6f}")
            best_loss = current_loss
            counter = 0
            
            # Salvar como Melhor Modelo (Copia os arquivos)
            best_dir = os.path.join(WEIGHTS_DIR, "best")
            os.makedirs(best_dir, exist_ok=True)
            for f in os.listdir(WEIGHTS_DIR):
                if f.endswith(".npy"):
                    shutil.copy2(os.path.join(WEIGHTS_DIR, f), os.path.join(best_dir, f))
            shutil.copy2(os.path.join(WEIGHTS_DIR, "weight_registry.json"), os.path.join(best_dir, "weight_registry.json"))
            print(f" [OK] Pesos persistidos em {best_dir}")
        else:
            counter += 1
            print(f" [ES] Sem melhoria significativa. Paciência: {counter}/{patience}")
            
        # 4. Atualizar Estado
        cursor.execute("INSERT OR REPLACE INTO train_state (key, value) VALUES ('best_loss', ?)", (str(best_loss),))
        cursor.execute("INSERT OR REPLACE INTO train_state (key, value) VALUES ('patience_counter', ?)", (str(counter),))
        conn.commit()
        
        return counter >= patience

def run_training_session(data_path, batch_size=4, seq_length=8, epochs=1, steps_per_epoch=None):
    """
    Orquestra uma sessão completa de treinamento com suporte a múltiplas épocas e early stopping.
    """
    from vocab import sequence_generator
    
    epoch_start, batch_start = load_training_checkpoint()
    print(f"\n>>> Iniciando Sessão de Treino: Época {epoch_start} a {epoch_start + epochs}")
    
    for epoch in range(epoch_start, epoch_start + epochs):
        print(f"\n--- Época {epoch} ---")
        gen = sequence_generator(data_path, batch_size, seq_length)
        
        current_step = 0
        epoch_loss = 0
        
        for X, Y in gen:
            current_step += 1
            
            # Pular se estivermos retomando de um batch específico
            if epoch == epoch_start and current_step <= batch_start:
                continue
                
            loss, t = train_step(X, Y)
            epoch_loss += loss
            
            if current_step % 10 == 0:
                print(f" [Step {current_step}] Loss: {loss:.4f} | Global T: {t}")
                log_training_metrics(epoch, current_step, loss)
                save_training_checkpoint(epoch, current_step)
            
            # Check Early Stopping
            if current_step % 50 == 0:
                if check_early_stopping(loss):
                    print(f" [STOP] Interrompendo por Early Stopping na Época {epoch}, Step {current_step}")
                    return
            
            if steps_per_epoch and current_step >= steps_per_epoch:
                break
        
        print(f"\nÉpoca {epoch} Concluída. Loss Média Estimada: {epoch_loss/current_step:.4f}")
        save_training_checkpoint(epoch + 1, 0) # Reinicia batch para próxima época
