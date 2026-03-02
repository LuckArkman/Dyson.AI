import pyopencl as cl
import numpy as np
import os
import re
import json
from typing import List, Tuple, Dict, Any, Optional

from core.tensor import Tensor, IndividualFileTensorManager
from gpu.gpu_math_engine import GpuMathEngine
from core.vocabulary_manager import VocabularyManager
from core.swap_manager import DiskSwapManager
from brain.adam_optimizer import AdamOptimizer

class ModelWeights:
    def __init__(self):
        self.embedding = None
        self.w_if = None; self.w_hf = None; self.b_f = None
        self.w_ii = None; self.w_hi = None; self.b_i = None
        self.w_ic = None; self.w_hc = None; self.b_c = None
        self.w_io = None; self.w_ho = None; self.b_o = None
        self.w_hy = None; self.b_y = None

class NeuralNetworkLSTM:
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, output_size: int, math_engine: GpuMathEngine):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.math_engine = math_engine
        
        # Sessão baseada na configuração para evitar conflitos de shape (broadcasting error)
        self.session_id = f"lstm_e{embedding_size}_h{hidden_size}_v{vocab_size}"
        
        self.tensor_manager = IndividualFileTensorManager(self.session_id)
        self.swap_manager = DiskSwapManager()
        
        self.weight_ids = self.initialize_weights()
        
        # Buffers Adam na VRAM (Persistent por sessão)
        self._m_gpu = {}
        self._v_gpu = {}
        for name, tid in self.weight_ids.items():
            shape = self.tensor_manager.load_tensor(tid).data.shape
            size = np.prod(shape)
            self._m_gpu[name] = self.math_engine.create_buffer(size=int(size * 4))
            self._v_gpu[name] = self.math_engine.create_buffer(size=int(size * 4))
            self.math_engine.fill(self._m_gpu[name], 0.0, size)
            self.math_engine.fill(self._v_gpu[name], 0.0, size)
        
        self._t = 0
        self.hidden_state_id = self.tensor_manager.store_tensor(Tensor((1, self.hidden_size)), "HiddenState")
        self.cell_state_id = self.tensor_manager.store_tensor(Tensor((1, self.hidden_size)), "CellState")

    def initialize_weights(self) -> Dict[str, str]:
        # Busca no cache da sessão por pesos já existentes
        config_path = os.path.join(self.tensor_manager.tensor_dir, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
                cached_vocab = config_data.get("vocab_size", 0)
                cached_hidden = config_data.get("hidden_size", 0)
                
                # SE O SHAPE MUDOU (Vocabulário ou Hidden), REGENERA TUDO
                if cached_vocab == self.vocab_size and cached_hidden == self.hidden_size:
                    print(f"[NeuralNetwork] Carregando pesos compatíveis do cache (Vocab: {self.vocab_size})")
                    return config_data["weight_ids"]
                else:
                    print(f"[Aviso] Inconsistência de Shape no Cache ({cached_vocab} vs {self.vocab_size}). Regenerando Pesos...")

        ids = {}
        def create_w(r, c, name):
            limit = np.sqrt(6.0 / (r + c))
            data = np.random.uniform(-limit, limit, (r, c)).astype(np.float32)
            return self.tensor_manager.store_tensor(Tensor((r, c), data), name)

        ids["W_embedding"] = create_w(self.vocab_size, self.embedding_size, "WeightsEmbedding")
        for gate in ["f", "i", "c", "o"]:
            ids[f"W_i{gate}"] = create_w(self.embedding_size, self.hidden_size, f"WeightsInput{gate}")
            ids[f"W_h{gate}"] = create_w(self.hidden_size, self.hidden_size, f"WeightsHidden{gate}")
            ids[f"B_{gate}"] = self.tensor_manager.store_tensor(Tensor((1, self.hidden_size)), f"Bias{gate}")
        
        ids["W_hy"] = create_w(self.hidden_size, self.output_size, "WeightsOutputFinal")
        ids["B_y"] = self.tensor_manager.store_tensor(Tensor((1, self.output_size)), "BiasOutputFinal")
        
        # Salva mapeamento E os hiperparâmetros para validação de forma futura
        with open(config_path, "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "weight_ids": ids
            }, f)
        return ids


    def save_model(self, model_path: str):
        config = {
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "weight_ids": self.weight_ids
        }
        with open(model_path, "w") as f:
            json.dump(config, f, indent=4)

    def get_model_weights(self) -> ModelWeights:
        w = ModelWeights()
        w.embedding = self.tensor_manager.load_tensor(self.weight_ids["W_embedding"]).data
        for gate in ["f", "i", "c", "o"]:
            setattr(w, f"w_i{gate}", self.tensor_manager.load_tensor(self.weight_ids[f"W_i{gate}"]).data)
            setattr(w, f"w_h{gate}", self.tensor_manager.load_tensor(self.weight_ids[f"W_h{gate}"]).data)
            setattr(w, f"b_{gate}", self.tensor_manager.load_tensor(self.weight_ids[f"B_{gate}"]).data)
        w.w_hy = self.tensor_manager.load_tensor(self.weight_ids["W_hy"]).data
        w.b_y = self.tensor_manager.load_tensor(self.weight_ids["B_y"]).data
        return w


    def train_batch(self, batch: List[Tuple[np.ndarray, np.ndarray]], learning_rate: float, weights: ModelWeights, w_gpu: dict = None) -> float:
        total_batch_loss = 0


        
        # Mapeamento para persistência RAM-GPU
        attr_map = {
            "W_embedding": "embedding", "W_if": "w_if", "W_hf": "w_hf", "B_f": "b_f",
            "W_ii": "w_ii", "W_hi": "w_hi", "B_i": "b_i", "W_ic": "w_ic", "W_hc": "w_hc", "B_c": "b_c",
            "W_io": "w_io", "W_ho": "w_ho", "B_o": "b_o", "W_hy": "w_hy", "B_y": "b_y"
        }

        if w_gpu is None:
            w_gpu = {name: self.math_engine.create_buffer(getattr(weights, attr_map[name])) for name in self.weight_ids}

        for input_indices, target_indices in batch:
            loss, swap_files = self.forward_pass_zero_ram(input_indices, target_indices, weights, w_gpu)
            total_batch_loss += loss
            grads = self.backward_pass_zero_ram(input_indices, target_indices, swap_files, weights)
            
            for name, grad_data in grads.items():
                param_buf = w_gpu[name]
                g_buf = self.math_engine.create_buffer(grad_data)
                size = grad_data.size
                
                self.math_engine.program.adam_update(
                    self.math_engine.queue, (size,), None,
                    param_buf, g_buf, self._m_gpu[name], self._v_gpu[name],
                    np.float32(learning_rate), np.float32(0.9), np.float32(0.999), 
                    np.float32(1e-8), np.int32(self._t + 1)
                )
                self.math_engine.release_buffer(g_buf)
            
            self._t += 1
            for f in swap_files: self.swap_manager.delete_swap_file(f)

        # Atualiza RAM e Disco no final da época/batch
        for name in w_gpu:
            data = np.empty_like(getattr(weights, attr_map[name]))
            cl.enqueue_copy(self.math_engine.queue, data, w_gpu[name])
            setattr(weights, attr_map[name], data)
            self.tensor_manager.overwrite_tensor(self.weight_ids[name], Tensor(data.shape, data))

        return total_batch_loss / len(batch)


    def forward_pass_zero_ram(self, input_indices, target_indices, weights: ModelWeights, w_gpu: dict):
        total_loss = 0
        swap_files = []
        
        h_prev_data = self.tensor_manager.load_tensor(self.hidden_state_id).data
        c_prev_data = self.tensor_manager.load_tensor(self.cell_state_id).data
        
        # Buffers de VRAM persistentes durante o loop para evitar overhead de criação
        h_prev_gpu = self.math_engine.create_buffer(h_prev_data)
        
        # Buffers para resultados dos portões
        z_f_gpu = self.math_engine.create_buffer(size=self.hidden_size * 4)
        z_i_gpu = self.math_engine.create_buffer(size=self.hidden_size * 4)
        z_c_gpu = self.math_engine.create_buffer(size=self.hidden_size * 4)
        z_o_gpu = self.math_engine.create_buffer(size=self.hidden_size * 4)

        for t in range(len(input_indices)):
            x_t_data = weights.embedding[input_indices[t]:input_indices[t]+1]
            x_t_gpu = self.math_engine.create_buffer(x_t_data)
            
            # Executa MatMuls na GPU para todos os portões (Aceleração Total)
            self.math_engine.matrix_multiply(x_t_gpu, w_gpu["W_if"], z_f_gpu, 1, self.embedding_size, self.hidden_size)
            self.math_engine.matrix_multiply(x_t_gpu, w_gpu["W_ii"], z_i_gpu, 1, self.embedding_size, self.hidden_size)
            self.math_engine.matrix_multiply(x_t_gpu, w_gpu["W_ic"], z_c_gpu, 1, self.embedding_size, self.hidden_size)
            self.math_engine.matrix_multiply(x_t_gpu, w_gpu["W_io"], z_o_gpu, 1, self.embedding_size, self.hidden_size)

            # Sincroniza e aplica Bias + Ativação (Manter em CPU por enquanto para estabilidade de BPTT)
            def compute_gate(buf_gpu, b_ram):
                res = np.empty((1, self.hidden_size), dtype=np.float32)
                cl.enqueue_copy(self.math_engine.queue, res, buf_gpu)
                return res + b_ram

            f_t = 1.0 / (1.0 + np.exp(-(compute_gate(z_f_gpu, weights.b_f) + np.dot(h_prev_data, weights.w_hf))))
            i_t = 1.0 / (1.0 + np.exp(-(compute_gate(z_i_gpu, weights.b_i) + np.dot(h_prev_data, weights.w_hi))))
            cc_t = np.tanh(compute_gate(z_c_gpu, weights.b_c) + np.dot(h_prev_data, weights.w_hc))
            o_t = 1.0 / (1.0 + np.exp(-(compute_gate(z_o_gpu, weights.b_o) + np.dot(h_prev_data, weights.w_ho))))

            c_t = f_t * c_prev_data + i_t * cc_t
            h_t = o_t * np.tanh(c_t)
            
            logits = np.dot(h_t, weights.w_hy) + weights.b_y
            logits_max = np.max(logits)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / (np.sum(exp_logits) + 1e-12)
            
            target_id = int(target_indices[t])
            if target_id >= probs.shape[1]: 
                target_id = 1 # Fallback para <UNK>
            
            total_loss += -np.log(probs[0, target_id] + 1e-9)

            # Swaps para BPTT
            swap_files.extend([
                self.swap_manager.swap_out(h_prev_data, f"h_p_{t}"), 
                self.swap_manager.swap_out(c_prev_data, f"c_p_{t}"),
                self.swap_manager.swap_out(probs, f"pr_{t}"), 
                self.swap_manager.swap_out(h_t, f"h_t_{t}"),
                self.swap_manager.swap_out(c_t, f"c_t_{t}"), 
                self.swap_manager.swap_out(f_t, f"f_t_{t}"),
                self.swap_manager.swap_out(i_t, f"i_t_{t}"), 
                self.swap_manager.swap_out(cc_t, f"cc_t_{t}"),
                self.swap_manager.swap_out(o_t, f"o_t_{t}"), 
                self.swap_manager.swap_out(x_t_data, f"x_t_{t}")
            ])

            
            h_prev_data, c_prev_data = h_t, c_t
            self.math_engine.release_buffer(x_t_gpu)

            
        # Limpeza final dos buffers GPU
        for b in [h_prev_gpu, z_f_gpu, z_i_gpu, z_c_gpu, z_o_gpu]: 
            self.math_engine.release_buffer(b)
            
        return total_loss / len(input_indices), swap_files




    def backward_pass_zero_ram(self, input_indices, target_indices, swap_files, weights: ModelWeights):
        grads = {name: np.zeros_like(self.tensor_manager.load_tensor(tid).data) for name, tid in self.weight_ids.items()}
        num_swaps = 10 # Swaps por passo de tempo
        
        dh_next = np.zeros((1, self.hidden_size))
        dc_next = np.zeros((1, self.hidden_size))
        
        for t in reversed(range(len(input_indices))):
            idx = t * num_swaps
            h_prev = self.swap_manager.load_from_swap(swap_files[idx])
            c_prev = self.swap_manager.load_from_swap(swap_files[idx+1])
            probs  = self.swap_manager.load_from_swap(swap_files[idx+2])
            h_t    = self.swap_manager.load_from_swap(swap_files[idx+3])
            c_t    = self.swap_manager.load_from_swap(swap_files[idx+4])
            f_t    = self.swap_manager.load_from_swap(swap_files[idx+5])
            i_t    = self.swap_manager.load_from_swap(swap_files[idx+6])
            cc_t   = self.swap_manager.load_from_swap(swap_files[idx+7])
            o_t    = self.swap_manager.load_from_swap(swap_files[idx+8])
            x_t    = self.swap_manager.load_from_swap(swap_files[idx+9])

            # 1. Output Gradients
            dy = probs.copy()
            target_id = int(target_indices[t])
            if target_id >= dy.shape[1]:
                target_id = 1 # Fallback
            dy[0, target_id] -= 1
            grads["W_hy"] += np.dot(h_t.T, dy)
            grads["B_y"] += dy


            # 2. Backprop into Hidden State
            dh = np.dot(dy, weights.w_hy.T) + dh_next
            
            # 3. Gate Derivations
            tanh_ct = np.tanh(c_t)
            do = dh * tanh_ct * o_t * (1 - o_t)
            dc = dh * o_t * (1 - tanh_ct**2) + dc_next
            df = dc * c_prev * f_t * (1 - f_t)
            di = dc * cc_t * i_t * (1 - i_t)
            dcc = dc * i_t * (1 - cc_t**2)

            # 4. Weight Gradients Acumulation
            def update_gate_grads(gate_prefix, d_gate):
                grads[f"W_i{gate_prefix}"] += np.dot(x_t.T, d_gate)
                grads[f"W_h{gate_prefix}"] += np.dot(h_prev.T, d_gate)
                grads[f"B_{gate_prefix}"] += d_gate

            update_gate_grads("f", df)
            update_gate_grads("i", di)
            update_gate_grads("c", dcc)
            update_gate_grads("o", do)
            
            # Embedding Gradient
            dx = np.dot(df, weights.w_if.T) + np.dot(di, weights.w_ii.T) + np.dot(dcc, weights.w_ic.T) + np.dot(do, weights.w_io.T)
            
            # Garante que o índice de embedding está no limite
            safe_id = int(target_indices[t])
            if safe_id < grads["W_embedding"].shape[0]:
                grads["W_embedding"][safe_id:safe_id+1] += dx
            
            # 5. Passthrough for next T
            dh_next = np.dot(df, weights.w_hf.T) + np.dot(di, weights.w_hi.T) + np.dot(dcc, weights.w_hc.T) + np.dot(do, weights.w_ho.T)
            dc_next = f_t * dc

        return grads



class GenerativeNeuralNetworkLSTM(NeuralNetworkLSTM):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, dataset_path: str, math_engine: GpuMathEngine):
        # Caminho absoluto para o vocabulário
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(os.path.dirname(base_dir), "Dayson", "vocab.db")
        
        self.vocab_manager = VocabularyManager(db_path)
        actual_vocab_size = self.vocab_manager.build_vocabulary(dataset_path)
        
        final_vocab_size = max(vocab_size, actual_vocab_size)
        super().__init__(final_vocab_size, embedding_size, hidden_size, final_vocab_size, math_engine)

    def generate_response(self, input_text: str, max_length: int = 50) -> str:
        tokens = re.split(r'(\w+|[.,!?;:\'\"/\-])', input_text.lower())
        tokens = [t for t in tokens if t and not t.isspace()]
        input_ids = [self.vocab_manager.get_token_index(t) for t in tokens]
        
        output_tokens = []
        weights = self.get_model_weights()
        h = np.zeros((1, self.hidden_size))
        c = np.zeros((1, self.hidden_size))

        # Seed the network
        curr_id = 1
        if input_ids: curr_id = input_ids[-1]

        for _ in range(max_length):
            # Forward step simples para inferência
            x = weights.embedding[curr_id : curr_id+1]
            
            # Gate f
            f = 1.0 / (1.0 + np.exp(-(np.dot(x, weights.w_if) + np.dot(h, weights.w_hf) + weights.b_f)))
            # Gate i
            i = 1.0 / (1.0 + np.exp(-(np.dot(x, weights.w_ii) + np.dot(h, weights.w_hi) + weights.b_i)))
            # Gate cc
            cc = np.tanh(np.dot(x, weights.w_ic) + np.dot(h, weights.w_hc) + weights.b_c)
            # Cell update
            c = f * c + i * cc
            # Gate o
            o = 1.0 / (1.0 + np.exp(-(np.dot(x, weights.w_io) + np.dot(h, weights.w_ho) + weights.b_o)))
            # Hidden update
            h = o * np.tanh(c)
            
            logits = np.dot(h, weights.w_hy) + weights.b_y
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)
            
            # Amostragem greedy para teste
            curr_id = np.argmax(probs)
            if curr_id <= 1: break # <PAD> ou <UNK>
            
            word = self.vocab_manager.get_token(int(curr_id))
            output_tokens.append(word)
            
        return " ".join(output_tokens)
