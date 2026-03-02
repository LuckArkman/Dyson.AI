import os
import re
import io
import struct
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from core.vocabulary_manager import VocabularyManager

class BinaryTreeFileStorage:
    def __init__(self, path: str):
        self.path = path
        self.handle = open(self.path, "ab+") # Append + Read Binary
        self.offsets = [] # Lista de ponteiros no disco
        print(f"[BinaryTreeStorage] Base de dados de lotes em: {path}")

    def store_data(self, data: bytes) -> int:
        offset = self.handle.tell()
        self.handle.write(struct.pack("I", len(data))) # Header: Tamanho
        self.handle.write(data)                        # Data
        self.handle.flush()
        return offset

    def get_data_bytes(self, offset: int) -> bytes:
        self.handle.seek(offset)
        length = struct.unpack("I", self.handle.read(4))[0]
        return self.handle.read(length)

    def clear(self):
        self.handle.close()
        self.handle = open(self.path, "wb")
        self.handle.close()
        self.handle = open(self.path, "ab+")

    def dispose(self):
        self.handle.close()

class DatasetService:
    def __init__(self, swap_file_path: str):
        self.swap_file_path = swap_file_path
        batch_storage_path = os.path.join(os.path.dirname(swap_file_path) or "Dayson", "batches.bts")
        os.makedirs(os.path.dirname(batch_storage_path), exist_ok=True)
        self.batch_storage = BinaryTreeFileStorage(batch_storage_path)
        self.train_batch_offsets = []
        self.validation_batch_offsets = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()

    def dispose(self):
        if hasattr(self, 'batch_storage'):
            self.batch_storage.dispose()

    def initialize_and_split(self, dataset_path: str, context_window: int, vocab_manager: VocabularyManager, 
                             batch_size: int, validation_split: float):
        if os.path.exists(self.batch_storage.path) and os.path.getsize(self.batch_storage.path) > 1024 * 1024:
            print("[DatasetService] Lotes já existentes. Reconstruindo índices de busca...")
            self._rebuild_offsets(validation_split)
            return


        print("[DatasetService] Iniciando processamento de dataset em streaming...")
        self.batch_storage.clear()
        self.train_batch_offsets = []
        self.validation_batch_offsets = []
        
        token_count = 0
        all_indices = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                tokens = re.split(r'(\w+|[.,!?;:\'\"/\-])', line.lower())
                tokens = [t for t in tokens if t and not t.isspace()]
                for token in tokens:
                    all_indices.append(vocab_manager.get_token_index(token))
                    token_count += 1
                
                if token_count % 100000 == 0:
                    print(f"[DatasetService] Tokens processados: {token_count:,}")


        total_sequences = max(0, len(all_indices) - context_window - 1)
        if total_sequences == 0:
            raise ValueError("Dataset muito reduzido para esta janela de contexto.")

        validation_size = int(total_sequences * validation_split)
        train_size = total_sequences - validation_size
        
        # Geração dos lotes no disco
        self._generate_batches(all_indices, 0, train_size, context_window, batch_size, self.train_batch_offsets)
        self._generate_batches(all_indices, train_size, total_sequences, context_window, batch_size, self.validation_batch_offsets)

        all_indices.clear() # Libera RAM
        print(f"[DatasetService] Processamento concluído. RAM liberada.")
        print(f"[DatasetService] Lotes Treino: {len(self.train_batch_offsets)} | Lotes Validação: {len(self.validation_batch_offsets)}")


    def _rebuild_offsets(self, validation_split: float):
        self.train_batch_offsets = []
        self.validation_batch_offsets = []
        all_offsets = []
        
        with open(self.batch_storage.path, "rb") as f:
            while True:
                pos = f.tell()
                header = f.read(4)
                if not header: break
                length = struct.unpack("I", header)[0]
                all_offsets.append(pos)
                f.seek(length, 1) # Pula o corpo do lote

        val_count = int(len(all_offsets) * validation_split)
        self.train_batch_offsets = all_offsets[:len(all_offsets)-val_count]
        self.validation_batch_offsets = all_offsets[len(all_offsets)-val_count:]
        print(f"[DatasetService] Índices restaurados. Lotes: {len(all_offsets)}")

    def _generate_batches(self, data, start_index, count, context_window, batch_size, offsets_list):

        current_batch = []
        for i in range(count):
            abs_idx = start_index + i
            if abs_idx + context_window + 1 > len(data): break
            
            input_seq = data[abs_idx : abs_idx + context_window]
            target_seq = data[abs_idx + 1 : abs_idx + context_window + 1]
            current_batch.append((input_seq, target_seq))

            if len(current_batch) == batch_size:
                offsets_list.append(self._save_batch_to_disk(current_batch))
                current_batch = []

        if current_batch:
            offsets_list.append(self._save_batch_to_disk(current_batch))

    def _save_batch_to_disk(self, batch) -> int:
        with io.BytesIO() as buf:
            # Struct: [int count] -> [int seq_len][bytes input][int seq_len][bytes target]
            buf.write(struct.pack("i", len(batch)))
            for input_seq, target_seq in batch:
                input_arr = np.array(input_seq, dtype=np.int32)
                target_arr = np.array(target_seq, dtype=np.int32)
                
                buf.write(struct.pack("i", len(input_arr)))
                buf.write(input_arr.tobytes())
                buf.write(struct.pack("i", len(target_arr)))
                buf.write(target_arr.tobytes())
            
            return self.batch_storage.store_data(buf.getvalue())

    def load_batch_from_disk(self, offset: int):
        data = self.batch_storage.get_data_bytes(offset)
        with io.BytesIO(data) as buf:
            count = struct.unpack("i", buf.read(4))[0]
            batch = []
            for _ in range(count):
                in_len = struct.unpack("i", buf.read(4))[0]
                in_arr = np.frombuffer(buf.read(in_len * 4), dtype=np.int32)
                tg_len = struct.unpack("i", buf.read(4))[0]
                tg_arr = np.frombuffer(buf.read(tg_len * 4), dtype=np.int32)
                batch.append((in_arr, tg_arr))
            return batch
