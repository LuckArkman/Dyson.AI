# Sprint 12: Loop de Treinamento - Sequenciamento de Batches [CONCLUÍDA]

## Objetivos
- [x] Implementar o iterador de dataset para o arquivo `pt_0.txt`.
- [x] Desenvolver o loop principal de épocas (Epochs) e passos (Steps).

## Ferramentas & Pacotes
- [x] **Python (itertools/generators)**: Iteradores de alta performance e economia de RAM.
- [x] **SQLite3 (Checkpointing)**: Salvar progresso de batche (ID do último token processado).

## Funções e Implementações
- [x] `sequence_generator(batch_size, seq_length)`: Gera fatias de dados (X, Y) do arquivo de texto.
- [x] `save_training_checkpoint(batch_id, epoch)`: Gravar no SQLite para retomar treino após falha.
- [x] `train_step(x, y)`: Orquestra o Forward, Backward e Update.

## Detalhes Técnicos
Batches muito grandes aumentarão a RAM proporcionalmente. O ZeroRAM-GEN deve manter o batch_size pequeno para garantir o limite de memória.
