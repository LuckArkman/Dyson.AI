# Sprint 12: Loop de Treinamento - Sequenciamento de Batches

## Objetivos
- Implementar o iterador de dataset para o arquivo `pt_0.txt`.
- Desenvolver o loop principal de épocas (Epochs) e passos (Steps).

## Ferramentas & Pacotes
- **Python (itertools)**: Iteradores de alta performance.
- **SQLite3 (Checkpointing)**: Salvar progresso de batche (ID do último token processado).

## Funções e Implementações
- `dataset_iterator(batch_size, seq_length)`: Gera fatias de dados (X, Y) do arquivo de texto.
- `checkpoint_save(batch_id, epoch)`: Gravar no SQLite para retomar treino após falha.
- `training_main_loop(max_epochs)`: Orquestra o Forward, Backward e Update.

## Detalhes Técnicos
Batches muito grandes aumentarão a RAM proporcionalmente. O ZeroRAM-GEN deve manter o batch_size pequeno para garantir o limite de memória.
