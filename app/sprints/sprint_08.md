# Sprint 08: Funções de Perda (Loss) no ZeroRAM-GEN [CONCLUÍDA]

## Objetivos
- [x] Implementar o cálculo de Cross Entropy (Entropia Cruzada) entre o Next Token e o Target.
- [x] Desenvolver rotina de Log de Loss e armazenamento em disco.

## Ferramentas & Pacotes
- [x] **NumPy**: Para cálculo de Softmax e Cross Entropy.
- [x] **SQLite3 (Logging)**: Tabela `TRAIN_LOG` para persistência de progresso.

## Funções e Implementações
- [x] `compute_softmax(logits)`: Transformar pesos brutos em probabilidades de tokens.
- [x] `calculate_loss(predicted, actual)`: Medir o erro do modelo em relação ao dataset brutos (`pt_0.txt`).
- [x] `log_training_metrics(epoch, step, loss)`: Gravar no SQLite para análise posterior.

## Detalhes Técnicos
A Loss deve ser calculada de forma segmentada (Per-batch) para evitar acúmulo de tensores em memória RAM.
