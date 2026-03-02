# Sprint 08: Funções de Perda (Loss) no ZeroRAM-GEN

## Objetivos
- Implementar o cálculo de Cross Entropy (Entropia Cruzada) entre o Next Token e o Target.
- Desenvolver rotina de Log de Loss e armazenamento em disco.

## Ferramentas & Pacotes
- **NumPy**: Para cálculo de Softmax e Cross Entropy.
- **SQLite3 (Logging)**: Tabela `TRAIN_LOG` para persistência de progresso.

## Funções e Implementações
- `compute_softmax(logits)`: Transformar pesos brutos em probabilidades de tokens.
- `calculate_loss(predicted, actual)`: Medir o erro do modelo em relação ao dataset brutos (`pt_0.txt`).
- `log_training_metrics(epoch, step, loss)`: Gravar no SQLite para análise posterior.

## Detalhes Técnicos
A Loss deve ser calculada de forma segmentada (Per-batch) para evitar acúmulo de tensores em memória RAM.
