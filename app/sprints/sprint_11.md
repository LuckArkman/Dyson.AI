# Sprint 11: Implementação de Otimizadores (Optimizer) no Disco [CONCLUÍDA]

## Objetivos
- [x] Implementar o otimizador ADAM ou SGD para atualização dos pesos.
- [x] Desenvolver o mecanismo de Momentum e Taxa de Aprendizado (Learning Rate) persistente.

## Ferramentas & Pacotes
- [x] **NumPy (Math)**: Implementação das equações do Adam.
- [x] **SQLite3 (Optimizer State)**: Tabela `TRAIN_STATE` para guardar o passo global `t`.

## Funções e Implementações
- [x] `adam_update_step(layer_name, weights, grad)`: Atualização de pesos com base no estado do otimizador.
- [x] `get_optimizer_state(layer_name)`: Carrega momentos M e V do disco (ZeroRAM).
- [x] `save_optimizer_state(layer_name, m, v)`: Persiste estados M e V em arquivos `.npy` na pasta `optim`.

## Detalhes Técnicos
O estado do Adam (M e V) dobra o espaço em disco necessário para os pesos. O ZeroRAM-GEN deve carregar pesos, gradientes e estados de otimização em sincronia.
