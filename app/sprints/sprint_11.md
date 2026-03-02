# Sprint 11: Implementação de Otimizadores (Optimizer) no Disco

## Objetivos
- Implementar o otimizador ADAM ou SGD para atualização dos pesos.
- Desenvolver o mecanismo de Momentum e Taxa de Aprendizado (Learning Rate) persistente.

## Ferramentas & Pacotes
- **NumPy (Math)**: Implementação das equações do Adam.
- **SQLite3 (Optimizer State)**: Tabela `OPTIM_STATE` para guardar os momentos M e V de cada peso.

## Funções e Implementações
- `adam_update_step(grad, weights, m, v, t)`: Atualização de pesos com base no estado do otimizador.
- `load_optim_state(layer_name)`: Recuperar momentos do SQLite.
- `save_optim_state(layer_name, m, v)`: Persistir estado do otimizador no Apache Parquet ou SQLite.

## Detalhes Técnicos
O estado do Adam (M e V) dobra o espaço em disco necessário para os pesos. O ZeroRAM-GEN deve carregar pesos, gradientes e estados de otimização em sincronia.
