# Sprint 33: Decomposição de Tensores (SVD)

## Objetivos
- Reduzir a dimensionalidade dos tensores de peso usando Singular Value Decomposition (SVD).
- Implementar o "Low-Rank" para acelerar o cálculo matricial.

## Ferramentas & Pacotes
- **NumPy (linalg.svd)**: Funções de decomposição.
- **Python (Compression)**: Comprimir duas matrizes menores em vez de uma grande.

## Funções e Implementações
- `decompose_weights(weights, rank_threshold)`: Dividir pesos em matrizes U, S e V.
- `dot_svd_approximation(input, u, s, v)`: Executar Forward através da aproximação de baixo rank.
- `measure_reconstruction_error(w, u, s, v)`: Avaliar perda de fidelidade.

## Detalhes Técnicos
Isso permite carregar menos dados do disco por camada, reduzindo o I/O, embora exija dois cálculos menores de produto escalar.
