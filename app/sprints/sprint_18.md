# Sprint 18: Geração de Próximo Token (Top-k Sampling)

## Objetivos
- Implementar o mecanismo de amostragem (Sampling) para escolher o próximo ID baseado em probabilidades.
- Desenvolver rotina de "Temperatura" para controlar a criatividade (Softmax Temp).

## Ferramentas & Pacotes
- **NumPy (Probabilistic Sampling)**: Funções de escolha baseadas em pesos.
- **SQLite3 (Lookup)**: Conversão final do ID escolhido para texto.

## Funções e Implementações
- `apply_temperature(logits, temp)`: Redimensionar probabilidades (Macia vs. Determinística).
- `top_k_sampling(probs, k)`: Filtrar os K melhores candidatos para evitar tokens incoerentes.
- `greedy_sampling(probs)`: Escolher o token mais provável (Determinístico).

## Detalhes Técnicos
Essencial para a qualidade do texto do ZeroRAM-GEN. O equilíbrio da temperatura é a chave para o comportamento humano do modelo.
