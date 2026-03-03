# Sprint 18: Geração de Próximo Token (Top-k Sampling) [CONCLUÍDA]

## Objetivos
- [x] Implementar o mecanismo de amostragem (Sampling) para escolher o próximo ID baseado em probabilidades.
- [x] Desenvolver rotina de "Temperatura" para controlar a criatividade (Softmax Temp).

## Ferramentas & Pacotes
- [x] **NumPy (Probabilistic Sampling)**: Funções de escolha baseadas em pesos.
- [x] **Inference Module**: Modularização das técnicas de amostragem.

## Funções e Implementações
- [x] `apply_temperature(logits, temp)`: Redimensionar probabilidades (Macia vs. Determinística).
- [x] `top_k_sampling(probs, k)`: Filtrar os K melhores candidatos para evitar tokens incoerentes.
- [x] `greedy_sampling(probs)`: Escolher o token mais provável (Determinístico).

## Detalhes Técnicos
O motor de inferência foi refatorado para suportar diferentes estratégias de amostragem. O suporte a FP16 foi mantido, garantindo que o `Softmax` com temperatura não cause instabilidades numéricas.
