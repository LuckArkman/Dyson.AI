# Sprint 13: Subsistema de Inferência (Text Gen) [CONCLUÍDA]

## Objetivos
- [x] Implementar funções para geração de texto palavra a palavra (auto-regressiva).
- [x] Desenvolver mecanismos de Sampling (Greedy Search, Temperature, Top-K).

## Ferramentas & Pacotes
- [x] **NumPy**: Para amostragem multinomial de probabilidades.
- [x] **tokenizer.py**: Conversão de saída do modelo em texto humano.

## Funções e Implementações
- [x] `predict_next_token(ids, temp, top_k)`: Predição isolada de um único token.
- [x] `generate_text(prompt, max_tokens)`: Loop de geração mantendo contexto.

## Detalhes Técnicos
A inferência no ZeroRAM-GEN consome muito pouco recurso de RAM, permitindo rodar em dispositivos extremamente limitados, sacrificando velocidade de geração devido ao I/O de disco a cada token.
