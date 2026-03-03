# Sprint 19: Geração Recursiva (Autoregressiva) [CONCLUÍDA]

## Objetivos
- [x] Implementar o loop de geração automática de texto (Próximo token entra como entrada do próximo passo).
- [x] Definir limites de geração (Comprimento máximo, Token de fim de frase).

## Ferramentas & Pacotes
- [x] **Python (Recursion/Looping)**: Estrutura de repetição no motor de inferência.
- [x] **Stop Tokens (SQL)**: Detecção de sinais de pontuação para interrupção natural.

## Funções e Implementações
- [x] `generate_text(...)`: Loop auto-regressivo com janela de contexto deslizante.
- [x] **Lógica de Parada**: Interrupção ao detectar tokens como `.`, `!` ou `?`.

## Detalhes Técnicos
A geração auto-regressiva foi validada. O motor agora consome os tokens gerados para prever os subsequentes, mantendo a coerência local. O uso de pesos em FP16 durante este processo cíclico não apresentou degradação perceptível na lógica do loop.
