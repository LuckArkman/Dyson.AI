# Sprint 19: Geração Recursiva (Autoregressiva)

## Objetivos
- Implementar o loop de geração automática de texto (Próximo token entra como entrada do próximo passo).
- Definir limites de geração (Comprimento máximo, Token de fim de frase).

## Ferramentas & Pacotes
- **Python (Recursion/Looping)**: Estrutura de repetição.
- **Stop Tokens (SQL)**: Lista de tokens de interrupção ( `.` , `!` , `?` ).

## Funções e Implementações
- `autoregressive_gen_loop(prompt_ids, length)`: Gera tokens sequenciais até o limite.
- `concatenate_output(ids)`: Transforma a lista final de IDs em uma string legível.
- `detect_stop_token(token_id)`: Interrompe a geração ao encontrar o token de fim.

## Detalhes Técnicos
Cada token gerado requer um novo ciclo de Forward carregando os pesos do disco. O atraso acumulado por token é o maior gargalo.
