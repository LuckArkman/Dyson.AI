# Sprint 16: Motor de Inferência - Etapa 1 (Prompt de Entrada)

## Objetivos
- Desenvolver a interface para receber o prompt do usuário.
- Implementar a função de mapeamento de caracteres de entrada para tokens (SQLite).

## Ferramentas & Pacotes
- **Python (input/argparse)**: Para recepção do texto.
- **SQLite3 (Lookup)**: Tradução do prompt do usuário em IDs inteiros.

## Funções e Implementações
- `get_user_prompt()`: Carregar input textual para o motor Zero RAM.
- `map_prompt_to_tokens(prompt_text)`: Tradução string->int baseada em vocabulário.
- `pad_prompt(tokens, length)`: Ajustar prompt para o formato de entrada esperado pelo modelo.

## Detalhes Técnicos
A inferência no ZeroRAM-GEN usa os mesmos mecanismos de carregamento sob demanda do treinamento. O prompt deve ser tratado como uma "sequência de IDs".
