# Sprint 23: Modulação Comportamental - Etapa 1 (Prefix Tokens)

## Objetivos
- Implementar o sistema de Prefixos Comportamentais ("Persona").
- Desenvolver tokens especiais (e.g. `<|system|>`, `<|user|>`, `<|assistant|>`).

## Ferramentas & Pacotes
- **SQLite3 (Persona)**: Tabela `PROMPT_TEMPLATES`.
- **Special Indexing (SQL)**: IDs reservados para tokens de sistema.

## Funções e Implementações
- `inject_system_prefix(persona_name)`: Carrega o prompt de persona como os primeiros tokens da sequência.
- `map_special_tokens(text)`: Traduz tags `<|tag|>` em IDs específicos do vocabulário.
- `apply_chat_format(history)`: Formatar o diálogo para o estilo LLM Chat (User/Assistant).

## Detalhes Técnicos
A persona muda drasticamente o comportamento do modelo, mesmo com poucos parâmetros. O vocabulário deve aceitar esses marcadores.
