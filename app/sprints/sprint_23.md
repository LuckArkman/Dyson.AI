# Sprint 23: Modulação Comportamental - Etapa 1 (Prefix Tokens) [CONCLUÍDA]

## Objetivos
- [x] Implementar o sistema de Prefixos Comportamentais ("Persona").
- [x] Desenvolver tokens especiais (e.g. `<|system|>`, `<|user|>`, `<|assistant|>`).

## Ferramentas & Pacotes
- [x] **SQLite3 (Persona)**: Tabela `vocab` atualizada com marcadores especiais.
- [x] **Regex Utility**: `tokenizer.py` modificado para suportar preservação de tags.

## Funções e Implementações
- [x] `apply_chat_format(messages)`: Converte mensagens do LangChain para o formato de tokens do modelo.
- [x] `map_special_tokens(text)`: Mapeia sequências com tags para IDs de tokens correspondentes.
- [x] **Ajuste de Normalização**: Garantia de que tags como `<|system|>` sejam tratadas como átomos únicos.

## Detalhes Técnicos
A modulação comportamental agora permite que o modelo entenda quem está falando (Usuário, Sistema ou Assistente). Isso é fundamental para manter a coerência em diálogos e seguir instruções de persona.
