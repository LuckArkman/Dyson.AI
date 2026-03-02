# Sprint 04: Desserialização de Tokens (ID para Texto)

## Objetivos
- Implementar a função de reversão (ID -> Texto) para inferência.
- Desenvolver rotinas de decodificação de sequências numéricas após o processamento.

## Ferramentas & Pacotes
- **SQLite3 (Lookup)**: Consultas por IDs inteiros.
- **Cache Local (LRU)**: Cache efêmero em RAM para os 5.000 tokens mais usados (Otimização).

## Funções e Implementações
- `get_text_by_id(id)`: Retornar a string original correspondente a um ID no banco.
- `decode_sequence(ids)`: Converter uma lista de IDs em uma frase humana legível.
- `handle_unknown_token()`: Implementar token de Fallback (UNK) para palavras não encontradas.

## Detalhes Técnicos
Esta sprint garante que o "resultado" do ZeroRAM-GEN possa ser traduzido de volta para o usuário, fundamental para o monitoramento do treinamento.
