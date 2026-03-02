# Sprint 04: Desserialização de Tokens (ID para Texto) [CONCLUÍDA]

## Objetivos
- [x] Implementar a função de reversão (ID -> Texto) para inferência.
- [x] Desenvolver rotinas de decodificação de sequências numéricas após o processamento.

## Ferramentas & Pacotes
- [x] **SQLite3 (Lookup)**: Consultas por IDs inteiros.
- [x] **Cache Local (LRU)**: Cache efêmero em RAM para os 5.000 tokens mais usados (Otimização).

## Funções e Implementações
- [x] `get_text_by_id(id)`: Retornar a string original correspondente a um ID no banco.
- [x] `decode_sequence(ids)`: Converter uma lista de IDs em uma frase humana legível.
- [x] `handle_unknown_token()`: Implementar token de Fallback (UNK) para palavras não encontradas.

## Detalhes Técnicos
Esta sprint garante que o "resultado" do ZeroRAM-GEN possa ser traduzido de volta para o usuário, fundamental para o monitoramento do treinamento.
