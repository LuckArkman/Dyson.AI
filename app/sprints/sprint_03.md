# Sprint 03: Mapeamento de Vocabulário no SQLite [CONCLUÍDA]

## Objetivos
- [x] Converter todos os tokens normalizados no ID correspondente no banco SQLite.
- [x] Garantir que o token `<PAD>` (Preenchimento) seja o ID `0` e que não haja duplicidade de palavras no banco de dados.

## Ferramentas & Pacotes
- [x] **SQLite3 (Indexação)**: Criação de índices únicos para otimização de consultas de texto.
- [x] **Batches**: Inserção em massa (Commit por blocos de 5.000 tokens).

## Funções e Implementações
- [x] `get_or_create_id(word)`: Função principal que verifica se a palavra já existe no banco e retorna seu ID ou insere nova.
- [x] `bulk_insert_vocab(words)`: Melhora a performance de inserção de vocabulário inicial.
- [x] `create_index_on_text()`: Criar índice na coluna `TEXT` para acelerar consultas SQL (`SELECT ID FROM VOCAB WHERE TEXT = ?`).

## Detalhes Técnicos
Crucial para o ZeroRAM-GEN, pois o treinamento fará consultas contínuas a esta tabela. A latência da consulta SQL deve ser minimizada.
