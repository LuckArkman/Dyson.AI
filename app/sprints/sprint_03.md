# Sprint 03: Mapeamento de Vocabulário no SQLite

## Objetivos
- Converter todos os tokens normalizados no ID correspondente no banco SQLite.
- Garantir que o token `<PAD>` (Preenchimento) seja o ID `0` e que não haja duplicidade de palavras no banco de dados.

## Ferramentas & Pacotes
- **SQLite3 (Indexação)**: Criação de índices únicos para otimização de consultas de texto.
- **Batches**: Inserção em massa (Commit por blocos de 10.000 tokens).

## Funções e Implementações
- `get_or_create_id(word)`: Função principal que verifica se a palavra já existe no banco e retorna seu ID ou insere nova.
- `bulk_insert_vocab(words)`: Melhora a performance de inserção de vocabulário inicial.
- `create_index_on_text()`: Criar índice na coluna `TEXT` para acelerar consultas SQL (`SELECT ID FROM VOCAB WHERE TEXT = ?`).

## Detalhes Técnicos
Crucial para o ZeroRAM-GEN, pois o treinamento fará consultas contínuas a esta tabela. A latência da consulta SQL deve ser minimizada.
