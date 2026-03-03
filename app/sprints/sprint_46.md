# Sprint 46: Ingestão Massiva de Dados e Expansão do Vocabulário

## Objetivos
- [ ] Desenvolver um processador de corpus para leitura de grandes arquivos de texto.
- [ ] Implementar a tokenização em lote (bulk) e inserção eficiente no SQLite.
- [ ] Validar a integridade do vocabulário após a ingestão de milhões de tokens.

## Ferramentas & Pacotes
- **Python (itertools / collections)**: Para processamento eficiente de streams de texto.
- **SQLite3**: Inserções em lote (`executemany`) e gerenciamento de transações.

## Funções e Implementações
- `stream_corpus(file_path)`: Gerador para leitura linha a linha de arquivos gigantes.
- `bulk_tokenize_and_store(tokens)`: Sistema de inserção em massa com cache de IDs.
- `verify_vocab_coverage()`: Relatório de densidade do vocabulário.

## Detalhes Técnicos
O ZeroRAM-GEN precisa de "combustível" (dados). Nesta sprint, focamos em como transformar gigabytes de texto bruto em uma tabela `vocab` indexada e pronta para o treinamento sem estourar a RAM do servidor de ingestão.
