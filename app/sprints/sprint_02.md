# Sprint 02: Pipeline de Tokenização - Etapa 1 (Ingestão) [CONCLUÍDA]

## Objetivos
- [x] Desenvolver o mecanismo de extração de tokens a partir do arquivo bruto `pt_0.txt`.
- [x] Implementar as rotinas de limpeza e normalização de texto (Letras minúsculas, remoção de caracteres especiais).

## Ferramentas & Pacotes
- [x] **Python Standard Library (re)**: Expressões regulares para tokenização avançada.
- [x] **PT_0.txt**: Dataset bruto disponível em `Dayson/pt_0.txt`.

## Funções e Implementações
- [x] `load_raw_data(path)`: Carregar o arquivo original de forma eficiente (Line-by-line para evitar estouro de memória).
- [x] `normalize_text(text)`: Limpeza de strings para garantir unicidade no vocabulário.
- [x] `tokenize_line(line)`: Divisão de frases em tokens unitários (Words/Subwords).

## Detalhes Técnicos
O foco desta sprint é a eficiência de leitura, preparando os dados para a indexação no SQLite. Deve ser compatível com o dataset de ~600.000 tokens.
