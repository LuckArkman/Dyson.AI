# Sprint 29: Validação Cruzada (Cross-check) com Modelos Professores

## Objetivos
- Implementar o sistema de "Teacher-Student" (Seção 3.3 do White Paper).
- Utilizar Google Gemini (ou equivalente) para gerar perguntas/respostas para o "Golden Data".

## Ferramentas & Pacotes
- **Google Generative AI API**: Client para Gemini 1.5/2.5.
- **SQLite3 (Evaluation)**: Tabela `GOLD_DATA`.

## Funções e Implementações
- `request_golden_data(prompt)`: Chamada à API para gerar dados sintéticos de treino de alta qualidade.
- `compare_outputs(val_data)`: Comparar frase gerada pelo ZeroRAM-GEN com a gerada pelo Gemini.
- `calculate_similarity_score(text1, text2)`: Medir a qualidade semântica da resposta.

## Detalhes Técnicos
O ZeroRAM-GEN amadurece "destilando" o conhecimento do modelo maior (Dyson-Brain) e gravando no `pt_0.txt`.
