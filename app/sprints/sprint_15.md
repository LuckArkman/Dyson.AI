# Sprint 15: Conclusão do Motor de Treinamento (MVP)

## Objetivos
- Executar o primeiro treinamento real (Epoch 0) com o dataset `pt_0.txt`.
- Verificar se o ZeroRAM-GEN está atualizando pesos corretamente e reduzindo a Loss.

## Ferramentas & Pacotes
- **TensorFlow (Grafo de validação)**: Cross-check de resultados matemáticos (Controle de qualidade).
- **SQLite3 (Resultados V0)**: Guardar pesos iniciais treinados.

## Funções e Implementações
- `run_first_epoch(dataset_path)`: Treino piloto de 1.000 batches para verificação de convergência.
- `verify_gradient_flow()`: Sanity check para garantir que o erro está chegando à camada de Embeddings.
- `export_v0_model()`: Congelar pesos treinados em diretório `models/v0/`.

## Detalhes Técnicos
Primeiro teste de ponta a ponta: Tokenização (SQL) -> Forward (Disco) -> Loss -> Backward (Disco) -> Update (Adam/SQL) -> Save (Disco).
