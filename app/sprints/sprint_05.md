# Sprint 05: Setup das Dimensões dos Tensores em Disco [CONCLUÍDA]

## Objetivos
- [x] Definir a estrutura dos arquivos de pesos (.npy ou .dat).
- [x] Criar os arquivos iniciais para Embeddings, Pesos de Camada e Bia (Vazios ou aleatórios).

## Ferramentas & Pacotes
- [x] **NumPy (init)**: Gerador inicial de matrizes aleatórias (Xavier/He Initialization).
- [x] **FileSystem (OS)**: Gerenciar nomes de arquivos baseados na camada do modelo.

## Funções e Implementações
- [x] `initialize_layer_weights(shape, name)`: Criar o arquivo em disco com o nome da camada.
- [x] `create_weight_registry()`: Manter um JSON com os metadados de cada arquivo de peso (Dimensões, tipo de dado).

## Detalhes Técnicos
Os arquivos de pesos nunca residirão em RAM integralmente. Eles serão segmentados por matrizes correspondentes às fatias de cálculo. O diretório `weights/` deve ser criado.
