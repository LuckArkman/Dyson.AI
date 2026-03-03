# Sprint 24: Embeddings Condicionais (Persona Embedding) [CONCLUÍDA]

## Objetivos
- [x] Implementar a camada de Embeddings condicionais para ajuste fino de comportamento em tempo de execução.
- [x] Desenvolver rotina de "Bias" comportamental por categoria (Criativo vs. Técnico).

## Ferramentas & Pacotes
- [x] **NumPy (Math)**: Soma vetorial de embeddings sob demanda (Zero RAM).
- [x] **SQLite3 (Modulation)**: Tabela `bias_templates` para gerenciar os vetores de viés.

## Funções e Implementações
- [x] `apply_behavioral_bias(embeddings, bias_name)`: Injeção de viés matemático nos vetores de embedding.
- [x] `store_bias_vector(name, vector)`: Persistência de arquétipos de persona no disco.
- [x] `initialize_default_biases()`: Criação de perfis padrão (Creative/Technical).

## Detalhes Técnicos
A injeção de viés nos embeddings permite alterar o estilo de resposta do modelo sem necessidade de retreinamento. Isso simula um "ajuste fino leve" em tempo real, aproveitando a maleabilidade dos vetores de alta dimensão.
