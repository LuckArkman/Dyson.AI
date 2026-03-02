# Sprint 21: Orquestração via LangChain - Etapa 1 (Chain Básica)

## Objetivos
- Integrar o motor de inferência do ZeroRAM-GEN ao ecossistema LangChain.
- Implementar a Classe `ZeroRAMLLM(BaseLLM)` para compatibilidade de API.

## Ferramentas & Pacotes
- **LangChain (core)**: Biblioteca de orquestração de prompts.
- **Python (wrappers)**: Criação do wrapper customizado de LLM.

## Funções e Implementações
- `_call(prompt, stop)`: Implementação obrigatória da LangChain para gerar texto através do ZeroRAM-GEN.
- `_llm_type()`: Definição da ID do modelo no ecossistema ("zeroram-gen").
- `init_langchain_wrapper()`: Inicializa a conexão entre LangChain e o motor de disco.

## Detalhes Técnicos
A LangChain permitirá usar ferramentas de memória e encadeamento no futuro, tratando o ZeroRAM-GEN como um LLM padrão.
