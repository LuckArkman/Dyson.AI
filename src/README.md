# Galileu.Node - Migração para Python (FastAPI/PyOpenCL)

Esta é a migração completa da infraestrutura de nó descentralizado para Python, mantendo a performance e a arquitetura original.

## Tecnologias Implementadas
- **FastAPI:** API de alto desempenho para o nó e P2P.
- **PyOpenCL:** Implementação direta dos kernels C para cálculo na GPU.
- **NumPy:** Gestão eficiente de memória e arrays.
- **Zero-RAM Strategy:** Persistência de tensores por ID em arquivos `.bin`.
- **SQLite:** Vocabulário persistente para baixo uso de RAM.

## Estrutura do Projeto (src/)
- `main.py`: Ponto de entrada (FastAPI).
- `brain/`: Lógica LSTM e gestão de tensores.
- `core/`: Vocabulário e abstração de tensores.
- `gpu/`: Motor de cálculo PyOpenCL.
- `api/`: Controladores e rotas.

## Como Executar

### 1. Criar Ambiente e Instalar Dependências:
```bash
pip install -r requirements.txt
```

### 2. Rodar o Nó:
```bash
python main.py
```

### 3. Acessar a Documentação (Swagger):
[http://localhost:8000/docs](http://localhost:8000/docs)
