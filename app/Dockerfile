# Dockerfile para ZeroRAM-GEN - Produção
FROM python:3.9-slim

# Instalar dependências de sistema (SQLite e build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /app

# Copiar arquivos de requisitos e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante da aplicação
COPY . .

# Criar diretórios para pesos e banco de dados
RUN mkdir -p /app/weights /app/Dayson && chmod -R 777 /app/weights /app/Dayson

# Expor portas se necessário (ex: para o Swarm/Rede na Sprint 36+)
EXPOSE 8000

# Variável de ambiente para garantir que os logs saiam em tempo real
ENV PYTHONUNBUFFERED=1

# Comando para inicializar o banco e aguardar instruções ou rodar o motor
CMD ["python", "database_manager.py"]
