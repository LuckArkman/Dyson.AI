# ZeroRAM-GEN: Documentação Técnica da API

Este documento descreve as principais funções do motor de disco, quantização e rede.

## Motor de Inferência (engine.py)
### `apply_activation(tensor: numpy.ndarray, act_type: str = 'relu') -> numpy.ndarray`
Aplica a função de ativação especificada a um tensor.

Args:
    tensor: Tensor de entrada.
    act_type: Tipo de ativação ('relu', 'sigmoid', 'relu6', 'linear').
    
Returns:
    np.ndarray: Tensor ativado.

### `apply_behavioral_bias(embeddings: numpy.ndarray, bias_name: str) -> numpy.ndarray`
Aplica um vetor de viés (Bias) comportamental aos embeddings.

Args:
    embeddings: Matriz de embeddings original.
    bias_name: Nome do template de viés (ex: 'creative').
    
Returns:
    np.ndarray: Embeddings modificados.

### `compute_loss(probs: numpy.ndarray, target_ids: numpy.ndarray) -> float`
Calcula a Categorical Cross-Entropy (Perda).

### `compute_softmax(logits: numpy.ndarray) -> numpy.ndarray`
Transforma logits em probabilidades usando Softmax estável.

### `d_relu(x: numpy.ndarray) -> numpy.ndarray`
Derivada da função ReLU para backpropagation.

### `dense_layer_forward(input_tensor: numpy.ndarray, weights_name: str, bias_name: str | None = None, activation: str = 'relu') -> numpy.ndarray`
Executa o Forward de uma camada densa carregando componentes sob demanda do disco.
Prioridades de I/O: LZ4 (Comprimido) > SVD (Low-Rank) > MMap (Quantizado ou FP).

Args:
    input_tensor: Tensor de entrada.
    weights_name: Nome da matriz de pesos.
    bias_name: Nome do vetor de bias opcional.
    activation: Nome da função de ativação.
    
Returns:
    np.ndarray: Resultado do cálculo da camada.

### `embedding_lookup(token_ids: List[int] | numpy.ndarray) -> numpy.ndarray`
Busca os vetores de embedding para uma lista de IDs de tokens.
Suporta Sharding (Segmentação física) se a matriz estiver fragmentada.

Args:
    token_ids: Lista ou array de IDs de tokens.
    
Returns:
    np.ndarray: Matriz de vetores brutos (ou dequantizados).

### `relu(x: numpy.ndarray) -> numpy.ndarray`
Função de Ativação ReLU (Rectified Linear Unit).

### `relu6(x: numpy.ndarray) -> numpy.ndarray`
Função de Ativação ReLU6 (limitada a 6).

### `sigmoid(x: numpy.ndarray) -> numpy.ndarray`
Função de Ativação Sigmoid.

## Gerenciador de Tensores (tensor_manager.py)
### `__annotate__(format, /)`
Sem descrição.

### `calculate_weight_hash(path: str) -> str | None`
Calcula o hash SHA256 de um arquivo de pesos.

### `convert_weights_to_fp16() -> None`
Converte pesos e otimizador para FP16 globalmente.

### `convert_weights_to_int8() -> None`
Quantiza pesos para INT8 globalmente.

### `create_tensor_shards(name: str, tensor: numpy.ndarray, ids_per_shard: int = 1000) -> None`
Divide um tensor gigante em fragmentos físicos (Shards) no disco.

### `create_weight_registry(layers_info: Dict[str, Any]) -> None`
Cria um registro JSON com os metadados de todos os tensores do modelo.

Args:
    layers_info: Dicionário contendo metadados das camadas.

### `decompose_weights_svd(weights: numpy.ndarray, rank_ratio: float = 0.5) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]`
Decompõe pesos via SVD para baixa aproximação (Low-Rank).

### `dequantize_from_int8(q_tensor: numpy.ndarray, scale: float, zero_point: float) -> numpy.ndarray`
Restaura um tensor INT8 para ponto flutuante.

### `dispose_tensor(tensor_obj: Any) -> None`
Remove a referência do objeto para auxiliar o Garbage Collector.

### `ensure_v0_weights() -> None`
Inicializa pesos padrão Xavier/Zeros se o diretório de pesos estiver vazio.

### `ensure_weights_dir() -> None`
Garante que o diretório de pesos exista no sistema de arquivos.

### `get_layer_metadata(name: str) -> Dict[str, Any] | None`
Recupera metadados de uma camada específica do registro global.

### `get_quant_params(name: str) -> Dict[str, Any] | None`
Recupera metadados de quantização (.meta) de uma camada.

### `get_svd_params(name: str) -> Dict[str, Any] | None`
Recupera metadados de SVD (.svd_meta).

### `initialize_layer_weights(shape: Tuple[int, ...], name: str, init_type: str = 'xavier', dtype: Any = <class 'numpy.float32'>) -> Tuple[str, Tuple[int, ...]]`
Inicializa um tensor de peso e salva no disco.

Args:
    shape: Dimensões do tensor.
    name: Nome identificador da camada.
    init_type: Tipo de inicialização ('xavier', 'zeros', 'normal').
    dtype: Tipo de dado numérico.
    
Returns:
    Tuple[str, Tuple[int, ...]]: Caminho do arquivo salvo e o shape do tensor.

### `load_compressed_tensor(name: str) -> numpy.ndarray | None`
Carrega e descompacta tensor LZ4 do disco.

### `load_tensor_disk(name: str, folder: str = 'temp') -> numpy.ndarray | None`
Carrega um tensor temporário do disco, suportando desquantização automática.

### `load_tensor_mmap(name: str) -> numpy.ndarray`
Carrega um tensor do disco no modo memory-mapped (Zero RAM).

### `lookup_shard_for_id(tensor_name: str, original_id: int) -> Tuple[str, int] | None`
Localiza o fragmento e o deslocamento interno para um ID global.

### `quantize_to_int8(tensor: numpy.ndarray) -> Tuple[numpy.ndarray, float, float]`
Converte um tensor para INT8 usando escala linear e ponto zero.

### `reset_accumulated_grads() -> None`
Remove diretórios de gradientes e ativações temporárias do disco para limpeza.

### `save_compressed_tensor(name: str, tensor: numpy.ndarray) -> None`
Salva tensor comprimido com LZ4 para otimização de I/O de disco.

### `save_quantized_tensor(name: str, tensor: numpy.ndarray) -> None`
Quantiza e salva um tensor no disco acompanhado de seu arquivo .meta.

### `save_svd_weights(name: str, weights: numpy.ndarray, rank_ratio: float = 0.5) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]`
Decompõe e salva componentes SVD (U, S, V) no disco.

### `save_tensor_logged(path: str, tensor: numpy.ndarray, name: str = 'unknown') -> None`
Salva um tensor registrando a latência de escrita na telemetria.

### `store_bias_vector(name: str, vector: numpy.ndarray, description: str = '') -> None`
Salva um vetor de viés comportamental e registra no banco de dados.

### `store_tensor_disk(name: str, tensor: numpy.ndarray, folder: str = 'temp', quantize: bool = False) -> str`
Salva um tensor (gradiente ou ativação) no disco, opcionalmente quantizado.

### `verify_tensor_integrity(name: str) -> bool`
Verifica se a integridade do tensor no disco confere com o hash esperado.

## Banco de Dados & Vocabulário (database_manager.py)
### `__annotate__(format, /)`
Sem descrição.

### `build_hot_token_cache(size: int = 5000) -> None`
Carrega os tokens mais usados do banco para a RAM (Cache Hot-Tokens).

Args:
    size: Quantidade de tokens frequentes a carregar.

### `bulk_insert_vocab(words: List[str]) -> None`
Insere uma lista de palavras únicas no banco de forma eficiente.

Args:
    words: Lista de strings a serem inseridas no vocabulário.

### `create_index_on_text() -> None`
Cria índice na coluna 'text' da tabela vocab para acelerar buscas.

### `get_db_connection() -> Generator[sqlite3.Connection, NoneType, NoneType]`
Context manager para conexão segura com o SQLite.

Yields:
    sqlite3.Connection: Conexão ativa com o banco de dados especificado em DB_PATH.

### `get_or_create_id(text: str) -> int`
Retorna o ID de um token, buscando no cache ou criando no banco se necessário.

Args:
    text: Texto do token.
    
Returns:
    int: ID único do token.

### `get_text_by_id(token_id: int) -> str`
Retorna o texto original correspondente a um ID, usando cache se possível.

Args:
    token_id: ID do token.
    
Returns:
    str: Texto do token ou '<UNK>' se não encontrado.

### `init_db() -> None`
Inicializa as tabelas do banco de dados conforme as especificações do ZeroRAM-GEN.
Cria as tabelas de vocabulário, logs de treino, telemetria, histórico, biased templates,
sessão, dados sintéticos, estatísticas de vocabulário, sharding e rede.

### `log_telemetry(metric_name: str, value: float, context: str | None = None) -> None`
Grava o log de telemetria para análise de performance de I/O e RAM.

Args:
    metric_name: Nome da métrica (ex: 'io_read_latency').
    value: Valor numérico da métrica.
    context: Contexto adicional opcional.

### `log_training_metrics(epoch: int, step: int, loss: float) -> None`
Grava métricas de treinamento (Loss por step) no banco de dados.

Args:
    epoch: Época atual.
    step: Passo global de treino.
    loss: Valor do erro (Loss).

### `update_vocab_usage(token_id: int) -> None`
Incrementa o contador de uso de um token para otimização de cache.

Args:
    token_id: ID do token a ser incrementado.

