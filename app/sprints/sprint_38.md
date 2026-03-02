# Sprint 38: Controle de Erros em Tensores Sincronizados

## Objetivos
- Implementar o sistema de Verificação de Integridade (Hash Check) para arquivos de pesos em Swarm.
- Desenvolver a rotina de recuperação de pesos corrompidos.

## Ferramentas & Pacotes
- **Hashlib (sha256)**: Para verificação de arquivos no disco.
- **Python (shutil)**: Para backup e restore de pesos corrompidos.

## Funções e Implementações
- `calculate_weight_hash(path)`: Criar a digital do arquivo de pesos atual.
- `verify_remote_hash(node_id, local_hash)`: Comparar a digital local com a remota no Swarm.
- `restore_weight_from_peer(peer_node_id, target_layer)`: Corrigir pesos locais através de um parceiro de rede.

## Detalhes Técnicos
Crucial para o ZeroRAM-GEN devido à frequência de escrita/leitura. A integridade física do disco torna-se um fator de erro.
