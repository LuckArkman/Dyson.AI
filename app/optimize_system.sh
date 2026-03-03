#!/bin/bash
# Tuning de sistema para ZeroRAM-GEN em ambiente Linux
echo "ZeroRAM-GEN: Otimizando sistema para I/O de disco e SQLite..."

# 1. Habilitar modo 'writeback' e 'noatime' para os volumes de pesos (se montados separadamente)
# Exemplo para o volume de dados do Docker
# mount -o remount,noatime,nodiratime /var/lib/docker/volumes

# 2. Configurações de Kernel para I/O Intensivo
sysctl -w vm.dirty_ratio=10
sysctl -w vm.dirty_background_ratio=5
sysctl -w vm.vfs_cache_pressure=50

# 3. Tuning de SQLite no Nível de SO
# Sugestão: Usar disco em RAM (tmpfs) para arquivos temporários se houver RAM disponível
rm -rf /app/weights/temp
mkdir -p /app/weights/temp
# mount -t tmpfs -o size=1G tmpfs /app/weights/temp

echo "[OK] Otimizações de sistema aplicadas para produção."
