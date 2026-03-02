import time
import json
import os

class MetricsService:
    def __init__(self, log_dir: str = "Dayson/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_file = os.path.join(log_dir, f"metrics_{int(time.time())}.jsonl")
        print(f"[Metrics] Gravando telemetria em: {self.metrics_file}")

    def record_metrics(self, epoch: int, batch: int, loss: float, memory_mb: float):
        record = {
            "timestamp": time.time(),
            "epoch": epoch,
            "batch": batch,
            "loss": float(loss),
            "ram_usage_mb": float(memory_mb)
        }
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_summary(self):
        # Poderia ler o arquivo e retornar estatísticas
        return {"status": "recording"}
