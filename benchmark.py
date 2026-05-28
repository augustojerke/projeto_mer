import time
import psutil
import torch
import threading
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict


@dataclass
class EpocaMetrics:
    epoca:           int
    train_loss:      float
    val_loss:        float
    rmse_arousal:    float
    rmse_valence:    float
    mae_arousal:     float
    mae_valence:     float
    pearson_arousal: float
    pearson_valence: float
    ccc_arousal:     float
    ccc_valence:     float
    tempo_seg:       float
    gpu_mem_mb:      float


@dataclass
class ExperimentoMetrics:
    nome:                    str
    modo:                    str
    device:                  str
    total_params:            int
    epocas:                  List[EpocaMetrics] = field(default_factory=list)
    # métricas de regressão — por segmento
    test_rmse_arousal:       float = 0.0
    test_rmse_valence:       float = 0.0
    test_mae_arousal:        float = 0.0
    test_mae_valence:        float = 0.0
    test_pearson_arousal:    float = 0.0
    test_pearson_valence:    float = 0.0
    test_ccc_arousal:        float = 0.0
    test_ccc_valence:        float = 0.0
    test_r2_arousal:         float = 0.0
    test_r2_valence:         float = 0.0
    # RMSE agrupado por música (padrão PMEmo)
    test_rmse_songs_arousal: float = 0.0
    test_rmse_songs_valence: float = 0.0
    # classificação emocional por quadrante
    quadrant_accuracy:       float = 0.0
    quadrant_f1_macro:       float = 0.0
    quadrant_report:         Dict  = field(default_factory=dict)
    # CV summary
    cv_metricas:             Dict  = field(default_factory=dict)
    # recursos
    tempo_total_seg:         float = 0.0
    cpu_percent_medio:       float = 0.0
    gpu_mem_pico_mb:         float = 0.0


class BenchmarkMonitor:
    def __init__(self):
        self._cpu_samples = []
        self._gpu_samples = []
        self._monitorando = False
        self._thread      = None
        self.tem_gpu      = torch.cuda.is_available()

    def iniciar(self):
        self._cpu_samples = []
        self._gpu_samples = []
        self._monitorando = True
        self._thread      = threading.Thread(target=self._coletar, daemon=True)
        self._thread.start()

    def parar(self):
        self._monitorando = False
        if self._thread:
            self._thread.join()

    def _coletar(self):
        while self._monitorando:
            self._cpu_samples.append(psutil.cpu_percent(interval=0.5))
            if self.tem_gpu:
                self._gpu_samples.append(torch.cuda.memory_allocated() / 1024 ** 2)

    def cpu_medio(self):
        return sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else 0.0

    def gpu_pico(self):
        return max(self._gpu_samples) if self._gpu_samples else 0.0

    def gpu_atual_mb(self):
        return torch.cuda.memory_allocated() / 1024 ** 2 if self.tem_gpu else 0.0


def salvar_resultados(resultados: dict, path="resultados.json"):
    serializavel = {nome: asdict(exp) for nome, exp in resultados.items()}
    with open(path, "w") as f:
        json.dump(serializavel, f, indent=2)
    print(f"Resultados salvos em {path}")
