from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass

# semplice classe per rappresentare una rilevazione
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int

# classe principale per il rilevatore YoloV8
class YoloV8Detector:

    # inizializza il rilevatore con i parametri specificati
    def __init__(self, model_path: str, conf_threshold: float = 0.5, device: str = "cpu", classes: Optional[List[int]] = None):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.classes = classes
        self.model = None

    # carica il modello se non è già stato caricato
    def _ensure_model_loaded(self):
        # TODO: da implementare il caricamento del modello
        raise NotImplementedError()

    # esegue il rilevamento su un frame dato
    def detect(self, frame: np.ndarray) -> List[Detection]:
        # TODO: da implementare il rilevamento
        raise NotImplementedError()

    # estrae solo le coordinate delle bounding box dalle rilevazioni
    def detect_boxes(self, frame: np.ndarray):
        detections = self.detect(frame)
        return [(d.x1, d.y1, d.x2, d.y2) for d in detections]
