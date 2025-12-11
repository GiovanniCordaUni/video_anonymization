from typing import List, Optional, Mapping, Any
import numpy as np

from .yolov8_detector import YoloV8Detector
from ..detection_types import Detection


class YoloV12Detector(YoloV8Detector):
    """
        Wrapper per il rilevatore YoloV12.

        Estende YoloV8Detector riutilizzandone la logica interna.
        Cambia solo il “nome” della classe per chiarezza a livello di codice
        e per poter gestire configurazioni distinte (es. path del modello).

        Attributi:
            model_path (str): Percorso al file del modello YoloV12.
            conf_threshold (float): Soglia di confidenza per filtrare le rilevazioni.
            device (str): Dispositivo su cui eseguire il modello ("cpu" o "cuda").
            classes (Optional[List[int]]): Lista opzionale di ID di classi da rilevare.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
    ):
        # richiama il costruttore del wrapper YoloV8, riutilizzando la logica
        super().__init__(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device,
            classes=classes,
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "YoloV12Detector":
        """
        Inizializza YoloV12Detector da un dizionario di configurazione
        (tipicamente cfg["detector"]["yolov12"]).

        Esempio:
            yolov12_cfg = config["detector"]["yolov12"]
            detector = YoloV12Detector.from_config(yolov12_cfg)
        """
        model_path = cfg["model_path"]
        conf_threshold = float(cfg.get("conf_threshold", 0.5))
        device = cfg.get("device", "cpu")
        classes = cfg.get("classes")  # opzionale

        return cls(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device,
            classes=classes,
        )