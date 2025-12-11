from typing import List, Optional, Mapping, Any
import numpy as np

from face_detection import RetinaFace  # libreria RetinaFace
from src.detectors.detection_types import Detection


class RetinaFaceDetector:
    """
    Wrapper per il rilevatore RetinaFace.
    Supporta inizializzazione manuale oppure tramite config.yaml.

    Attributi:
        conf_threshold (float): soglia di confidenza.
        device (str): "cpu" o "cuda".
        class_id (int): ID di classe da usare nei Detection (es. 0 = 'face').
    """

    def __init__(
        self,
        conf_threshold: float = 0.8,
        device: str = "cpu",
        class_id: int = 0,
        resize: float = 1.0,
        max_size: int = 1080,
    ):
        # Configurazione dispositivo di calcolo (0 per GPU 0, -1 per CPU)
        gpu_id = 0 if device.lower().startswith("cuda") else -1

        self.device = device
        self.conf_threshold = conf_threshold
        self.class_id = class_id
        self.resize = resize
        self.max_size = max_size

        # modello RetinaFace
        self.model = RetinaFace(gpu_id=gpu_id)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "RetinaFaceDetector":
        """
        Crea un'istanza del detector leggendo i parametri dal dizionario
        derivante dalla sezione detector.retinaface del config.yaml.
        """
        conf_threshold = float(cfg.get("conf_threshold", 0.8))
        device = cfg.get("device", "cpu")
        class_id = int(cfg.get("class_id", 0))
        resize = float(cfg.get("resize", 1.0))
        max_size = int(cfg.get("max_size", 1080))

        return cls(
            conf_threshold=conf_threshold,
            device=device,
            class_id=class_id,
            resize=resize,
            max_size=max_size,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Esegue il rilevamento su un frame (BGR, np.ndarray HxWx3)
        e ritorna una lista di Detection del tipo: [x1, y1, x2, y2, confidence, class_id]
        """

        # La libreria accetta immagini HxWx3, 0–255.
        faces = self.model(
            frame,
            threshold=self.conf_threshold,
            resize=self.resize,
            max_size=self.max_size,
        )
        # faces è una lista di (box, landmarks, score)
        detections: List[Detection] = []

        for box, _landmarks, score in faces:
            score = float(score)
            if score < self.conf_threshold:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]

            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=score,
                    class_id=self.class_id,  # un’unica classe: volto
                )
            )

        return detections

    def detect_boxes(self, frame: np.ndarray):
        """
        Utilità: esegue il rilevamento e ritorna solo le bounding box
        """
        detections = self.detect(frame)
        return [(d.x1, d.y1, d.x2, d.y2) for d in detections]
