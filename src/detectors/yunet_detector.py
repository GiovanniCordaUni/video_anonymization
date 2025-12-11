import cv2
import numpy as np
from typing import Any, List, Mapping, Tuple
from src.detectors.detection_types import Detection


class YuNetDetector:
    """
    Wrapper per il face detector YuNet (OpenCV Zoo) che restituisce Detection.
    Supporta inizializzazione manuale oppure tramite config.yaml.
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (320, 320), #640x640 più accuratezza e meno falsi negativi; utile per video ad alta risoluzione
        score_threshold: float = 0.9, #soglia di confidenza minima più alta per ridurre i falsi positivi
        nms_threshold: float = 0.3, #soglia NMS più bassa per eliminare meglio le sovrapposizioni
        top_k: int = 5000, #numero massimo di detection da considerare
        face_class_id: int = 0, #class_id da assegnare alle facce (es. 0)
    ):
        """
        Args:
            model_path: percorso al file .onnx di YuNet.
            input_size: dimensione di input iniziale (w, h).
            score_threshold: soglia di confidenza minima.
            nms_threshold: soglia per NMS (Non-Maximum Suppression).
            top_k: numero massimo di detection.
            face_class_id: class_id da assegnare alle facce (es. 0).
        """
        self.input_size = input_size
        self.face_class_id = face_class_id

        self.detector = cv2.FaceDetectorYN_create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "YuNetDetector":
        """
        Crea un'istanza del detector leggendo i parametri dal dizionario
        derivante dalla sezione detector.yunet del config.yaml.
        """

        model_path = cfg["model_path"]
        input_size = tuple(cfg.get("input_size", (320, 320)))

        return cls(
            model_path=model_path,
            input_size=input_size,
            score_threshold=float(cfg.get("score_threshold", 0.9)),
            nms_threshold=float(cfg.get("nms_threshold", 0.3)),
            top_k=int(cfg.get("top_k", 5000)),
            face_class_id=int(cfg.get("face_class_id", 0)),
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Esegue la face detection su un frame BGR.

        Returns:
            List[Detection]: lista di bounding box + confidenza per le facce.
        """
        h, w = frame.shape[:2]

        # Aggiorna la dimensione di input in base al frame corrente
        if (w, h) != self.input_size:
            self.input_size = (w, h)
            self.detector.setInputSize(self.input_size)

        # detect ritorna (retval, faces)
        # retval: bool che indica se ha trovato facce, NOTA: non usato su per yunet (sempre 0)
        # faces: N x 15 -> [x, y, w, h, score, l0x, l0y, ..., l4x, l4y]
        # l0..l4 sono i landmark degli occhi, naso, bocca
        retval, faces = self.detector.detect(frame)

        detections: List[Detection] = []

        if faces is None:
            return detections

        for face in faces:
            x, y, width, height, score = face[:5]

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + width)
            y2 = int(y + height)

            # Clamp ai bordi dell'immagine per evitare box invalidi
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            det = Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(score),
                class_id=self.face_class_id,
            )
            detections.append(det)

        return detections
