from typing import Any, List, Optional, Mapping
import numpy as np
from ultralytics import YOLO
from src.detectors.detection_types import Detection 

# classe principale per il rilevatore YoloV8
class YoloV8Detector:

    """
        Wrapper per il rilevatore YoloV8.
        Attributi:
            model_path (str): Percorso al file del modello YoloV8.
            conf_threshold (float): Soglia di confidenza per filtrare le rilevazioni.
            device (str): Dispositivo su cui eseguire il modello ("cpu" o "cuda").
            classes (Optional[List[int]]): Lista opzionale di ID di classi da rilevare.

        Può essere inizializzato in due modi:

            1) In modo esplicito, passando i parametri:
                detector = YoloV8Detector(
                    model_path="models/yolov8/yolov8n-face.pt",
                    conf_threshold=0.5,
                    device="cpu",
                    classes=[0],
                )

            2) Dal file di configurazione (sezione detector.yolov8):
                cfg = load_config("config.yaml")
                yolov8_cfg = cfg["detector"]["yolov8"]
                detector = YoloV8Detector.from_config(yolov8_cfg)
    """

    # inizializza il rilevatore
    def __init__(self, model_path: str, conf_threshold: float = 0.5, device: str = "cpu", classes: Optional[List[int]] = None):

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.classes = classes
        self.model: Optional[YOLO] = None


    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "YoloV8Detector":
        """
        Inizializza il detector leggendo i parametri da un dict di configurazione,
        tipicamente derivato dalla sezione `detector.yolov8` del config.yaml.

        Esempio di cfg atteso:

        detector:
        yolov8:
            model_path: "models/yolov8/yolov8n-face.pt"
            conf_threshold: 0.5
            device: "cpu"
            # classes: [0]  # opzionale

        """
        model_path = cfg["model_path"]
        conf_threshold = float(cfg.get("conf_threshold", 0.5))
        device = cfg.get("device", "cpu")
        classes = cfg.get("classes")  # opzionale, può essere None

        return cls(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device,
            classes=classes,
        )
        

    # carica il modello, se non è già stato caricato
    def _ensure_model_loaded(self):
        if self.model is None:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)

    """
    # esegue il rilevamento su un frame dato
    # NOTA sintassi: def name_fun(param1: Type1, param2: Type2) -> ReturnType:
        # preannuncia che la funzione ritorna una lista di oggetti Detection;
    """

    def detect(self, frame: np.ndarray) -> List[Detection]:
        self._ensure_model_loaded()
        if self.model is None:
            raise RuntimeError("Modello non caricato correttamente.")
        
        # results è una lista che contiene le rilevazioni delle quali ci interessa la prima (e unica) immagine
        results = self.model(frame, device=self.device)[0]
        detections: List[Detection] = [] # lista vuota per memorizzare le rilevazioni

        for box in results.boxes:
            # estrai le coordinate della bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist() # xyxy è un tensor 1x4
            conf = float(box.conf[0]) 
            class_id = int(box.cls[0])

            # filtra in base alla soglia di confidenza e alle classi specificate
            if conf < self.conf_threshold:
                continue
            # continua se esistono classi specificate e la classe non è tra quelle specificate (esempio per rilevare solo persone)
            if self.classes is not None and class_id not in self.classes:
                continue

            detections.append(
                Detection(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    confidence=conf,
                    class_id=class_id
                )
            )

        return detections

    # estrae solo le coordinate delle bounding box dalle rilevazioni da passare all'anonymizer
    def detect_boxes(self, frame: np.ndarray):

        detections = self.detect(frame)
        return [(d.x1, d.y1, d.x2, d.y2) for d in detections]