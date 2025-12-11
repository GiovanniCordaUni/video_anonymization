# Video Anonymization Pipeline

Pipeline modulare per l'anonimizzazione di video tramite rilevatori di volto (YuNet, RetinaFace, YOLOv8-face, YOLOv12-face) e diverse tecniche di anonimizzazione (blur, pixelate, black mask), configurabile tramite file YAML.

La struttura del progetto è estensibile, i componenti sono separati per funzionalità (detector, anonymizer, utilità I/O, organizzazione dataset) ed è presente un unico `main.py` come entry point.

---

## Struttura del progetto

Schema semplificato delle cartelle principali:

```text
project/
│  main.py
│  README.md
│
├─ config/
│   └─ config.yaml          # configurazione detector/anonymizer/pipeline
│
├─ data/
│   ├─ input/
│   │   └─ videos/
│   │       └─ DATASET/     # video grezzi di input
│   └─ output/
│       └─ videos/          # output anonimizzati (soggettoNNN/...)
│
├─ models/
│   ├─ yunet/
│   │   └─ face_detection_yunet_2023mar.onnx
│   ├─ yolov8/
│   │   └─ yolov8n-face.pt
│   └─ yolov12/
│       └─ yolov12n-face.pt
│
└─ src/
    ├─ detectors/
    │   ├─ yolo/
    │   │   ├─ yolov8_detector.py
    │   │   └─ yolov12_detector.py
    │   ├─ yunet_detector.py
    │   └─ retinaFace_detector.py
    │
    ├─ anonymizers/
    │   ├─ gaussian_blur.py
    │   ├─ pixelation.py
    │   └─ black_mask.py
    │
    ├─ pipeline/
    │   └─ video_pipeline.py
    │
    └─ utils/
        ├─ io_utils.py
        ├─ boxes_enlarge.py
        ├─ dataset_organizer.py
        └─ draw_utils.py        # opzionale, per debug/visualizzazione
