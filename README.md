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
```

# Diagramma di flusso della pipeline

```mermaid

flowchart TD
A[main.py<br/>CLI: --config --input --output]

A --> B{args.input è<br/>una cartella?}

B -->|Sì| C[run_dataset_from_config]
B -->|No| D[run_single_as_dataset_from_config]

C --> E[load_config]
D --> E[load_config]

E --> F[build_detector_from_config<br/>YuNet / RetinaFace / YOLOv8 / YOLOv12]
F --> G[build_anonymizer_from_config<br/>blur / pixelate / black mask]

G --> H[build_anon_output_path_dataset<br/>soggettoNNN/NNN_ex_session_anon.ext]

H --> I[process_single_video]

I --> J[open_video + create_video_writer]

J --> K{Loop frame}

K --> L[Detector<br/>ogni frame_stride]
L --> M[Bounding boxes]

M --> N[Anonimizzazione box<br/>enlarge + ellipse opzionale]

N --> O[write frame]

O --> K

K -->|Fine video| P[Chiusura risorse input/output<br/>release VideoCapture and VideoWriter]

P --> Q[Video anonimizzato salvato]

```
