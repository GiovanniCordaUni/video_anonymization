"""
Video processing pipeline module.

Questo modulo definisce la pipeline per l'elaborazione dei video, includendo:
    - lettura dei frame,
    - applicazione del rilevatore di volti,
    - anonimizzazione delle regioni rilevate,
    - scrittura del video di output.

Funzionalità principali:
    - Il modulo può elaborare un singolo video o un'intera cartella di video.
    - Per ogni video:
        - legge i frame in sequenza,
        - applica il rilevatore di volti scelto,
        - anonimizza ciascun volto rilevato tramite una delle tecniche disponibili.
    
    - Rilevatori supportati:
        - YuNet
        - RetinaFace
        - yoloV8-face
        - yoloV12-face

    - Tecniche di anonimizzazione disponibili:
        - blur
        - pixelate
        - back mask (maschera opaca)
        - GAN-based (DeepPrivacy2) [DA IMPLEMENTARE]

Output e struttura delle cartelle:
    - Per i casi "dataset" (input = cartella), il nome del file di output viene
      derivato dal file di input (es. "4sst-Andrea1.mp4") e convertito nella forma
      "NNN_Excod_sessionX.ext", dove:
        - NNN  = ID numerico progressivo del soggetto,
        - Excod = codice univoco dell'esercizio (es. "4sst"),
        - X = numero della sessione registrata per lo stesso tipo di esercizio.
      I file sono organizzati in sottocartelle:
            output_root/soggettoNNN/NNN_Excod_sessionX.ext

Formati supportati:
    - Video: .mp4, .avi, .mov, .mkv
    - L'audio può essere preservato nel video di output (opzione attivabile a livello di codec).

Gestione errori:
    - Il modulo ignora file non validi o formati non supportati.
    - I video non leggibili vengono saltati e registrati nei log.

Note:
    - La pipeline è progettata per essere estensibile: nuovi detector, nuove tecniche
      di anonimizzazione e nuovi formati di output possono essere aggiunti senza
      modificare la logica principale.

-----------------------------------------------------------------------
Implementazione:

Usa:
  - config.yaml per:
      * scelta detector
      * scelta anonymizer
      * parametri base (frame_stride, ecc.)
  - utility già esistenti:
      * open_video, create_video_writer (io_utils)
      * enlarge_box (boxes_enlarge)
      * parsing dei nomi da dataset_organizer (_parse_name)
"""

import os
from pathlib import Path
from typing import List, Tuple, Callable, Dict, Any, Optional

import cv2
import torch
import yaml
import numpy as np

# Detector
from src.detectors.yolo.yolov8_detector import YoloV8Detector
from src.detectors.yolo.yolov12_detector import YoloV12Detector
from src.detectors.yunet_detector import YuNetDetector
from src.detectors.retinaFace_detector import RetinaFaceDetector

# Anonymizer
from src.anonymizers.gaussian_blur import apply_gaussian_blur, apply_oval_blur
from src.anonymizers.pixelation import apply_pixelation, apply_oval_pixelation
from src.anonymizers.black_mask import apply_black_mask, apply_oval_black_mask

# Utils già esistenti
from src.utils.io_utils import open_video, create_video_writer  # 
from src.utils.boxes_enlarge import enlarge_box                # 
from src.utils.dataset_organizer import _parse_name            # 
# from src.utils.draw_utils import draw_boxes   # opzionale, per debug

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


#carica file di configurazione yaml

def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Carica il file di configurazione YAML.

    Di default cerca "config/config.yaml" rispetto alla working directory.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# costruisce detector da config

def build_detector_from_config(cfg: Dict[str, Any]):
    """
    Inizializza il detector leggendo la sezione `detector` del config.yaml.
    Si appoggia ai metodi .from_config delle classi wrapper.
    """
    # gestione device (cpu/cuda) automatica se impostato a "auto"
    det_cfg = cfg.get("detector", {})
    active = str(det_cfg.get("active", "")).lower()

    if not active:
        raise ValueError("[ERROR] Nel config manca detector.active")

    if active not in det_cfg:
        raise ValueError(f"[ERROR] Nel config manca la sezione detector.{active}")

    det_params = det_cfg[active]
    device = str(det_params.get("device", "cpu")).lower()

    # Fallback CUDA -> CPU
    if device in {"cuda", "cuda:0"}:
        try:
            if not torch.cuda.is_available():
                print("[WARN] CUDA richiesta ma non disponibile. Uso CPU.")
                det_params = dict(det_params)  # copia
                det_params["device"] = "cpu"
        except Exception:
            print("[WARN] Torch/CUDA non disponibile. Uso CPU.")
            det_params = dict(det_params)
            det_params["device"] = "cpu"

    if active == "yolov8":
        return YoloV8Detector.from_config(det_cfg["yolov8"])

    if active == "yolov12":
        return YoloV12Detector.from_config(det_cfg["yolov12"])

    if active == "yunet":
        return YuNetDetector.from_config(det_cfg["yunet"])

    if active == "retinaface":
        return RetinaFaceDetector.from_config(det_cfg["retinaface"])

    raise ValueError(f"Detector attivo non supportato: {active}")


# costruisce anonymizer da config

def build_anonymizer_from_config(
    cfg: Dict[str, Any],
) -> Callable[[np.ndarray, Tuple[int, int, int, int]], np.ndarray]:
    """
    Ritorna una funzione (frame, box) -> frame che applica
    l'anonimizzazione scelta in config.yaml.

    Esempio di sezione config:

    anonymizer:
      type: "gaussian_blur"   # gaussian_blur | pixelate | black
      face_scale: 1.3         # fattore per enlarge_box
      use_ellipse: true
      blur_strength: 0.9
      pixel_size: 20
    """
    a_cfg = cfg["anonymizer"]
    a_type = a_cfg["type"].lower()
    face_scale = float(a_cfg.get("face_scale", 1.0))
    use_ellipse = bool(a_cfg.get("use_ellipse", False))

    # parametri specifici
    blur_strength = float(a_cfg.get("blur_strength", 0.9))
    pixel_size = int(a_cfg.get("pixel_size", 20))

    def anonymizer_fn(
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> np.ndarray:
        # ingrandisci la box se richiesto
        if face_scale != 1.0:
            box_enlarged = enlarge_box(box, scale=face_scale)
        else:
            box_enlarged = box

        if a_type == "gaussian_blur":
            if use_ellipse:
                # kernel approssimato a partire da blur_strength
                k = int(max(31, 101 * blur_strength))
                if k % 2 == 0:
                    k += 1
                return apply_oval_blur(frame, box_enlarged, ksize=(k, k))
            else:
                return apply_gaussian_blur(frame, box_enlarged, strength=blur_strength)

        elif a_type == "pixelate":
            if use_ellipse:
                return apply_oval_pixelation(frame, box_enlarged, pixel_size=pixel_size)
            else:
                return apply_pixelation(frame, box_enlarged, pixel_size=pixel_size)

        elif a_type == "black":
            if use_ellipse:
                return apply_oval_black_mask(frame, box_enlarged)
            else:
                return apply_black_mask(frame, box_enlarged)

        else:
            raise ValueError(f"Tipo di anonymizer non supportato: {a_type}")

    return anonymizer_fn


# elabora singolo video

def process_single_video(
    input_video_path: str,
    output_video_path: str,
    detector,
    anonymizer_fn: Callable[[np.ndarray, Tuple[int, int, int, int]], np.ndarray],
    cfg: Dict[str, Any],
) -> None:
    """
    Elabora un singolo video:
      - legge i frame con open_video
      - applica il detector
      - anonimizza le box
      - scrive il risultato con create_video_writer
    """
    cap, fps, (width, height), total_frames = open_video(input_video_path)
    print(f"[INFO] Input: {input_video_path}")
    print(f"[INFO] FPS: {fps}, Size: {width}x{height}, Frames: {total_frames}")

    writer = create_video_writer(output_video_path, fps, (width, height))
    print(f"[INFO] Output: {output_video_path}")

    # parametri video opzionali dal config
    v_cfg = cfg.get("video", {})
    frame_stride = int(v_cfg.get("frame_stride", 1)) # processa ogni N frame

    frame_idx = 0
    last_boxes: Optional[List[Tuple[int, int, int, int]]] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        run_detection = (frame_idx % frame_stride == 0)

        if run_detection:
            # usa detect_boxes se presente, altrimenti converte da Detection
            if hasattr(detector, "detect_boxes"):
                boxes = detector.detect_boxes(frame)
            else:
                detections = detector.detect(frame)
                boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]

            last_boxes = boxes
        else:
            boxes = last_boxes or []

        # applica l'anonimizzazione su tutte le box
        for box in boxes:
            frame = anonymizer_fn(frame, box)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[INFO] Completato: {output_video_path}")


# modalità singolo video: input file -> output file

def run_single_from_config(
    config_path: str,
    input_video_path: str,
    output_video_path: str,
) -> None:
    """
    Modalità semplice:
        L'output viene passato esplicitamente come path.
    """
    cfg = load_config(config_path)
    detector = build_detector_from_config(cfg)
    anonymizer_fn = build_anonymizer_from_config(cfg)

    process_single_video(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        detector=detector,
        anonymizer_fn=anonymizer_fn,
        cfg=cfg,
    )



# costruisce path output (soggettoNNN/NNN_exercise_session.ext) per dataset
def build_anon_output_path_dataset(
    input_video_path: str,
    output_root: str,
    subject_to_id: Dict[str, int],
    next_id: int,
):
    """
    Costruisce il path di output con struttura:
        output_root/soggettoNNN/NNN_exercise_session.ext

    Usando la stessa logica di dataset_organizer._parse_name, la funzione
    si limita a restituire il path target per il video anonimizzato
    e aggiorna la mappa soggetto->ID.

    Esempio:
        4sst-Andrea1.mp4 -> soggetto001/001_4sst_1.mp4
    """
    src = Path(input_video_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # parsing del nome, es: 4sst-Andrea1.mp4 -> ("4sst", "Andrea", "1")
    exercise, subject, session = _parse_name(src)

    # assegna ID soggetto
    if subject not in subject_to_id:
        subject_to_id[subject] = next_id
        next_id += 1

    sid = subject_to_id[subject]
    sid_str = f"{sid:03d}"

    subject_dir = output_root / f"soggetto{sid_str}"
    subject_dir.mkdir(parents=True, exist_ok=True)

    dst_name = f"{sid_str}_{exercise}_{session}{src.suffix}"
    dst = subject_dir / dst_name

    return str(dst), subject_to_id, next_id


# modalità dataset: input cartella -> output cartella strutturata
def run_dataset_from_config(
    config_path: str,
    input_dir: str,
    output_root: str,
) -> None:
    """
    Modalità "dataset":

    1) Prende una cartella di input (es. DATASET) con video tipo:
           4sst-Andrea1.mp4, 4sst-Luca1.mp4, ...
    2) Per ogni video, calcola il path destinazione nello stile:
           output_root/soggettoNNN/NNN_4sst_1.mp4
    3) Esegue l'anonimizzazione e salva direttamente il video anonimizzato
       nel path calcolato.

    """
    cfg = load_config(config_path)
    detector = build_detector_from_config(cfg)
    anonymizer_fn = build_anonymizer_from_config(cfg)

    input_dir_path = Path(input_dir)
    if not input_dir_path.exists() or not input_dir_path.is_dir():
        raise NotADirectoryError(f"Cartella di input non valida: {input_dir}")

    # mappa soggetto -> ID numerico
    subject_to_id: Dict[str, int] = {}
    next_id = 1

    # filtra solo i file video con estensioni supportate
    video_paths = sorted(
        p for p in input_dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXT
    )

    if not video_paths:
        print(f"[WARN] Nessun video valido trovato in: {input_dir}")
        return

    print(f"[INFO] Trovati {len(video_paths)} video in {input_dir}")
    print(f"[INFO] Output root: {output_root}")

    for src in video_paths:
        out_path, subject_to_id, next_id = build_anon_output_path_dataset(
            input_video_path=str(src),
            output_root=output_root,
            subject_to_id=subject_to_id,
            next_id=next_id,
        )

        print(f"[INFO] {src.name} -> {out_path}")
        process_single_video(
            input_video_path=str(src),
            output_video_path=out_path,
            detector=detector,
            anonymizer_fn=anonymizer_fn,
            cfg=cfg,
        )

    print("\n[INFO] Mappa soggetti:")
    for name, sid in sorted(subject_to_id.items(), key=lambda x: x[1]):
        print(f"  soggetto{sid:03d} <- {name}")

def run_single_as_dataset_from_config(
    config_path: str,
    input_video_path: str,
    output_root: str,
) -> None:
    """
    Variante di comodo: prende UN singolo video di input e lo tratta come un
    mini-dataset di un solo file.

    Output:
        output_root/soggettoNNN/NNN_exercise_session.ext
    """
    cfg = load_config(config_path)
    detector = build_detector_from_config(cfg)
    anonymizer_fn = build_anonymizer_from_config(cfg)

    # mappa iniziale vuota: un solo soggetto possibile
    subject_to_id: Dict[str, int] = {}
    next_id = 1

    out_path, subject_to_id, next_id = build_anon_output_path_dataset(
        input_video_path=input_video_path,
        output_root=output_root,
        subject_to_id=subject_to_id,
        next_id=next_id,
    )

    print(f"[INFO] (single-as-dataset) {input_video_path} -> {out_path}")
    process_single_video(
        input_video_path=input_video_path,
        output_video_path=out_path,
        detector=detector,
        anonymizer_fn=anonymizer_fn,
        cfg=cfg,
    )

    # opzionale: stampa mappa soggetto -> ID
    print("\n[INFO] Mappa soggetti (single file):")
    for name, sid in sorted(subject_to_id.items(), key=lambda x: x[1]):
        print(f"  soggetto{sid:03d} <- {name}")