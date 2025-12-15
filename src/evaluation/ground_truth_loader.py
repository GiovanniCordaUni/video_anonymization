from typing import Dict, List, Tuple
from pathlib import Path

Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


def video_name_to_gt_dir(video_path: str, gt_root: str) -> Path:
    """
    Dato il path del video (nominato in formato NNN_codex_session), costruisce il nome della cartella GT corrispondente.

    Esempio:
        video: NNN_codex_session.mp4  (es. 001_4sst_1.mp4)
        -> stem: "NNN_codex_session"  (es. "001_4sst_1")
        -> gt_dir_name: "gt_NNN_codex_session" (es. "gt_001_4sst_1")
    """

    video_stem = Path(video_path).stem
    gt_dir = Path(gt_root) / f"gt_{video_stem}"

    if not gt_dir.exists():
        raise FileNotFoundError(
            f"Cartella GT non trovata per il video '{video_stem}': {gt_dir}"
        )

    return gt_dir


def yolo_to_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_width: int,
    img_height: int,
) -> Box:
    """
    Converte una box YOLO normalizzata (cx, cy, w, h in [0,1])
    in coordinate assolute (x1, y1, x2, y2) in pixel (degli angoli superiore sinistro e inferiore destro).
    """
    cx_abs = cx * img_width
    cy_abs = cy * img_height
    w_abs = w * img_width
    h_abs = h * img_height

    x1 = cx_abs - w_abs / 2.0
    y1 = cy_abs - h_abs / 2.0
    x2 = cx_abs + w_abs / 2.0
    y2 = cy_abs + h_abs / 2.0

    # clamp ai bordi immagine
    x1 = max(0.0, min(x1, img_width - 1))
    y1 = max(0.0, min(y1, img_height - 1))
    x2 = max(0.0, min(x2, img_width - 1))
    y2 = max(0.0, min(y2, img_height - 1))

    # evita box degenerate
    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def load_video_ground_truth(
    video_path: str,
    gt_root: str,
    frame_size: Tuple[int, int],
) -> Dict[int, List[Box]]:
    """
    Carica la ground truth per un singolo video, usando una struttura YOLO:

        gt_root/
            gt_<video_name>/
                labels/
                train/
                    frame_000000.txt
                    frame_000001.txt
                    frame_000002.txt
                    ...

    Dove ciascun file .txt contiene righe nel formato YOLO:

        class_id cx cy w h

    con cx,cy,w,h normalizzati in [0,1].

    Args:
        video_path: path al file video (es. .../001_4sst_1.mp4)
        gt_root: cartella root della GT (es. data/ground_truth)
        frame_size: (width, height) del video originale

    Returns:
        dict: frame_idx -> lista di box (x1, y1, x2, y2) in pixel
    """
    img_width, img_height = frame_size
    gt_dir = video_name_to_gt_dir(video_path, gt_root)

    labels_dir = gt_dir / "labels" / "train"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Cartella labels/train non trovata in {gt_dir}")

    frame_to_boxes: Dict[int, List[Box]] = {}

    # scansiona tutti i file frame_XXXXXX.txt
    for txt_file in sorted(labels_dir.glob("frame_*.txt")):
        # es. frame_000123.txt -> 123
        stem = txt_file.stem  # "frame_000123"
        try:
            frame_idx = int(stem.split("_")[-1])
        except ValueError:
            # nome non conforme, salta
            continue

        boxes: List[Box] = []

        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                # sono accettate anche righe con campi extra: class cx cy w h [extra...]
                if len(parts) < 5:
                    continue

                # formato YOLO: class cx cy w h
                try:
                    """
                    class_id inizializzato ma non usato in quanto nella valutazione
                    corrente è previsto l'uso di una singola classe "face".
                    """
                    _class_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                except ValueError:
                    continue

                box = yolo_to_xyxy(cx, cy, bw, bh, img_width, img_height)
                if box is not None:
                    boxes.append(box)

        if boxes:
            frame_to_boxes[frame_idx] = boxes

    return frame_to_boxes