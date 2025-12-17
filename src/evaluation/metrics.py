from typing import List, Tuple, Dict
import numpy as np

Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2)

def compute_iou(box1: Box, box2: Box) -> float:

    """
    Calcola l'Intersection over Union (IoU) tra due box di delimitazione.
    Box nel formato (x1, y1, x2, y2).
    """

    # Calcola le coordinate dell'area di intersezione scegliendo i massimi e minimi appropriati
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1) # Larghezza dell'area di intersezione
    inter_height = max(0, y2 - y1) # Altezza dell'area di intersezione
    inter_area = inter_width * inter_height

    # controllo su aree non nulle
    if inter_area == 0:
        return 0.0
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])  # Area del box ottenuta moltiplicando larghezza per altezza [x2 - x1] * [y2 - y1]
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter_area
    if union == 0:
        return 0.0
    
    iou = inter_area / union
    return iou

def evaluate_frame_detections_center_based(
    gt_boxes: List[Box],
    pred_boxes: List[Box],
):
    """
    Valutazione center-based:
    - TP: il centro della predizione cade dentro una GT
    - FP: predizioni che non cadono in nessuna GT
    - FN: GT non matchate da nessuna predizione
    """

    gt_used = set()
    tp = 0
    fp = 0

    for pred in pred_boxes:
        # centro della predizione
        cx = (pred[0] + pred[2]) / 2.0
        cy = (pred[1] + pred[3]) / 2.0

        matched = False

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in gt_used:
                continue

            x1, y1, x2, y2 = gt
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                tp += 1
                gt_used.add(gt_idx)
                matched = True
                break

        if not matched:
            fp += 1

    fn = len(gt_boxes) - len(gt_used)

    return tp, fp, fn


def evaluate_frame_detections(
    gt_boxes: List[Box],
    pred_boxes: List[Box],
    iou_threshold: float = 0.5,
    allow_multiple_predictions: bool = False
):
    """
    Confronta le predizioni di un singolo frame con le box di ground truth.

    Ritorna:
        tp, fp, fn  (interi)

    Strategia:
      - se allow_multiple_predictions = False:
          matching one-to-one greedy: ogni GT può essere matchata da al massimo
          una predizione (comportamento "standard").
      - se allow_multiple_predictions = True:
          ogni pred con IoU >= soglia su QUALSIASI GT viene conteggiata come TP,
          le GT non matchate restano FN.
    """

    gt_used = set() # Indici delle box di ground truth già utilizzate, per il matching one-to-one

    tp = 0
    fp = 0

    for pred in pred_boxes:
        best_iou = 0.0 
        best_gt_idx = -1 # Indice della box di ground truth con la migliore IoU inizializzato a -1 (nessuna box trovata)

        for gt_idx, gt in enumerate(gt_boxes):
            if (not allow_multiple_predictions) and (gt_idx in gt_used):
                continue  # Salta le box di ground truth già utilizzate nel matching one-to-one

            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
    
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            if not allow_multiple_predictions:
                gt_used.add(best_gt_idx)  # Segna la box di ground truth come utilizzata
        else:
            fp += 1

    if allow_multiple_predictions:
        # se sono permessi più pred per GT, le FN sono le GT che non hanno almeno una pred con IoU >= soglia
        fn = 0

        for gt in gt_boxes:
            matched = any(
                compute_iou(gt, pred) >= iou_threshold for pred in pred_boxes
            )
            if not matched:
                fn += 1
    else:
        fn = len(gt_boxes) - len(gt_used)

    return tp, fp, fn

def precision_recall_f1(tp: int, fp: int, fn: int):
    """
    Calcola precision, recall e F1-score dati true positives, false positives e false negatives.
    """

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1

def aggregate_metrics(stats_per_frame: List[Dict[str, int]]):
    """
    Aggrega i TP/FP/FN di tutti i frame e ritorna:
        - tp_tot, fp_tot, fn_tot
        - precision, recall, f1 globali
    """

    tp_tot = sum(frame_stats['tp'] for frame_stats in stats_per_frame)
    fp_tot = sum(frame_stats['fp'] for frame_stats in stats_per_frame)
    fn_tot = sum(frame_stats['fn'] for frame_stats in stats_per_frame)

    precision, recall, f1 = precision_recall_f1(tp_tot, fp_tot, fn_tot)

    return {
        'tp': tp_tot,
        'fp': fp_tot,
        'fn': fn_tot,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }