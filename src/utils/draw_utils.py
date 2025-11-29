import cv2
from typing import List
from src.detectors.yolov8_detector import Detection

def draw_boxes(frame, detections: List[Detection], color=(0, 255, 0), thickness=2):
    """
    Disegna le bounding box sull'immagine.
    frame: immagine BGR
    detections: lista di Detection
    color: colore del rettangolo (BGR)
    thickness: spessore della linea
    return: immagine con le box disegnate
    """
    output = frame.copy()

    for det in detections:
        cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), color, thickness)

        label = f"{det.class_id} ({det.confidence:.2f})"
        cv2.putText(output, label, (det.x1, det.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output