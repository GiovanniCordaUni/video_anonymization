import cv2
import numpy as np

def apply_pixelation(frame, box, pixel_size=10):
    """
    Applica un effetto di pixelazione alla regione di interesse (ROI) specificata dalla box nell'immagine.

    Args:
        frame: L'immagine originale.
        box: Una lista o tupla con le coordinate della box nel formato [x1, y1, x2, y2].
        pixel_size: La dimensione dei pixel per l'effetto di pixelazione. Nota: più grande è il valore, più forte è l'effetto di pixelazione.
    """

    x1, y1, x2, y2 = map(int, box) #cast a int per sicurezza
    face = frame[y1:y2, x1:x2]

    # abbassa la risoluzione della ROI
    h, w = face.shape[:2]
    tmp = cv2.resize(face, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)

    # riporta la ROI alla risoluzione originale
    pixelated_face = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_NEAREST)

    # sostituisci la ROI originale con quella pixelata nell'immagine
    frame[y1:y2, x1:x2] = pixelated_face

    return frame

def apply_oval_pixelation(frame, box, pixel_size=10):
    
    x1, y1, x2, y2 = map(int, box)

    # Clipping ai bordi del frame
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_frame, x2)
    y2 = min(h_frame, y2)

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return frame  # niente da fare

    # ROI originale
    roi = frame[y1:y2, x1:x2]

    # ROI pixelata
    h, w = roi.shape[:2]
    tmp = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
    pixelated_roi = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_NEAREST)

    # Crea una maschera ovale
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (w // 2, h // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Maschera a 3 canali
    mask_3ch = cv2.merge([mask, mask, mask])

    # Combina ROI originale e pixelata usando la maschera
    oval_pixelated_roi = np.where(mask_3ch == 255, pixelated_roi, roi)

    # Sostituisci la ROI originale con quella pixelata nell'immagine
    frame[y1:y2, x1:x2] = oval_pixelated_roi

    return frame