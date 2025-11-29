import cv2
from matplotlib.pyplot import box

def apply_gaussian_blur(img, box, strength=0.9):
    """
    Applica un filtro gaussiano a una regione specificata dell'immagine (ROI).
    Il kernel sarà adattivo in base alla dimensione della ROI e alla forza specificata.
    
    img: immagine di input (array numpy).
    box: tuple/list con le coordinate della ROI (x1, y1, x2, y2).
    strength: intensità del filtro gaussiano (valore tra 0 e 1).

    Ritorna l'immagine con la ROI sfocata applicata sull'immagine originale.
    """
    # Estrai le coordinate della ROI
    # il detector fornisce float, ma OpenCV richiede int
    x1, y1, x2, y2 = map(int, box)

    # limito i valori delle coordinate per evitare che il box esca dai bordi dell'immagine
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    # Estrai la regione di interesse (ROI) dall'immagine
    roi = img[y1:y2, x1:x2]
    
    rh, rw = roi.shape[:2]
    # kernel proporzionale alla dimensione più grande della faccia
    k = int(max(rh, rw) * strength)

    # forza il kernel ad essere dispari e con un minimo sufficiente per anonimizzare
    if k % 2 == 0:
        k += 1
    k = max(k, 31)

    # Applica il filtro gaussiano alla ROI
    blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)
    
    # Sostituisci la ROI originale con quella sfocata nell'immagine
    img[y1:y2, x1:x2] = blurred_roi
    
    return img