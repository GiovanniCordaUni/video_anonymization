import cv2
from matplotlib.pyplot import box
import numpy as np

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

def apply_oval_blur(frame, box, ksize=(51, 51)):
  """
  box nel formato [x1, y1, x2, y2]
  """
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

  # ROI sfocata
  blurred_roi = cv2.GaussianBlur(roi, ksize, 0)

  # Maschera ellittica
  mask = np.zeros((h, w), dtype=np.uint8)
  center = (w // 2, h // 2)
  axes = (w // 2, h // 2)  # puoi squishare un po' se vuoi forma meno "piena"

  cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

  # Maschera a 3 canali
  mask_3ch = cv2.merge([mask, mask, mask])

  # Combina roi originale + blur solo dove mask == 255
  result = np.where(mask_3ch == 255, blurred_roi, roi)

  # Rimetti nel frame
  frame[y1:y2, x1:x2] = result

  return frame