import cv2
import os

def open_video(input_path):
    """
    Apre un file video utilizzando OpenCV.

    input_path: percorso del file video.

    Ritorna l'oggetto VideoCapture di OpenCV e le proprietà del video:
    - fps: fotogrammi al secondo
    - dimensioni: (larghezza, altezza)
    - total_frames: numero totale di fotogrammi
    
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Il file video non esiste: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise IOError(f"Impossibile aprire il file video: {input_path}")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return cap, fps, (width, height), total_frames