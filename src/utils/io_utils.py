import cv2
import os
from pathlib import Path

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

def create_video_writer(output_path, fps, frame_size):
    """
    Crea un oggetto VideoWriter di OpenCV per salvare un file video.

    output_path: percorso del file video di output.
    fps: fotogrammi al secondo.
    frame_size: dimensioni del fotogramma (larghezza, altezza).

    Ritorna l'oggetto VideoWriter di OpenCV.
    """
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec per file mp4 (tra i più comuni)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec per file avi (garante maggiore compatibilità)
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # crea la cartella se non esiste
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    if not out.isOpened():
        raise RuntimeError(f"Impossibile creare il file video: {output_path}")
    
    return out


def extract_frames_from_videos(input_dir: str, output_dir: str, valid_ext=None):
    """
    Estrae i frame da tutti i video presenti in input_dir.
    Ogni video avrà una propria cartella dentro output_dir.

    Args:
        input_dir (str): cartella che contiene i video
        output_dir (str): cartella di destinazione dei frame
        valid_ext (set, optional): estensioni video accettate
    """
    if valid_ext is None:
        valid_ext = {".mp4", ".avi", ".mov", ".mkv"}

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in input_dir.iterdir():
        if video_path.suffix.lower() not in valid_ext:
            continue  # ignora file non video

        video_name = video_path.stem  # es: video1
        video_output_folder = output_dir / video_name
        video_output_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n Estrazione frame da: {video_name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f" Impossibile aprire il video: {video_path}")
            continue

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = video_output_folder / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_filename), frame)

            frame_idx += 1

        cap.release()
        print(f" Estratti {frame_idx} frame in {video_output_folder}")

    print("\n Operazione completata.")
