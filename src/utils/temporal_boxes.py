
from src.detectors.detection_types import Detection

def apply_box_hysteresis(
    current_detections,
    last_valid_detections,
    gap_frames,
    *, # Tutti i parametri dopo * possono essere passati SOLO come keyword arguments per prevenire errori
    max_gap_frames=2,
    padding_ratio=0.10,
    padding_px=0,
    frame_size=None,
    min_area_px=0,
):
    """
    Applica una strategia di isteresi temporale alle bounding box di rilevazione
    per compensare brevi mancate rilevazioni consecutive (false negative).

    Questa funzione è pensata per pipeline di anonimizzazione video basate su
    face detection, dove il detector (es. YOLOv8) può perdere il volto per uno
    o pochi frame consecutivi. In assenza di nuove rilevazioni, la funzione può
    riutilizzare ("ereditare") le ultime bounding box valide per un numero
    limitato di frame, mantenendo attiva l'anonimizzazione.

    Parametri:

    current_detections : list[Detection]
        Lista delle rilevazioni ottenute nel frame corrente (frame t).
        Ogni rilevazione è rappresentata da un oggetto `Detection`.
        Se la lista è vuota, la funzione può ricorrere all'ereditarietà
        delle bounding box precedenti.
    last_valid_detections : list[Detection] or None
        Lista delle ultime rilevazioni valide disponibili,
        utilizzate come fallback in caso di mancata rilevazione nel frame
        corrente. Se None o vuota, l'ereditarietà non è possibile.
    gap_frames : int
        Numero di frame consecutivi senza rilevazioni osservati prima del
        frame corrente. Questo valore viene aggiornato e restituito.
    max_gap_frames : int, default=2
        Numero massimo di frame consecutivi per cui è consentito riutilizzare
        le bounding box precedenti. Superata questa soglia, nessuna bounding
        box viene restituita.
    padding_ratio : float, default=0.10
        Padding relativo applicato alle bounding box ereditate (ad esempio
        0.10 corrisponde a un'espansione del 10% in larghezza e altezza).
        Il padding viene applicato solo in caso di ereditarietà.
    padding_px : int, default=0
        Padding assoluto in pixel applicato alle bounding box ereditate,
        in aggiunta al padding relativo. Utile per video ad alta risoluzione.
    frame_size : tuple[int, int] or None, default=None
        Dimensioni del frame come (larghezza, altezza). Se specificato,
        le bounding box risultanti vengono limitate ai bordi dell'immagine.
    min_area_px : int, default=0
        Area minima in pixel consentita per una bounding box. Le box con
        area inferiore vengono scartate, per evitare la propagazione di
        rilevazioni instabili.

    Ritorna:

    final_detections : list[Detection]

        Lista delle rilevazioni da utilizzare nel frame corrente per
        l'anonimizzazione. Se il detector ha prodotto risultati, coincide
        con current_detections. In caso di ereditarietà, contiene le
        bounding box derivate da last_valid_detections. In caso contrario,
        è una lista vuota.
    
    updated_last_valid_detections : list[Detection] or None
    
        Ultime rilevazioni valide aggiornate da propagare al frame successivo.
        Se il detector ha prodotto rilevazioni, queste diventano le nuove
        last_valid_detections. In caso di ereditarietà, tipicamente le
        ultime rilevazioni valide rimangono invariate.
    
    updated_gap_frames : int
    
        Contatore aggiornato dei frame consecutivi senza rilevazioni dopo
        l'elaborazione del frame corrente. Viene azzerato se il detector
        produce rilevazioni, altrimenti incrementato.
    
    used_inheritance : bool
    
        Indica se nel frame corrente è stata applicata l'ereditarietà delle
        bounding box (True) oppure se sono state utilizzate rilevazioni
        dirette del detector (False).

    Note:

    - Questa strategia è pensata per colmare gap temporali molto brevi
      (1-5 frame) e non implementa meccanismi di tracking né associazione
      di identità tra più volti.
    - In presenza di occlusioni prolungate o frequenti mancate rilevazioni,
      si potrebbe implementare l'approccio con interpolazione temporale
      o tracker dedicati.
    """
    final_detections = []
    updated_last_valid_detections = last_valid_detections
    updated_gap_frames = gap_frames
    used_inheritance = False

    if current_detections:
        # Rilevazioni presenti nel frame corrente
        final_detections = current_detections
        updated_last_valid_detections = current_detections
        updated_gap_frames = 0
    else:
        # Nessuna rilevazione nel frame corrente
        # Incremento la streak di frame mancanti
        updated_gap_frames = gap_frames + 1

        # Eredito se sono ancora dentro la soglia
        if last_valid_detections and updated_gap_frames <= max_gap_frames:
            used_inheritance = True

            for det in last_valid_detections:
                box_width = det.x2 - det.x1
                box_height = det.y2 - det.y1
                pad_w = int(box_width * padding_ratio) + padding_px
                pad_h = int(box_height * padding_ratio) + padding_px

                new_x1 = det.x1 - pad_w
                new_y1 = det.y1 - pad_h
                new_x2 = det.x2 + pad_w
                new_y2 = det.y2 + pad_h

                if frame_size:
                    frame_w, frame_h = frame_size
                    new_x1 = max(0, new_x1)
                    new_y1 = max(0, new_y1)
                    new_x2 = min(frame_w - 1, new_x2)
                    new_y2 = min(frame_h - 1, new_y2)

                # controllo box valida
                if new_x2 <= new_x1 or new_y2 <= new_y1:
                    continue

                area = (new_x2 - new_x1) * (new_y2 - new_y1)
                if area >= min_area_px:
                    final_detections.append(
                        Detection(
                            x1=new_x1, y1=new_y1, x2=new_x2, y2=new_y2,
                            confidence=det.confidence,
                            class_id=det.class_id,
                        )
                    )
        else:
            final_detections = []


    return (
        final_detections,
        updated_last_valid_detections,
        updated_gap_frames,
        used_inheritance,
    )
