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
    - Il nome del file di output viene derivato dal file di input (es. "4sst-Andrea1.mp4")
      e convertito nella forma "NNN_Excod_sessionX.mp4",
      dove:
        - NNN  = ID numerico progressivo del soggetto,
        - Excod = codice univoco dell'esercizio (es. "4sst"),
        - X = numero della sessione registrata per lo stesso tipo di esercizio.

    - Se il soggetto NNN non è mai stato processato:
        viene creata la cartella:
            soggetto_NNN/NNN_ex_sessionM/
        dove il video anonimizzato verrà salvato.

    - Se il soggetto esiste già:
        il video viene salvato nella cartella soggetto_NNN/.

Formati supportati:
    - Video: .mp4, .avi, .mov, .mkv
    - L'audio può essere preservato nel video di output (opzione attivabile).

Gestione errori:
    - Il modulo ignora file non validi o formati non supportati.
    - I video non leggibili vengono saltati e registrati nei log.

Note:
    - La pipeline è progettata per essere estensibile: nuovi detector, nuove tecniche
      di anonimizzazione e nuovi formati di output possono essere aggiunti senza
      modificare la logica principale.
"""

