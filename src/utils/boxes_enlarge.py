
"""
    utility per ingrandire le bounding box
    scale: fattore di ingrandimento (default 1.3)

    NOTA: potrebbe essere necessario per aumentare il livello di anonimizzazione
    se la sfocatura non copre tutta la faccia (orecchie, mento, capelli, ecc.)
"""

def enlarge_box(box, scale=1.3):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    # centro
    cx = x1 + w / 2
    cy = y1 + h / 2

    # nuovo box scalato
    new_w = w * scale
    new_h = h * scale

    new_x1 = int(cx - new_w / 2)
    new_y1 = int(cy - new_h / 2)
    new_x2 = int(cx + new_w / 2)
    new_y2 = int(cy + new_h / 2)

    return [new_x1, new_y1, new_x2, new_y2]
