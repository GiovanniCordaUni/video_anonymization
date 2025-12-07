from dataclasses import dataclass

@dataclass
class Detection:
    """
    Rappresenta una rilevazione di oggetto.
    Attributi:
        x1 (int): Coordinata x del vertice superiore sinistro della bounding box.
        y1 (int): Coordinata y del vertice superiore sinistro della bounding box.
        x2 (int): Coordinata x del vertice inferiore destro della bounding box.
        y2 (int): Coordinata y del vertice inferiore destro della bounding box.
        confidence (float): Confidenza della rilevazione.
        class_id (int): ID della classe rilevata.

    """
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int