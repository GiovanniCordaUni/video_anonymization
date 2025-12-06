import cv2

def apply_black_mask(frame, box, color=(0,0,0)):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)
    return frame

def apply_oval_black_mask(frame, box, color=(0,0,0)):
    x1, y1, x2, y2 = box
    center = ((x1+x2)//2, (y1+y2)//2)
    axes = ((x2-x1)//2, (y2-y1)//2)
    cv2.ellipse(frame, center, axes, 0, 0, 360, color, -1)
    return frame
