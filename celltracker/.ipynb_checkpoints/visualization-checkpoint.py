import cv2
import numpy as np
from .color_utils import get_track_color

def draw_tracks(image: np.ndarray, tracks, thickness=2, draw_id=True) -> np.ndarray:
    for tr in tracks:
        color = tr.display_color if hasattr(tr, 'display_color') else get_track_color(tr.id)
        pts = np.array([[int(x), int(y)] for x, y in tr.poly.exterior.coords], dtype=np.int32)
        cv2.polylines(image, [pts], True, color, thickness)
        
        if draw_id:
            cx, cy = tr.history[-1]
            label = f"{tr.id}"
            if tr.generation > 1:
                label += f"({tr.generation})"
            cv2.putText(
                image,
                label,
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )
            
    return image
