from typing import Dict, Tuple

TRACK_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {}

def get_track_color(track_id: int) -> Tuple[int, int, int]:
    if track_id not in TRACK_COLOR_MAP:
        h = (track_id * 2654435761) & 0xFFFFFFFF
        r = 64 + ((h >> 0)  & 0xFF) // 2
        g = 64 + ((h >> 8)  & 0xFF) // 2
        b = 64 + ((h >> 16) & 0xFF) // 2
        TRACK_COLOR_MAP[track_id] = (int(b), int(g), int(r))
        
    return TRACK_COLOR_MAP[track_id]
