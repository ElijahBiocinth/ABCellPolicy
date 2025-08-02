import math
import time

_start_time = None
_smooth_frame_time = None
_prev_progress_len = 0

def format_eta(eta_sec: float) -> str:
    if eta_sec <= 0 or math.isinf(eta_sec) or math.isnan(eta_sec):
        return "--"
    if eta_sec >= 3600:
        h = int(eta_sec // 3600)
        m = int((eta_sec % 3600) // 60)
        return f"{h}h {m}m"
    if eta_sec >= 120:
        return f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
    if eta_sec >= 60:
        return f"1m {int(eta_sec % 60)}s"
    return f"{int(eta_sec)}s"

def start_timing(total_frames: int):
    global _start_time, _smooth_frame_time
    _start_time = time.time()
    _smooth_frame_time = None

def one_line_progress(current, total, extra=""):
    import sys
    global _start_time, _smooth_frame_time, _prev_progress_len

    if total <= 0:
        total = 1
    if _start_time is None:
        _start_time = time.time()

    pct = min(max(current / total, 0.0), 1.0)
    elapsed = time.time() - _start_time

    if current > 0:
        frame_time = elapsed / current
        alpha = 0.2
        if _smooth_frame_time is None:
            _smooth_frame_time = frame_time
        else:
            _smooth_frame_time = (1 - alpha) * _smooth_frame_time + alpha * frame_time
        eta_sec = _smooth_frame_time * (total - current)
    else:
        eta_sec = 0.0

    eta_str = format_eta(eta_sec)
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "#" * filled + "-" * (bar_len - filled)

    msg = (f"[{current:>4}/{total:<4}] |{bar}| {pct*100:5.1f}% "
           f"ETA:{eta_str:>7} Elap:{int(elapsed)}s {extra}")
    clear_tail = " " * max(0, _prev_progress_len - len(msg))

    sys.stdout.write("\r" + msg + clear_tail)
    sys.stdout.flush()
    _prev_progress_len = len(msg)

    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()
