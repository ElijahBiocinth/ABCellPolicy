import argparse
from pathlib import Path
from celltracker.config import DEFAULT_DB_PATH, DEBUG_FIRST_N_FRAMES, DEFAULT_BACKEND

from celltracker.pipeline import run_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Run cell tracking pipeline")
    parser.add_argument(
        "--db", type=Path,
        default=Path(DEFAULT_DB_PATH).expanduser(),
        help="Path to SQLite database (default: config.DEFAULT_DB_PATH)"
    )
    parser.add_argument(
        "--src", type=Path, required=True,
        help="Path to source image folder"
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to YOLO model weights file"
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output folder for annotated images"
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="Disable saving visualization images"
    )
    parser.add_argument(
        "--first-n", type=int, default=DEBUG_FIRST_N_FRAMES,
        help="Number of initial frames to debug (default: config.DEBUG_FIRST_N_FRAMES)"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Torch device index or 'cpu'"
    )
    parser.add_argument(
        "--jit", action="store_true",
        help="Enable numba JIT if installed"
    )
    parser.add_argument(
        "--backend", type=str, choices=["yolo","cellpose"],
        default=DEFAULT_BACKEND,
        help="Which detector to use: 'yolo' or 'cellpose'"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
