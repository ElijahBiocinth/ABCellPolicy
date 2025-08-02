DEFAULT_DB_PATH: str      = '~/test.sqlite'
DEFAULT_SRC_FOLDER: str   = '~/Scene_test'
DEFAULT_MODEL_PATH: str   = '~/weights_yolo.pt'

# Core 
MIN_AREA: float                   = 100.0
BASE_IOU_MATCH: float             = 0.45
SOFT_IOU_MIN: float               = 0.15
SOFT_IOU_MIN_RELAX: float         = 0.05
EARLY_FORCE_IOU_MIN: float        = 0.0
DIST_GATE_FACTOR: float           = 3.0
RELAX_DIST_MULT: float            = 1.4
EARLY_GATE_MULT: float            = 1.8
MAX_MISSED: int                   = 5
SPLIT_IOU_THRESH: float           = 0.5
MERGE_IOU_THRESH: float           = 0.55
DUP_IOU_SUPPRESS: float           = 0.90
SIMPLIFY_TOL: float               = 0.8

# Cost weighting
APPEARANCE_WEIGHT: float          = 0.20
IOU_WEIGHT: float                 = 0.45
DIST_WEIGHT: float                = 0.25
SHAPE_WEIGHT: float               = 0.10
MAX_COST_CAP: float               = 1.5

# Adaptive matching
ADAPTIVE_ALPHA: float             = 0.2
ADAPTIVE_COST_MULT: float         = 1.5
ADAPTIVE_COST_OFFSET: float       = 0.05
MIN_MATCH_RATIO: float            = 0.50
K_INIT: int                       = 3
EARLY_ZERO_MATCH_LIMIT: int       = 2
CENTROID_FALLBACK_MAX_DIST_MULT: float = 4.0

# New-track control
LIMIT_NEW_TRACK_GROWTH: bool      = True
NEW_TRACK_SURPLUS_FACTOR: float   = 1.25
RECLAIM_UNUSED_AFTER: int         = 2
CONTINUITY_TARGET: float          = 0.80

# Greedy Recovery passes
MAX_RECOVERY_PASSES: int          = 2
RECOVERY_GATE_MULT: float         = 2.2
RECOVERY_FEATURE_THRESH: float    = 0.65
RECOVERY_SHAPE_THRESH: float      = 0.55
RECOVERY_MAX_DIST_NORM: float     = 1.00
GREEDY_MAX_DIST_MULT: float       = 2.5
GREEDY_FEATURE_THRESH: float      = 0.70
GREEDY_SHAPE_THRESH: float        = 0.60
FORCE_DELAY_NEW_TRACKS_UNTIL_RECOVERY: bool = True

# Re-identification 
REID_MAX: int                     = 50
REID_MAX_AGE_GAP: int             = 25
REID_POS_DIST: float              = 150.0
REID_SCORE_THRESH: float          = 0.6
REID_MIN_AREA_RATIO: float        = 0.25
REID_MAX_AREA_RATIO: float        = 4.0
CONFIRM_SPLIT_FRAMES: int         = 2

# Tiling / overlap
TILE_SIZE: int                    = 1536
OVERLAP: int                      = 512

# Feature extraction flags
USE_INTENSITY: bool               = True
USE_HU: bool                      = True
USE_MOMENTS: bool                 = True

# Debug line
DEBUG_FIRST_N_FRAMES: int         = 5
PRINT_COST_STATS: bool            = True
PRINT_REID_MATCHES: bool          = True
PRINT_ASSIGN_DBG: bool            = True

DEFAULT_BACKEND: str              = "yolo"

try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper