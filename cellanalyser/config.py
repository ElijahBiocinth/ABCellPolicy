DB_PATHS = {}
OUT_DIR = ''
PX_TO_UM = 0.519933
FRAME_DT = 350.0
N_THREADS = 20
SMOOTH_WINDOW = 15
ALPHA_TEST = 0.005
PERFORM_STATS = True

METRICS_LIST = ['area', 'eccentricity', 'orientation', 'circularity']
DYN_TRACK_COLS = [
    'MSD','Directional Persistence','Meandering Index','Mean Turning Angle (deg)',
    'Radius of Gyration','Arrest Coefficient','Shape-Motion Coupling','Relative Motion Change',
    'Velocity Cross-Correlation','Directionality Ratio (deg)',
    'Mean Speed (µm/frame)','Mean Acceleration (µm/frame²)'
]

TURN_ANGLE_THRESH_DEG = 45.0
ARREST_SPEED_THRESH = 0.2
