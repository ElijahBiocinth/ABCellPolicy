import json
import argparse
from cellanalyser.pipeline import run_pipeline
from cellanalyser import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_paths', type=json.loads)
    parser.add_argument('--out_dir')
    parser.add_argument('--px_to_um', type=float)
    parser.add_argument('--frame_dt', type=float)
    parser.add_argument('--smooth_window', type=int)
    parser.add_argument('--alpha_test', type=float)
    parser.add_argument('--perform_stats', type=json.loads)
    parser.add_argument('--n_threads', type=int)
    args = parser.parse_args()

    config.DB_PATHS.clear()
    config.DB_PATHS.update(args.db_paths)
    config.OUT_DIR        = args.out_dir
    config.PX_TO_UM       = args.px_to_um
    config.FRAME_DT       = args.frame_dt
    config.SMOOTH_WINDOW  = args.smooth_window
    config.ALPHA_TEST     = args.alpha_test
    config.PERFORM_STATS  = args.perform_stats
    config.N_THREADS      = args.n_threads

    run_pipeline()

if __name__=='__main__':
    main()
