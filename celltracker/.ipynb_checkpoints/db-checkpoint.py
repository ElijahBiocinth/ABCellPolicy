import sqlite3
from pathlib import Path
from typing import List, Tuple, Any

def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image TEXT,
            time_frame INTEGER,
            frame_width INTEGER,
            frame_height INTEGER,
            track_id INTEGER,
            generation INTEGER,
            polygon_points TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tracking_metrics (
            frame INTEGER PRIMARY KEY,
            num_tracks INTEGER,
            matches INTEGER,
            avg_match_iou REAL,
            splits INTEGER,
            merges INTEGER,
            id_switches INTEGER,
            median_cost REAL,
            relaxed_used INTEGER,
            continuity REAL,
            recovery_passes INTEGER
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lineage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame INTEGER,
            parent_track_id INTEGER,
            child_track_id INTEGER,
            parent_generation INTEGER,
            child_generation INTEGER,
            mode TEXT
        );
    """)
    conn.commit()
    return conn

def insert_tracks(conn: sqlite3.Connection, rows: List[Tuple[Any, ...]]) -> None:
    conn.executemany("""
        INSERT INTO tracks
            (image, time_frame, frame_width, frame_height, track_id, generation, polygon_points)
        VALUES (?,?,?,?,?,?,?)
    """, rows)
    conn.commit()

def insert_metrics(conn: sqlite3.Connection, metrics: Tuple[Any, ...]) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO tracking_metrics
            (frame, num_tracks, matches, avg_match_iou,
             splits, merges, id_switches, median_cost,
             relaxed_used, continuity, recovery_passes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, metrics)
    conn.commit()

def insert_lineage(conn: sqlite3.Connection, rows: List[Tuple[Any, ...]]) -> None:
    conn.executemany("""
        INSERT INTO lineage
            (frame, parent_track_id, child_track_id,
             parent_generation, child_generation, mode)
        VALUES (?,?,?,?,?,?)
    """, rows)
    conn.commit()

def close_db(conn: sqlite3.Connection) -> None:
    conn.commit()
    conn.close()
