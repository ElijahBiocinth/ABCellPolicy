import sqlite3
import pandas as pd

def read_tracks(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tracks);")}
    sel = ['time_frame', 'polygon_points']
    sel += ['track_id'] if 'track_id' in cols else ['rowid as track_id']
    
    if 'well' in cols:
        sel.append('well')
    df = pd.read_sql_query(f"SELECT {','.join(sel)} FROM tracks", conn)
    conn.close()
    
    return df
