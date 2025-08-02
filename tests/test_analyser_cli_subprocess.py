import sqlite3, os, subprocess
from pathlib import Path

def make_dummy_db(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE tracks (
            track_id INTEGER,
            frame INTEGER,
            contour BLOB
        )
    """)
    c.execute("INSERT INTO tracks VALUES (1, 0, NULL)")
    conn.commit(); conn.close()

def test_cli_creates_outputs(tmp_path):
    db = tmp_path / "test.sqlite"
    make_dummy_db(str(db))
    out = tmp_path / "out"
    out.mkdir()

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "analyser_cli.py"

    cmd = [
        "python", str(script),
        str(db),
        "--static", "--dynamic", "--stats", "--plot",
        "--out-dir", str(out)
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr

    files = {f.name for f in out.iterdir()}
    expected = {
        f"static_metrics_{db.stem}.csv",
        f"dynamic_metrics_{db.stem}.csv",
        f"statistics_{db.stem}.csv"
    }
    assert expected.issubset(files)
    assert any(f.suffix == ".png" for f in out.iterdir())
