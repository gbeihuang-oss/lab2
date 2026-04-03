import sqlite3
import json
from pathlib import Path
from datetime import datetime
from utils.config import DB_PATH, DATA_DIR

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Initialize all database tables."""
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS records (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            module    TEXT NOT NULL,
            name      TEXT NOT NULL,
            file_path TEXT,
            metadata  TEXT,
            created_at TEXT DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS chat_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS workflows (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            steps       TEXT,
            formula_ref INTEGER,
            created_at  TEXT DEFAULT (datetime('now','localtime'))
        );
    """)
    conn.commit()
    conn.close()

def save_record(module: str, name: str, file_path: str = None, metadata: dict = None) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO records (module, name, file_path, metadata) VALUES (?, ?, ?, ?)",
        (module, name, str(file_path) if file_path else None,
         json.dumps(metadata, ensure_ascii=False) if metadata else None)
    )
    conn.commit()
    record_id = cur.lastrowid
    conn.close()
    return record_id

def get_records(module: str = None):
    conn = get_connection()
    cur = conn.cursor()
    if module:
        cur.execute("SELECT * FROM records WHERE module=? ORDER BY created_at DESC", (module,))
    else:
        cur.execute("SELECT * FROM records ORDER BY created_at DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def delete_record(record_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM records WHERE id=?", (record_id,))
    row = cur.fetchone()
    if row and row["file_path"]:
        p = Path(row["file_path"])
        if p.exists():
            p.unlink()
    cur.execute("DELETE FROM records WHERE id=?", (record_id,))
    conn.commit()
    conn.close()

def save_chat_message(role: str, content: str):
    conn = get_connection()
    conn.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

def get_chat_history(limit: int = 50):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM chat_history ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return list(reversed(rows))

def clear_chat_history():
    conn = get_connection()
    conn.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()

def save_workflow(name: str, steps: list, formula_ref: int = None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO workflows (name, steps, formula_ref) VALUES (?, ?, ?)",
        (name, json.dumps(steps, ensure_ascii=False), formula_ref)
    )
    conn.commit()
    wid = cur.lastrowid
    conn.close()
    return wid

def get_workflows():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM workflows ORDER BY created_at DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
