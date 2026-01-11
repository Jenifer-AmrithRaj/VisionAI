"""
File: utils/helper.py
Purpose: Manage patient screening history for VisionAI
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "patient_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_history (
            uid TEXT PRIMARY KEY,
            full_name TEXT,
            age REAL,
            gender TEXT,
            stage TEXT,
            confidence REAL,
            risk_score REAL,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_patient_record(uid, meta, pred):
    """Save a new screening record."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO patient_history (uid, full_name, age, gender, stage, confidence, risk_score, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        uid,
        meta.get("Full_Name", "Unknown"),
        meta.get("Age", 0),
        meta.get("Gender", "N/A"),
        pred.get("predicted_stage", "UNKNOWN"),
        pred.get("confidence", 0),
        pred.get("risk_score", 0),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def fetch_all_history():
    """Retrieve all past screening records."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_history ORDER BY date DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def fetch_record(uid):
    """Retrieve a specific patient's record."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_history WHERE uid=?", (uid,))
    row = cursor.fetchone()
    conn.close()
    return row
