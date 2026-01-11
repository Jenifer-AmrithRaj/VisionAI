"""
File: utils/auth_utils.py
Purpose: Simple authentication management for VisionAI
Auto-creates a default admin user (username: admin, password: visionai123)
so you can log in immediately without needing manual registration.
"""

import sqlite3
import hashlib
import os

DB_PATH = "user_auth.db"


def init_user_db():
    """Initialize user table if not exists and ensure default admin exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT
        )
    """)
    conn.commit()

    # --- ensure default admin user exists ---
    default_user = "admin"
    default_pass = "visionai123"
    hashed = hashlib.sha256(default_pass.encode()).hexdigest()
    cursor.execute("SELECT username FROM users WHERE username=?", (default_user,))
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (default_user, hashed)
        )
        print(f"✅ Default admin user created: username='admin' password='visionai123'")
        conn.commit()
    conn.close()


def hash_password(password):
    """Securely hash password."""
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password):
    """Register a new user (you can call this manually if needed)."""
    init_user_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)",
        (username, hash_password(password)),
    )
    conn.commit()
    conn.close()


def validate_user(username, password):
    """Validate credentials during login."""
    init_user_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return hash_password(password) == row[0]
    return False
