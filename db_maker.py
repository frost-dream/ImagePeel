from sqlite3 import connect
sql = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password_hash TEXT NOT NULL,
        is_verified TEXT,
        token TEXT,
        code TEXT,
        expire TEXT,
        created_at TEXT
    );
    """
db = connect('app.db')
db.executescript(sql)
db.commit()