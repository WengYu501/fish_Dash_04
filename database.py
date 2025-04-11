import sqlite3
import pandas as pd
from pathlib import Path

DB_FILE = Path("liquidity_cache.db")
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

cur.execute('''
CREATE TABLE IF NOT EXISTS stock_data (
    ticker TEXT,
    date TEXT,
    adj_close REAL,
    volume INTEGER,
    return REAL,
    amihud REAL,
    z_score REAL,
    if_anomaly INTEGER,
    PRIMARY KEY (ticker, date)
)
''')
conn.commit()

def write_to_db(ticker, df):
    df = df.copy()
    df['ticker'] = ticker
    df = df[['ticker', 'date', 'adj_close', 'volume', 'return', 'amihud', 'z_score', 'if_anomaly']]
    df.to_sql('stock_data', conn, if_exists='append', index=False)

def load_from_db(ticker):
    query = "SELECT * FROM stock_data WHERE ticker = ? ORDER BY date"
    df = pd.read_sql_query(query, conn, params=(ticker,))
    if df.empty:
        return None
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def has_data(ticker):
    result = cur.execute("SELECT COUNT(*) FROM stock_data WHERE ticker = ?", (ticker,)).fetchone()
    return result[0] > 0

def delete_data(ticker):
    cur.execute("DELETE FROM stock_data WHERE ticker = ?", (ticker,))
    conn.commit()

def close():
    conn.close()
