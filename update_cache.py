import database
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

def update_ticker(ticker):
    print(f"Updating {ticker}...")
    df = yf.download(ticker, period="6mo", interval="1d")
    df.dropna(inplace=True)
    df['Return'] = df['Adj Close'].pct_change()
    df['Amihud'] = (abs(df['Return']) / df['Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Z_Score'] = (df['Amihud'] - df['Amihud'].rolling(20).mean()).fillna(0) / df['Amihud'].rolling(20).std().replace(0, np.nan).fillna(1)
    model = IsolationForest(contamination=0.05, random_state=42)
    df['IF_Anomaly'] = model.fit_predict(df[['Amihud']].fillna(0))
    df = df.reset_index()
    df.rename(columns={'Date': 'date', 'Adj Close': 'adj_close', 'Volume': 'volume'}, inplace=True)
    database.delete_data(ticker)
    database.write_to_db(ticker, df)

if __name__ == '__main__':
    for ticker in tickers:
        update_ticker(ticker)
    print("✅ All tickers updated.")
