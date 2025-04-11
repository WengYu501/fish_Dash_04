import yfinance as yf
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from sklearn.ensemble import IsolationForest
import plotly.graph_objs as go
from datetime import datetime as dt
import database

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Liquidity Dashboard"

stock_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

def fetch_data(ticker):
    df_cached = database.load_from_db(ticker)
    if df_cached is not None:
        return df_cached

    df = yf.download(ticker, period="6mo", interval="1d")
    df.dropna(inplace=True)
    df['Return'] = df['Adj Close'].pct_change()
    df['Amihud'] = (abs(df['Return']) / df['Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Z_Score'] = (df['Amihud'] - df['Amihud'].rolling(20).mean()).fillna(0) / df['Amihud'].rolling(20).std().replace(0, np.nan).fillna(1)
    model = IsolationForest(contamination=0.05, random_state=42)
    df['IF_Anomaly'] = model.fit_predict(df[['Amihud']].fillna(0))
    df = df.reset_index()
    df.rename(columns={'Date': 'date', 'Adj Close': 'adj_close', 'Volume': 'volume'}, inplace=True)
    database.write_to_db(ticker, df)
    df.set_index('date', inplace=True)
    return df

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("📈 Stock Liquidity Dashboard"), width=8),
        dbc.Col(dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': s, 'value': s} for s in stock_list],
            value='AAPL', clearable=False
        ), width=4)
    ]),
    html.Hr(),
    dcc.Tabs(id='tabs', value='overview', children=[
        dcc.Tab(label='Liquidity Overview', value='overview'),
        dcc.Tab(label='Liquidity Backtest', value='backtest')
    ]),
    html.Div(id='tab-content')
])

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('stock-dropdown', 'value')
)
def render_tab(tab, ticker):
    df = fetch_data(ticker)
    if tab == 'overview':
        fig_amihud = go.Figure()
        fig_amihud.add_trace(go.Scatter(x=df.index, y=df['amihud'], mode='lines', name='Amihud Ratio'))
        fig_amihud.add_trace(go.Scatter(
            x=df[df['if_anomaly'] == -1].index,
            y=df[df['if_anomaly'] == -1]['amihud'],
            mode='markers', marker=dict(color='red', size=6), name='Anomaly'))
        fig_amihud.update_layout(title=f'{ticker} Amihud Illiquidity with Anomalies')

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume'))
        fig_vol.update_layout(title=f'{ticker} Daily Volume')

        return html.Div([
            dcc.Graph(figure=fig_amihud),
            dcc.Graph(figure=fig_vol)
        ])

    elif tab == 'backtest':
        return html.Div([
            html.Br(),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=df.index.min().date(),
                max_date_allowed=df.index.max().date(),
                start_date=df.index[-60].date(),
                end_date=df.index[-1].date()
            ),
            html.Div(id='backtest-graph')
        ])

@app.callback(
    Output('backtest-graph', 'children'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('stock-dropdown', 'value')
)
def update_backtest(start_date, end_date, ticker):
    df = fetch_data(ticker)
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    df_range = df.loc[mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_range.index, y=df_range['return'].cumsum(), mode='lines', name='Cumulative Return'))
    fig.add_trace(go.Scatter(x=df_range.index, y=df_range['amihud'], mode='lines', name='Amihud Ratio', yaxis='y2'))
    fig.update_layout(
        title=f'{ticker} Backtest: Return vs Amihud',
        yaxis=dict(title='Cumulative Return'),
        yaxis2=dict(title='Amihud Ratio', overlaying='y', side='right')
    )
    return dcc.Graph(figure=fig)

if __name__ == '__main__':
    app.run(debug=False)
