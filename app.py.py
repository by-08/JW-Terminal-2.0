# app.py
import os
import time
import threading
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, send_file
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global configurations - adjustable via /config endpoint (POST)
CONFIG = {
    'body_top_percent': 0.20,  # Body in top 20% of range
    'rel_vol_threshold': 1.5,  # Relative volume > 1.5
    'vol_multiplier': 1.5,     # Today's range > 1.5 * rolling vol
    'historical_periods': 50,  # Last 50 historical hammers
    'return_target': 0.02,     # 2% rise/fall
    'lookforward_periods': 5,  # In 5 hours/bars
    'interval': '1h',          # Default 1h, options: '1d', '4h', '1h', '15m' - but scan is hourly, so '1h' primary
    'historical_start': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
    'historical_end': datetime.now().strftime('%Y-%m-%d'),
    'rolling_periods': 30,     # For rel vol and volatility
    'volatility_measure': 'std_range',  # 'std_range' or 'atr' - simple std of ranges for now
}

# Cache for historical data: dict of ticker: df (OHLCV)
HISTORICAL_CACHE = {}
# Current signals: list of dicts
SIGNALS = []

# Watchlist
WATCHLIST = []

def get_watchlist():
    global WATCHLIST
    if WATCHLIST:
        return WATCHLIST
    # S&P 500
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(sp500_url)[0]
    sp500_tickers = sp500_table['Symbol'].tolist()
    # Nasdaq 100
    nasdaq_url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    nasdaq_table = pd.read_html(nasdaq_url)[1]  # Second table
    nasdaq_tickers = nasdaq_table['Ticker'].tolist()
    # Big ETFs
    etfs = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'EEM', 'XLF', 'XLE', 'XLK', 'XLV']
    WATCHLIST = list(set(sp500_tickers + nasdaq_tickers + etfs))
    return WATCHLIST

def download_historical_data(start, end, interval='1h'):
    """Download and cache historical data for all tickers"""
    global HISTORICAL_CACHE
    tickers = get_watchlist()
    HISTORICAL_CACHE = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if not df.empty:
                df['Ticker'] = ticker
                HISTORICAL_CACHE[ticker] = df
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    # Save cache
    with open('historical_cache.pkl', 'wb') as f:
        pickle.dump(HISTORICAL_CACHE, f)

def load_historical_cache():
    """Load cached historical data"""
    global HISTORICAL_CACHE
    if os.path.exists('historical_cache.pkl'):
        with open('historical_cache.pkl', 'rb') as f:
            HISTORICAL_CACHE = pickle.load(f)

def get_current_data(ticker, period='5d', interval='1h'):
    """Get recent data for scanning"""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if not df.empty:
            df['Range'] = df['High'] - df['Low']
            df['Body'] = abs(df['Close'] - df['Open'])
            df['Body_Low'] = np.minimum(df['Open'], df['Close'])
            df['Lower_Wick'] = df['Body_Low'] - df['Low']
            df['Upper_Wick'] = df['High'] - np.maximum(df['Open'], df['Close'])
            # Rolling avg volume
            df['Avg_Vol'] = df['Volume'].rolling(window=CONFIG['rolling_periods']).mean()
            df['Rel_Vol'] = df['Volume'] / df['Avg_Vol']
            # Volatility: std of ranges
            if CONFIG['volatility_measure'] == 'std_range':
                df['Volatility'] = df['Range'].rolling(window=CONFIG['rolling_periods']).std()
            # Today's range: assume daily high-low, but for hourly, approximate with max range today or fetch 1d
            # For simplicity, use current bar range vs rolling vol
            df['Current_Range'] = df['Range']
            return df
    except:
        return pd.DataFrame()

def is_hammer(df, idx, top_percent):
    """Check if candle at idx is hammer: body in top X% of range"""
    if idx < 0:
        return False
    row = df.iloc[idx]
    range_val = row['Range']
    if range_val == 0:
        return False
    body_low_rel = (row['Body_Low'] - row['Low']) / range_val
    return body_low_rel >= (1 - top_percent)

def is_inverted_hammer(df, idx, bottom_percent=0.20):  # Symmetric for bottom
    """Inverted hammer: body in bottom 20%"""
    if idx < 0:
        return False
    row = df.iloc[idx]
    range_val = row['Range']
    if range_val == 0:
        return False
    body_high_rel = (row['High'] - np.maximum(row['Open'], row['Close'])) / range_val
    return body_high_rel <= bottom_percent

def get_todays_range(ticker):
    """Get today's high-low so far"""
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        daily = yf.download(ticker, start=today, period='1d', interval='1d', progress=False)
        if not daily.empty:
            return daily['High'].max() - daily['Low'].min()
    except:
        pass
    return 0

def scan_for_signals():
    """Scan all tickers for signals"""
    global SIGNALS
    tickers = get_watchlist()
    now = datetime.now()
    current_minute = now.minute
    if current_minute != 55:  # Run only at :55
        return
    new_signals = []
    for ticker in tickers:
        df = get_current_data(ticker)
        if df.empty or len(df) < CONFIG['rolling_periods'] + CONFIG['lookforward_periods']:
            continue
        latest_idx = -1
        row = df.iloc[latest_idx]
        top_percent = CONFIG['body_top_percent']
        
        # Hammer long
        if (is_hammer(df, latest_idx, top_percent) and
            row['Rel_Vol'] > CONFIG['rel_vol_threshold'] and
            row['Current_Range'] > CONFIG['vol_multiplier'] * row['Volatility']):
            
            prob = calculate_probability(ticker, df, is_long=True)
            signal = {
                'ticker': ticker,
                'time': now.strftime('%Y-%m-%d %H:%M'),
                'type': 'Hammer Long',
                'probability': prob,
                'is_long': True
            }
            new_signals.append(signal)
        
        # Inverted Hammer short
        bottom_percent = CONFIG['body_top_percent']  # Symmetric
        if (is_inverted_hammer(df, latest_idx, bottom_percent) and
            row['Rel_Vol'] > CONFIG['rel_vol_threshold'] and
            row['Current_Range'] > CONFIG['vol_multiplier'] * row['Volatility']):
            
            prob = calculate_probability(ticker, df, is_long=False)
            signal = {
                'ticker': ticker,
                'time': now.strftime('%Y-%m-%d %H:%M'),
                'type': 'Inverted Hammer Short',
                'probability': prob,
                'is_long': False
            }
            new_signals.append(signal)
    
    SIGNALS = new_signals  # Update global

def find_historical_hammers(ticker, df_hist, is_long=True, n_periods=50):
    """Find last n historical hammers in historical df"""
    hammers = []
    top_percent = CONFIG['body_top_percent']
    for i in range(len(df_hist) - CONFIG['lookforward_periods'], 0, -1):
        idx = i - 1  # Past
        row = df_hist.iloc[idx]
        rel_vol = row['Rel_Vol']
        volatility = row['Volatility']
        
        if is_long:
            if is_hammer(df_hist, idx, top_percent):
                # Outcome: did close after lookforward > (1+target) * close_at_hammer
                future_close = df_hist.iloc[idx + CONFIG['lookforward_periods']]['Close']
                hammer_close = row['Close']
                outcome = 1 if (future_close > hammer_close * (1 + CONFIG['return_target'])) else 0
                hammers.append({'rel_vol': rel_vol, 'volatility': volatility, 'outcome': outcome})
        else:
            if is_inverted_hammer(df_hist, idx, top_percent):
                future_close = df_hist.iloc[idx + CONFIG['lookforward_periods']]['Close']
                hammer_close = row['Close']
                outcome = 1 if (future_close < hammer_close * (1 - CONFIG['return_target'])) else 0
                hammers.append({'rel_vol': rel_vol, 'volatility': volatility, 'outcome': outcome})
        
        if len(hammers) >= n_periods:
            break
    return hammers[:n_periods]

def calculate_probability(ticker, current_df, is_long=True):
    """Fit logistic regression on historical hammers and predict"""
    if ticker not in HISTORICAL_CACHE:
        return 0.5  # Default
    df_hist = HISTORICAL_CACHE[ticker].copy()
    # Align interval, assume same
    df_hist['Range'] = df_hist['High'] - df_hist['Low']
    df_hist['Body'] = abs(df_hist['Close'] - df_hist['Open'])
    df_hist['Body_Low'] = np.minimum(df_hist['Open'], df_hist['Close'])
    df_hist['Lower_Wick'] = df_hist['Body_Low'] - df_hist['Low']
    df_hist['Upper_Wick'] = df_hist['High'] - np.maximum(df_hist['Open'], df_hist['Close'])
    df_hist['Avg_Vol'] = df_hist['Volume'].rolling(window=CONFIG['rolling_periods']).mean()
    df_hist['Rel_Vol'] = df_hist['Volume'] / df_hist['Avg_Vol']
    if CONFIG['volatility_measure'] == 'std_range':
        df_hist['Volatility'] = df_hist['Range'].rolling(window=CONFIG['rolling_periods']).std()
    
    hammers = find_historical_hammers(ticker, df_hist, is_long, CONFIG['historical_periods'])
    if len(hammers) < 10:  # Min for fit
        return 0.5
    
    X = np.array([[h['rel_vol'], h['volatility']] for h in hammers])
    y = np.array([h['outcome'] for h in hammers])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    # Current features
    current_row = current_df.iloc[-1]
    current_features = np.array([[current_row['Rel_Vol'], current_row['Volatility']]])
    current_scaled = scaler.transform(current_features)
    
    prob = model.predict_proba(current_scaled)[0][1]
    return prob

def create_plotly_chart(ticker, signal_time, is_long=True):
    """Create Plotly chart with highlighted candle"""
    df = get_current_data(ticker, period='10d')  # Last 10 days
    if df.empty:
        return None
    
    # Highlight the signal candle - assume last one for simplicity
    highlight_idx = -1
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Price', 'Volume'),
                        row_width=[0.7, 0.3])
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name='OHLC',
                                 increasing_line_color='#00ff00', decreasing_line_color='#ff0000'), row=1, col=1)
    
    # Highlight
    highlight_candle = df.iloc[highlight_idx]
    fig.add_hrect(y0=highlight_candle['Low'], y1=highlight_candle['High'],
                  fillcolor='yellow', opacity=0.3, line_width=0, row=1, col=1)
    
    # Volume
    colors = ['green' if df['Close'][i] >= df['Open'][i] else 'red' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=2, col=1)
    
    fig.update_layout(title=f'{ticker} Chart - Signal at {signal_time}',
                      xaxis_rangeslider_visible=False,
                      height=600, template='plotly_dark')
    fig.update_xaxes(rangeslider_visible=False)
    
    chart_html = fig.to_html(full_html=False)
    return chart_html

# Background thread
def background_scanner():
    while True:
        now = datetime.now()
        minutes_to_55 = (55 - now.minute) % 60
        seconds_to_wait = minutes_to_55 * 60 + (60 - now.second)
        time.sleep(seconds_to_wait)
        scan_for_signals()
        time.sleep(60)  # Wait one more min to ensure bar close

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signals')
def get_signals():
    return jsonify(SIGNALS)

@app.route('/config', methods=['GET', 'POST'])
def config():
    if request.method == 'POST':
        global CONFIG
        data = request.json
        for key, value in data.items():
            if key in CONFIG:
                CONFIG[key] = value
        # If historical dates changed, download
        if 'historical_start' in data or 'historical_end' in data:
            download_historical_data(CONFIG['historical_start'], CONFIG['historical_end'], CONFIG['interval'])
        return jsonify({'status': 'updated'})
    return jsonify(CONFIG)

@app.route('/chart/<ticker>/<signal_time>/<is_long>')
def chart(ticker, signal_time, is_long):
    is_long_bool = is_long == 'True'
    chart_html = create_plotly_chart(ticker, signal_time, is_long_bool)
    if chart_html:
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>{ticker} Chart</title><script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>
        <body style="background: #1a1a1a; color: white;">
            <div>{chart_html}</div>
        </body>
        </html>
        """
    return "Chart not available", 404

@app.route('/download_historical')
def download_historical():
    start = request.args.get('start', CONFIG['historical_start'])
    end = request.args.get('end', CONFIG['historical_end'])
    interval = request.args.get('interval', CONFIG['interval'])
    download_historical_data(start, end, interval)
    load_historical_cache()  # Reload
    return jsonify({'status': 'downloaded'})

if __name__ == '__main__':
    get_watchlist()
    load_historical_cache()
    # Start background thread
    scanner_thread = threading.Thread(target=background_scanner, daemon=True)
    scanner_thread.start()
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))