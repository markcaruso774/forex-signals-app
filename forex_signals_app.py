import streamlit as st
from twelvedata import TDClient
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import streamlit.components.v1 as components
import time
import math
import talib 

# === CONFIG ===
TD_API_KEY = "e02de9a60165478aaf1da8a7b2096e05" # Mock Key

ALL_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY",
    "USD/CAD", "AUD/USD", "NZD/USD",
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "USD/CHF"
]
FREE_PAIR = "EUR/USD"
PREMIUM_PAIRS = ALL_PAIRS

# FULL TIMEFRAMES SUPPORTED
INTERVALS = {
    "1min": "1min", "5min": "5min", "15min": "15min", 
    "30min": "30min", "1h": "60min"
}
OUTPUTSIZE = 500

# === PAGE CONFIG ===
st.set_page_config(page_title="PipWizard", page_icon="Chart", layout="wide")

# Mock function for data fetching
@st.cache_data(ttl=60)
def fetch_data(symbol, interval):
    """Mocks loading historical data for demonstration purposes."""
    np.random.seed(42) # Seed for reproducible mock data
    
    # Generate mock data
    start_date = pd.to_datetime('2024-01-01')
    freq_map = {"1min": 'T', "5min": '5T', "15min": '15T', "30min": '30T', "1h": 'H'}
    
    periods = OUTPUTSIZE
    if interval in ['1min', '5min']: periods = OUTPUTSIZE * 2
    
    timestamps = pd.date_range(start=start_date, periods=periods, freq=freq_map[interval])
    
    base_price = 1.0850
    noise = np.random.randn(periods) * 0.001 
    
    close = base_price + np.cumsum(noise)
    open_p = close.shift(1).fillna(base_price)
    
    high = np.maximum(open_p, close) + np.abs(np.random.randn(periods) * 0.0001)
    low = np.minimum(open_p, close) - np.abs(np.random.randn(periods) * 0.0001)

    df = pd.DataFrame({'open': open_p, 'high': high, 'low': low, 'close': close}, index=timestamps)
    
    df['high'] = df[['open', 'close']].max(axis=1) + 0.0001
    df['low'] = df[['open', 'close']].min(axis=1) - 0.0001
    
    return df.dropna().tail(OUTPUTSIZE)


# === HELPER FUNCTIONS (Alerts) ===

def send_alert_email(signal_type, price, pair):
    """Mocks sending a real-time email alert to a premium user."""
    st.sidebar.markdown(f"**ALERT SENT**")
    st.sidebar.warning(f"**{signal_type.upper()}** on {pair} at {price:.5f}")

def check_for_live_signal(df, pair):
    """Checks the latest bar of the DataFrame for a BUY or SELL signal."""
    latest_bar = df.iloc[-1]
    signal = latest_bar['signal']
    price = latest_bar['close']
    
    if signal == 1:
        send_alert_email("BUY", price, pair)
    elif signal == -1:
        send_alert_email("SELL", price, pair)


# === THEME ===
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

def apply_theme():
    dark = st.session_state.theme == "dark"
    return f"""
    <style>
        .stApp {{ background-color: {'#0e1117' if dark else '#ffffff'}; color: {'#f0f0f0' if dark else '#212529'}; }}
        .buy-signal {{ color: #26a69a; }}
        .sell-signal {{ color: #ef5350; }}
        .impact-high {{ color: #ef5350; font-weight: bold; }}
        .impact-medium {{ color: #ff9800; }}
        .impact-low {{ color: #26a69a; }}
    </style>
    """

st.markdown(apply_theme(), unsafe_allow_html=True)

# === HEADER ===
col1, col2 = st.columns([6, 1])
with col1:
    st.title("PipWizard – Live Forex Signals")
with col2:
    theme_label = "Light" if st.session_state.theme == "dark" else "Dark"
    if st.button(theme_label, key="theme_toggle", on_click=toggle_theme):
        st.rerun()

# === SIDEBAR & CONTROLS ===
st.sidebar.title("PipWizard")

# PREMIUM LOCK
is_premium = st.sidebar.checkbox("Premium User?", value=True)

if is_premium:
    selected_pair = st.sidebar.selectbox("Select Pair", PREMIUM_PAIRS, index=0)
    st.sidebar.success(f"Premium Active – {len(PREMIUM_PAIRS)} Pairs")
else:
    selected_pair = FREE_PAIR
    st.sidebar.warning("Free Tier: EUR/USD Only")
    st.info("Premium unlocks **10+ pairs** → [Get Premium](#)")

# TIMEFRAME SELECTOR
selected_interval = st.sidebar.selectbox(
    "Timeframe",
    options=list(INTERVALS.keys()),
    index=3, # Default to 30min
    format_func=lambda x: x.replace("min", " minute").replace("1h", "1 hour")
)

# INDICATOR PERIOD CONTROLS
st.sidebar.markdown("---")
st.sidebar.subheader("Indicator Configuration")

# UX Toggles
show_rsi = st.sidebar.checkbox("Show RSI Chart", value=True)
show_macd = st.sidebar.checkbox("Show MACD Chart", value=True) 

