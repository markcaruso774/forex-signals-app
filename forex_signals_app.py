import streamlit as st
from twelvedata import TDClient
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import streamlit.components.v1 as components

# === CONFIG ===
TD_API_KEY = "e02de9a60165478aaf1da8a7b2096e05"

# ALL SUPPORTED PAIRS (Add more anytime!)
ALL_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY",
    "USD/CAD", "AUD/USD", "NZD/USD",
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "USD/CHF"
]

# Free = Only 1, Premium = All
FREE_PAIR = "EUR/USD"
PREMIUM_PAIRS = ALL_PAIRS

INTERVALS = {"1min": "1min", "5min": "5min", "15min": "15min", "1h": "60min"}
OUTPUTSIZE = 500

# === PAGE CONFIG ===
st.set_page_config(page_title="PipWizard", page_icon="💹", layout="wide")

# === THEME ===
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

def apply_theme():
    if st.session_state.theme == "dark":
        return """
        <style>
            .stApp { background-color: #0e1117; color: #f0f0f0; }
            .buy-signal { color: #26a69a; }
            .sell-signal { color: #ef5350; }
            .impact-high { color: #ef5350; font-weight: bold; }
            .impact-medium { color: #ff9800; }
            .impact-low { color: #26a69a; }
        </style>
        """
    else:
        return """
        <style>
            .stApp { background-color: #ffffff; color: #212529; }
            .buy-signal { color: #28a745; }
            .sell-signal { color: #dc3545; }
            .impact-high { color: #dc3545; font-weight: bold; }
            .impact-medium { color: #fd7e14; }
            .impact-low { color: #28a745; }
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

# === SIDEBAR ===
st.sidebar.title("PipWizard")

# PREMIUM LOCK
is_premium = st.sidebar.checkbox("Premium User?", value=False)

if is_premium:
    selected_pair = st.sidebar.selectbox("Select Pair", PREMIUM_PAIRS, index=0)
    st.sidebar.success(f"Premium Active – {len(PREMIUM_PAIRS)} Pairs")
else:
    selected_pair = FREE_PAIR
    st.sidebar.warning("Free Tier: EUR/USD Only")
    st.info("Premium unlocks **10+ pairs** → [Get Premium](#)")

selected_interval = st.sidebar.selectbox("Timeframe", list(INTERVALS.keys()), index=0)
alert_rsi_low = st.sidebar.slider("Buy RSI <", 20, 40, 30)
alert_rsi_high = st.sidebar.slider("Sell RSI >", 60, 80, 70)
st.sidebar.markdown("---")
st.sidebar.info("Premium ($9.99/mo):\n• 10+ Pairs\n• Real-time Alerts\n• Backtesting")
st.sidebar.markdown("[Get Premium](https://buy.stripe.com/test_123)")

# === FETCH DATA ===
@st.cache_data(ttl=60)
def fetch_data(symbol, interval):
    td = TDClient(apikey=TD_API_KEY)
    try:
        ts = td.time_series(symbol=symbol, interval=INTERVALS[interval], outputsize=OUTPUTSIZE).as_pandas()
        if ts.empty:
            return pd.DataFrame()
        df = ts[['open', 'high', 'low', 'close']].copy()
        df.index = pd.to_datetime(df.index)
        return df[::-1].tail(500)
    except Exception as e:
        st.error(f"Data Error: {str(e)}")
        return pd.DataFrame()

df = fetch_data(selected_pair, selected_interval)
if df.empty:
    st.error("No data. Check connection or API key.")
    st.stop()

# === SIGNALS ===
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi'] = calculate_rsi(df['close'])
df['sma_20'] = df['close'].rolling(20).mean()
df['signal'] = 0
df.loc[(df['rsi'] < alert_rsi_low) & (df['close'] > df['sma_20']), 'signal'] = 1
df.loc[(df['rsi'] > alert_rsi_high) & (df['close'] < df['sma_20']), 'signal'] = -1
df['confidence'] = (abs(df['rsi'] - 50) / 50).round(3)
df = df.dropna()

if df.empty:
    st.warning("Waiting for data...")
    st.stop()

# === CHART ===
st.subheader(f"{selected_pair} – {selected_interval} – Last {len(df)} Candles")

chart_type = st.radio("Chart Type", ["Candlestick", "Line", "Bar"], horizontal=True, label_visibility="collapsed")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                                 name="Price"), row=1, col=1)
elif chart_type == "Line":
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name="Price", line=dict(color="#2196f3")), row=1, col=1)
elif chart_type == "Bar":
    fig.add_trace(go.Ohlc(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name="SMA(20)", line=dict(color="#ff9800")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name="RSI", line=dict(color="#9c27b0")), row=2, col=1)
fig.add_hline(y=alert_rsi_high, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=alert_rsi_low, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(
    xaxis_rangeslider_visible=False,
    template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
    height=550,
    yaxis1_title="Price",
    yaxis2_title="RSI",
    yaxis2_range=[0, 100]
)
st.plotly_chart(fig, use_container_width=True)

# === DASHBOARD ===
col1, col2 = st.columns(2)
with col1:
    st.subheader("Live Status")
    latest = df.iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Price", f"{latest['close']:.5f}")
    m2.metric("RSI", f"{latest['rsi']:.1f}")
    m3.metric("SMA(20)", f"{latest['sma_20']:.5f}")

    st.subheader("Latest Signals")
    signals_df = df[df['signal'] != 0].tail(5)
    if not signals_df.empty:
        for idx, row in signals_df.iterrows():
            sig = "BUY" if row['signal'] == 1 else "SELL"
            color = "buy-signal" if row['signal'] == 1 else "sell-signal"
            st.markdown(f"<span class='{color}'>**{sig}**</span> `{idx.strftime('%m-%d %H:%M')}` | `{row['close']:.5f}`", unsafe_allow_html=True)
    else:
        st.info(f"RSI {latest['rsi']:.1f} – Neutral")

with col2:
    st.subheader("Mock Backtest")
    signals = df[df['signal'] != 0].copy()
    if not signals.empty:
        signals['pips'] = np.where(signals['signal'] == 1, 10, -10)
        win_rate = (signals['pips'] > 0).mean()
        total_pips = signals['pips'].sum()
        b1, b2 = st.columns(2)
        b1.metric("Win Rate", f"{win_rate:.1%}")
        b2.metric("Pips", f"{total_pips:+}")
    else:
        st.info("No signals")

    st.subheader("Economic Calendar (Today)")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    st.markdown(f"""
    | Date | Time (UTC) | Event | Impact |
    |------|------------|-------|--------|
    | {today} | 14:00 | US GDP | <span class='impact-high'>High</span> |
    | {today} | 16:00 | EU CPI | <span class='impact-medium'>Medium</span> |
    | {today} | 18:00 | FOMC | <span class='impact-high'>High</span> |
    """, unsafe_allow_html=True)

# === CTA ===
st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    if st.button("Simulate Trade", type="primary"):
        st.balloons()
        st.success("+3.2 pips!")
with c2:
    st.markdown("[Get Premium Now](https://buy.stripe.com/test_123)")

st.caption("Not financial advice. Trade responsibly.")

# === AUTO REFRESH ===
components.html("<meta http-equiv='refresh' content='61'>", height=0)
