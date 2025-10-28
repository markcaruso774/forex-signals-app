import streamlit as st
from twelvedata import TDClient
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# === CONFIG ===
TD_API_KEY = "e02de9a60165478aaf1da8a7b2096e05"
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY"]
INTERVAL = "1min"
OUTPUTSIZE = 500

# === PAGE CONFIG ===
st.set_page_config(page_title="PipWizard", page_icon="💹", layout="wide")

# === THEME TOGGLE ===
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# === APPLY THEME ===
def apply_theme():
    if st.session_state.theme == "dark":
        return """
        <style>
            .stApp { background-color: #0e1117; color: #f0f0f0; }
            .stMetric > label { color: #b0b0b0; }
            .stMetric > div > div { color: #ffffff; }
            .buy-signal { color: #26a69a; }
            .sell-signal { color: #ef5350; }
            .stSelectbox > div > div { background-color: #1f1f1f; color: #f0f0f0; }
            .stSlider > div > div { background-color: #333; }
            .stTable { color: #f0f0f0; }
        </style>
        """
    else:  # light
        return """
        <style>
            .stApp { background-color: #ffffff; color: #212529; }
            .stMetric > label { color: #6c757d; }
            .stMetric > div > div { color: #212529; }
            .buy-signal { color: #28a745; }
            .sell-signal { color: #dc3545; }
            .stSelectbox > div > div { background-color: #f8f9fa; color: #212529; }
            .stSlider > div > div { background-color: #e9ecef; }
            .stTable { color: #212529; }
        </style>
        """

st.markdown(apply_theme(), unsafe_allow_html=True)

# === HEADER WITH TOGGLE BUTTON ===
col1, col2 = st.columns([6, 1])
with col1:
    st.title("PipWizard – Live Forex Signals")
with col2:
    theme_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
    if st.button(theme_label, key="theme_toggle", on_click=toggle_theme):
        st.rerun()

# === SIDEBAR ===
st.sidebar.title("PipWizard")
selected_pair = st.sidebar.selectbox("Select Pair", PAIRS, index=0)
alert_rsi_low = st.sidebar.slider("Alert RSI <", 20, 40, 30)
alert_rsi_high = st.sidebar.slider("Alert RSI >", 60, 80, 70)
st.sidebar.markdown("---")
st.sidebar.info("Premium ($9.99/mo): Alerts, 10+ Pairs, Backtesting")
st.sidebar.markdown("[Get Premium](https://buy.stripe.com/test_123)")

# === FETCH DATA ===
@st.cache_data(ttl=60)
def fetch_data(symbol):
    td = TDClient(apikey=TD_API_KEY)
    try:
        ts = td.time_series(symbol=symbol, interval=INTERVAL, outputsize=OUTPUTSIZE).as_pandas()
        if ts.empty:
            return pd.DataFrame()
        df = ts[['open', 'high', 'low', 'close']].copy()
        df.index = pd.to_datetime(df.index)
        return df[::-1].tail(500)
    except Exception as e:
        st.error(f"Data Error: {str(e)}")
        return pd.DataFrame()

df = fetch_data(selected_pair)
if df.empty:
    st.error("No data. Check connection.")
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
df.loc[(df['rsi'] < 30) & (df['close'] > df['sma_20']), 'signal'] = 1
df.loc[(df['rsi'] > 70) & (df['close'] < df['sma_20']), 'signal'] = -1
df['confidence'] = (abs(df['rsi'] - 50) / 50).round(3)

# === CHART ===
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    name="Price",
    increasing_line_color="#26a69a" if st.session_state.theme == "dark" else "#28a745",
    decreasing_line_color="#ef5350" if st.session_state.theme == "dark" else "#dc3545"
))
fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name="SMA(20)", line=dict(color="#ff9800")))
fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name="RSI", yaxis="y2", line=dict(color="#9c27b0")))
fig.update_layout(
    title=f"{selected_pair} – Last {len(df)} Minutes",
    yaxis=dict(title="Price"),
    yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100]),
    xaxis_rangeslider_visible=False,
    template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# === METRICS + SIGNALS ===
col1, col2, col3 = st.columns(3)
col1.metric("Price", f"{df['close'].iloc[-1]:.5f}")
col2.metric("RSI", f"{df['rsi'].iloc[-1]:.1f}")
col3.metric("SMA(20)", f"{df['sma_20'].iloc[-1]:.5f}")

st.subheader("Latest Signals")
signals_df = df[df['signal'] != 0].tail(5)
if not signals_df.empty:
    for idx, row in signals_df.iterrows():
        sig = "BUY" if row['signal'] == 1 else "SELL"
        color_class = "buy-signal" if row['signal'] == 1 else "sell-signal"
        st.markdown(f"<span class='{color_class}'>**{sig}**</span> at `{idx.strftime('%H:%M')}` | "
                    f"Price: `{row['close']:.5f}` | RSI: `{row['rsi']:.1f}`", unsafe_allow_html=True)
else:
    st.info("No signals. RSI in neutral zone.")

# === BACKTEST ===
st.subheader("Mock Backtest")
signals = df[df['signal'] != 0]
if not signals.empty:
    signals['pips'] = np.where(signals['signal'] == 1, 10, -10)
    win_rate = (signals['pips'] > 0).mean()
    total_pips = signals['pips'].sum()
    col1, col2 = st.columns(2)
    col1.metric("Win Rate", f"{win_rate:.1%}")
    col2.metric("Pips", f"{total_pips:+}")
else:
    st.info("No signals to backtest.")

# === ECONOMIC CALENDAR (Mock) ===
st.subheader("Economic Calendar")
calendar = pd.DataFrame({
    "Time": ["14:00", "16:00", "18:00"],
    "Event": ["US GDP", "EU CPI", "FOMC"],
    "Impact": ["High", "Medium", "High"]
})
st.table(calendar)

# === CTA ===
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("Simulate Trade", type="primary"):
        st.balloons()
        st.success("+3.2 pips!")
with col2:
    st.markdown("[Get Premium Now](https://buy.stripe.com/test_123)")

st.caption("Not financial advice. Trade responsibly.")
