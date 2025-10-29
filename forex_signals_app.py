import streamlit as st
from twelvedata import TDClient
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import streamlit.components.v1 as components
import talib

# === CONFIG ===
TD_API_KEY = "e02de9a60165478aaf1da8a7b2096e05"

ALL_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY",
    "USD/CAD", "AUD/USD", "NZD/USD",
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "USD/CHF"
]
FREE_PAIR = "EUR/USD"
PREMIUM_PAIRS = ALL_PAIRS

INTERVALS = {
    "1min": "1min", "5min": "5min", "15min": "15min", 
    "30min": "30min", "1h": "60min"
}
OUTPUTSIZE = 500

# === PAGE CONFIG ===
st.set_page_config(page_title="PipWizard", page_icon="PIP", layout="wide")

# === LIVE DATA FETCH ===
@st.cache_data(ttl=60)
def fetch_data(symbol, interval):
    td = TDClient(apikey=TD_API_KEY)
    try:
        ts = td.time_series(
            symbol=symbol,
            interval=INTERVALS[interval],
            outputsize=OUTPUTSIZE
        ).as_pandas()
        if ts.empty:
            return pd.DataFrame()
        df = ts[['open', 'high', 'low', 'close']].copy()
        df.index = pd.to_datetime(df.index)
        return df[::-1].tail(500)
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return pd.DataFrame()

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
        .buy-signal {{ color: #26a69a; font-weight: bold; }}
        .sell-signal {{ color: #ef5350; font-weight: bold; }}
        .impact-high {{ color: #ef5350; font-weight: bold; }}
        .impact-medium {{ color: #ff9800; }}
        .impact-low {{ color: #26a69a; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ padding: 8px; text-align: left; }}
        th {{ background-color: #1f1f1f; color: #ccc; }}
        tr {{ border-bottom: 1px solid #333; }}
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

# TIMEFRAME
selected_interval = st.sidebar.selectbox(
    "Timeframe",
    options=list(INTERVALS.keys()),
    index=0,
    format_func=lambda x: x.replace("min", " minute").replace("1h", "1 hour")
)

# INDICATORS
st.sidebar.markdown("---")
st.sidebar.subheader("Indicators")
show_rsi = st.sidebar.checkbox("Show RSI", True)
show_macd = st.sidebar.checkbox("Show MACD", True)

rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
sma_period = st.sidebar.slider("SMA Period", 10, 50, 20)
alert_rsi_low = st.sidebar.slider("Buy RSI <", 20, 40, 30)
alert_rsi_high = st.sidebar.slider("Sell RSI >", 60, 80, 70)

macd_fast = st.sidebar.slider("MACD Fast", 1, 26, 12)
macd_slow = st.sidebar.slider("MACD Slow", 13, 50, 26)
macd_signal = st.sidebar.slider("MACD Signal", 1, 15, 9)

# BACKTESTING
st.sidebar.markdown("---")
st.sidebar.subheader("Backtest")
initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000, 100000, 10000)
risk_pct = st.sidebar.slider("Risk %", 0.5, 5.0, 1.0) / 100
sl_pips = st.sidebar.number_input("Stop Loss (pips)", 10, 200, 50)
tp_pips = st.sidebar.number_input("Take Profit (pips)", 10, 300, 75)

st.sidebar.markdown("---")
st.sidebar.markdown("[Get Premium](https://buy.stripe.com/test_123)")

# === FETCH & INDICATORS ===
df = fetch_data(selected_pair, selected_interval)
if df.empty:
    st.error("No data. Check API key.")
    st.stop()

df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)
df['sma'] = df['close'].rolling(sma_period).mean()
df['macd_line'], df['macd_signal'], df['macd_hist'] = talib.MACD(
    df['close'], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal
)

# SIGNALS
df['signal'] = 0
df.loc[(df['rsi'] < alert_rsi_low) & (df['close'] > df['sma']), 'signal'] = 1
df.loc[(df['rsi'] > alert_rsi_high) & (df['close'] < df['sma']), 'signal'] = -1
df = df.dropna()

# === BACKTEST (PREMIUM) ===
def run_backtest(df, capital, risk, sl, tp):
    df = df.copy()
    df['entry_signal'] = df['signal'].shift(1).fillna(0)
    trades = df[df['entry_signal'] != 0].copy()
    if trades.empty:
        return 0, 0, 0, capital, pd.DataFrame()

    pip = 0.0001
    risk_usd = capital * risk
    reward_usd = risk_usd * (tp / sl)

    trades['entry'] = trades['open']
    trades.loc[trades['entry_signal'] == 1, 'sl'] = trades['entry'] - sl * pip
    trades.loc[trades['entry_signal'] == 1, 'tp'] = trades['entry'] + tp * pip
    trades.loc[trades['entry_signal'] == -1, 'sl'] = trades['entry'] + sl * pip
    trades.loc[trades['entry_signal'] == -1, 'tp'] = trades['entry'] - tp * pip

    trades['result'] = 'LOSS'
    trades.loc[(trades['entry_signal'] == 1) & (trades['high'] >= trades['tp']), 'result'] = 'WIN'
    trades.loc[(trades['entry_signal'] == -1) & (trades['low'] <= trades['tp']), 'result'] = 'WIN'
    trades['pl'] = np.where(trades['result'] == 'WIN', reward_usd, -risk_usd)

    win_rate = (trades['result'] == 'WIN').mean()
    total_pl = trades['pl'].sum()
    final_cap = capital + total_pl

    trades['direction'] = trades['entry_signal'].map({1: 'BUY', -1: 'SELL'})
    return len(trades), win_rate, total_pl, final_cap, trades[['direction', 'entry', 'result', 'pl']]

if is_premium:
    trades, win_rate, profit, final_cap, log = run_backtest(df, initial_capital, risk_pct, sl_pips, tp_pips)
else:
    trades = win_rate = profit = final_cap = log = 0

# === CHART ===
st.subheader(f"**{selected_pair}** – **{selected_interval}** – Last {len(df)} Candles")
chart_type = st.radio("Chart", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")

rows = 1 + show_rsi + show_macd
heights = [0.6] + [0.2] * (rows - 1)
fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=heights)

# Price
if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
else:
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name="Price", line=dict(color="#2196f3")), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['sma'], name=f"SMA({sma_period})", line=dict(color="#ff9800")), row=1, col=1)

# Signals
buy = df[df['signal'] == 1]
sell = df[df['signal'] == -1]
fig.add_trace(go.Scatter(x=buy.index, y=buy['low'] * 0.999, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#26a69a'), name='BUY'), row=1, col=1)
fig.add_trace(go.Scatter(x=sell.index, y=sell['high'] * 1.001, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ef5350'), name='SELL'), row=1, col=1)

# RSI
if show_rsi:
    r = 2 if show_macd else 2
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name=f"RSI({rsi_period})", line=dict(color="#9c27b0")), row=r, col=1)
    fig.add_hline(y=alert_rsi_low, line_dash="dash", line_color="green", row=r, col=1)
    fig.add_hline(y=alert_rsi_high, line_dash="dash", line_color="red", row=r, col=1)

# MACD
if show_macd:
    r = 3 if show_rsi else 2
    fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name="Hist", marker_color=np.where(df['macd_hist'] >= 0, '#26a69a', '#ef5350')), row=r, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_line'], name="MACD", line=dict(color="#00bfa5")), row=r, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name="Signal", line=dict(color="#ff9800")), row=r, col=1)

fig.update_layout(height=700, template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white")
st.plotly_chart(fig, use_container_width=True)

# === ECONOMIC CALENDAR (FIXED) ===
st.subheader("Economic Calendar (Today & Tomorrow)")

today_utc = datetime.utcnow().strftime("%Y-%m-%d")
tomorrow_utc = (datetime.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

events = [
    {"date": today_utc, "time": "13:30", "event": "US Nonfarm Payrolls", "impact": "high"},
    {"date": today_utc, "time": "14:00", "event": "US Unemployment Rate", "impact": "high"},
    {"date": today_utc, "time": "15:00", "event": "ISM Manufacturing PMI", "impact": "medium"},
    {"date": tomorrow_utc, "time": "12:30", "event": "ECB Interest Rate Decision", "impact": "high"},
    {"date": tomorrow_utc, "time": "14:00", "event": "US Retail Sales", "impact": "medium"},
]

today_events = [e for e in events if e["date"] in [today_utc, tomorrow_utc]]

if today_events:
    calendar_html = """
    <table>
        <tr>
            <th>Date</th>
            <th>Time (UTC)</th>
            <th>Event</th>
            <th>Impact</th>
        </tr>
    """
    for e in today_events:
        impact_class = {"high": "impact-high", "medium": "impact-medium", "low": "impact-low"}.get(e["impact"], "impact-low")
        calendar_html += f"""
        <tr>
            <td>{e['date']}</td>
            <td>{e['time']}</td>
            <td>{e['event']}</td>
            <td><span class='{impact_class}'>• {e['impact'].title()}</span></td>
        </tr>
        """
    calendar_html += "</table>"
    st.markdown(calendar_html, unsafe_allow_html=True)
else:
    st.info("No major events scheduled.")

# === BACKTEST RESULTS ===
if is_premium and trades > 0:
    st.subheader("Backtest Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", trades)
    c2.metric("Win Rate", f"{win_rate:.1%}")
    c3.metric("Profit", f"${profit:,.0f}")
    c4.metric("Final", f"${final_cap:,.0f}")
    st.dataframe(log)

# === LIVE SIGNAL ===
st.subheader("Latest Signal")
latest = df.iloc[-1]
sig = "BUY" if latest['signal'] == 1 else "SELL" if latest['signal'] == -1 else "NEUTRAL"
color = "buy-signal" if sig == "BUY" else "sell-signal" if sig == "SELL" else ""
st.markdown(f"<p class='{color}'><b>{sig}</b> at {latest['close']:.5f} | RSI: {latest['rsi']:.1f}</p>", unsafe_allow_html=True)

# === AUTO REFRESH ===
components.html("<meta http-equiv='refresh' content='61'>", height=0)
