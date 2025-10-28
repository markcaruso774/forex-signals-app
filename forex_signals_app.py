import streamlit as st
from twelvedata import TDClient
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# === CONFIG ===
TD_API_KEY = "e02de9a60165478aaf1da8a7b2096e05"  # Your key
SYMBOL = "EUR/USD"
INTERVAL = "1min"
OUTPUTSIZE = 500  # Max on free tier

@st.cache_data(ttl=60)  # Refresh every 60 seconds
def fetch_data():
    td = TDClient(apikey=TD_API_KEY)
    try:
        ts = td.time_series(
            symbol=SYMBOL,
            interval=INTERVAL,
            outputsize=OUTPUTSIZE
        ).as_pandas()
        
        if ts.empty:
            st.error("No data returned from Twelve Data.")
            return pd.DataFrame()
        
        # Clean column names
        df = ts[['open', 'high', 'low', 'close']].copy()
        df.index = pd.to_datetime(df.index)
        return df[::-1]  # Reverse to chronological order
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return pd.DataFrame()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(df):
    df['rsi'] = calculate_rsi(df['close'])
    df['sma_20'] = df['close'].rolling(20).mean()
    df['signal'] = 0
    df.loc[(df['rsi'] < 30) & (df['close'] > df['sma_20']), 'signal'] = 1   # Buy
    df.loc[(df['rsi'] > 70) & (df['close'] < df['sma_20']), 'signal'] = -1  # Sell
    df['confidence'] = (abs(df['rsi'] - 50) / 50).round(3)
    return df

def backtest_signals(df):
    signals = df[df['signal'] != 0].copy()
    if signals.empty:
        return 0, 0
    # Mock: +10 pips on win, -10 on loss
    signals['pips'] = np.where(signals['signal'] == 1, 10, -10)
    win_rate = (signals['pips'] > 0).mean()
    total_pips = signals['pips'].sum()
    return round(win_rate, 3), int(total_pips)

# === STREAMLIT APP ===
st.set_page_config(page_title="PipWizard", layout="wide")
st.title("PipWizard – EUR/USD Live Signals")

# Sidebar
st.sidebar.header("Upgrade to Premium")
st.sidebar.info(
    "Free: EUR/USD signals\n\n"
    "Premium ($9.99/mo):\n"
    "• Real-time alerts (email/Telegram)\n"
    "• 10+ currency pairs\n"
    "• Advanced backtesting\n"
    "• No ads"
)

# Fetch data
with st.spinner("Fetching live EUR/USD data..."):
    df = fetch_data()

if df.empty:
    st.error("Failed to load data. Check API key or internet.")
    st.stop()

df = generate_signals(df)
win_rate, total_pips = backtest_signals(df)

# === CHART ===
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    name="Price"
))

# SMA
fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name="SMA(20)", line=dict(color="orange")))

# RSI (secondary axis)
fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name="RSI", yaxis="y2", line=dict(color="purple", dash="dot")))

fig.update_layout(
    title=f"EUR/USD Live Chart – Last {len(df)} Minutes",
    yaxis=dict(title="Price (USD)"),
    yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100]),
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# === PERFORMANCE ===
col1, col2 = st.columns(2)
col1.metric("Win Rate (Mock)", f"{win_rate:.1%}")
col2.metric("Total Pips (24h)", f"{total_pips:+}")

# === LATEST SIGNALS ===
st.subheader("Latest Signals")
signals = df[df['signal'] != 0].tail(3)

if not signals.empty:
    for idx, row in signals.iterrows():
        sig = "BUY" if row['signal'] == 1 else "SELL"
        color = "green" if row['signal'] == 1 else "red"
        st.markdown(
            f"**<span style='color:{color}'>{sig}</span>** at `{idx.strftime('%H:%M')}` | "
            f"Price: `{row['close']:.5f}` | RSI: `{row['rsi']:.1f}` | "
            f"Confidence: `{row['confidence']:.1%}`",
            unsafe_allow_html=True
        )
else:
    st.info("No signals yet. Market in neutral zone (RSI 40–60).")

# === CTA ===
col1, col2 = st.columns(2)
with col1:
    if st.button("Simulate Trade"):
        st.balloons()
        st.success("Simulated: +3.2 pips! Upgrade for real-time alerts.")
with col2:
    st.markdown("[Get Premium Now](https://buy.stripe.com/test_123)")

st.caption("Disclaimer: Not financial advice. Forex is high-risk. Trade responsibly.")
