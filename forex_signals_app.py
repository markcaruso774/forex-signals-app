import streamlit as st
from alpha_vantage.foreignexchange import ForeignExchange
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Config
ALPHA_API_KEY = "QSSK6RC0JXCMF7EN"  # Your Alpha Vantage API key
TICKER = "EUR/USD"  # Alpha Vantage forex pair format

@st.cache_data(ttl=60)  # Cache for 1 min to avoid API limits
def fetch_data(days_back=1):
    cc = ForeignExchange(key=ALPHA_API_KEY, output_format='pandas')
    try:
        data, meta = cc.get_currency_exchange_intraday(
            from_symbol='EUR',
            to_symbol='USD',
            interval='1min',
            outputsize='compact'  # ~100 data points on free tier
        )
        df = data
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close'})
            df = df[['open', 'high', 'low', 'close']].tail(500)  # Limit to recent data
        return df
    except Exception as e:
        st.error(f"API error: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(df):
    df['rsi'] = calculate_rsi(df['close'])
    df['sma'] = df['close'].rolling(window=20).mean()
    df['signal'] = 0
    df.loc[(df['rsi'] < 30) & (df['close'] > df['sma']), 'signal'] = 1  # Buy
    df.loc[(df['rsi'] > 70) & (df['close'] < df['sma']), 'signal'] = -1  # Sell
    df['confidence'] = np.abs(df['rsi'] - 50) / 50
    return df

def simulate_backtest(df):
    signals = df[df['signal'] != 0].copy()
    if signals.empty:
        return 0, 0
    signals['result'] = np.where(signals['signal'] == 1, 10, -10)  # Mock pip gain/loss
    win_rate = len(signals[signals['result'] > 0]) / len(signals) if len(signals) > 0 else 0
    total_pips = signals['result'].sum()
    return win_rate, total_pips

# Streamlit App
st.title("🚀 PipWizard – EUR/USD Forex Signals")
st.sidebar.header("Why Upgrade?")
st.sidebar.info("Free EUR/USD signals. Get Premium for alerts, more pairs, and backtesting! Only $9.99/mo.")

df = fetch_data()
if df.empty:
    st.error("No data fetched. Check API key or market hours.")
else:
    df = generate_signals(df)
    
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Price'
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['sma'], name='SMA(20)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', yaxis='y2', line=dict(color='purple')))
    fig.update_layout(
        yaxis_title='Price', yaxis2_title='RSI', yaxis2_side='right', yaxis2_range=[0,100],
        title=f"EUR/USD Signals (Last {len(df)} mins)",
        height=600, template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Backtest Summary
    win_rate, total_pips = simulate_backtest(df)
    st.subheader("Strategy Performance (Last 24h)")
    st.metric("Win Rate", f"{win_rate:.1%}")
    st.metric("Total Pips", f"{total_pips:+.1f}")
    
    # Recent Signals
    signals = df[df['signal'] != 0].tail(3)
    st.subheader("Latest Signals")
    if not signals.empty:
        for idx, row in signals.iterrows():
            sig_text = "🟢 BUY" if row['signal'] == 1 else "🔴 SELL"
            conf = f"{row['confidence']:.1%} confidence"
            st.write(f"**{sig_text}** at {idx.strftime('%Y-%m-%d %H:%M')} | Price: {row['close']:.5f} | RSI: {row['rsi']:.1f} | {conf}")
    else:
        st.info("No signals yet. RSI in neutral zone (40–60).")
    
    # Call to Action
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Simulate Trade"):
            st.balloons()
            st.success("Simulated: +3.2 pips! Premium unlocks real-time alerts.")
    with col2:
        st.markdown("[Get Premium Now](https://buy.stripe.com/test_123) | [Trade with Alternative](https://example.com)")  # Update link
    
    st.caption("Disclaimer: Forex trading is high-risk. Past performance ≠ future results. Trade responsibly.")
