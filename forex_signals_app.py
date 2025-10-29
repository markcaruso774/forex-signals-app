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
st.set_page_config(page_title="PipWizard", page_icon="💹", layout="wide")

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
    st.sidebar.markdown(f"**🚨 ALERT SENT**")
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

st.sidebar.markdown("**RSI / SMA (Signal)**")
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14) 
sma_period = st.sidebar.slider("SMA Period", 10, 50, 20) 
alert_rsi_low = st.sidebar.slider("Buy RSI <", 20, 40, 30)
alert_rsi_high = st.sidebar.slider("Sell RSI >", 60, 80, 70)

# MACD CONTROLS
st.sidebar.markdown("**MACD (Confirmation)**")
macd_fast = st.sidebar.slider("MACD Fast Period", 1, 26, 12)
macd_slow = st.sidebar.slider("MACD Slow Period", 13, 50, 26)
macd_signal = st.sidebar.slider("MACD Signal Period", 1, 15, 9)


# === BACKTESTING PARAMETERS ===
st.sidebar.markdown("---")
st.sidebar.subheader("Backtesting Parameters")

initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=10000)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0) / 100

sl_pips = st.sidebar.number_input("Stop Loss (Pips)", min_value=10, max_value=200, value=50)
tp_pips = st.sidebar.number_input("Take Profit (Pips)", min_value=10, max_value=300, value=75)

st.sidebar.markdown("---")
st.sidebar.info("Premium ($9.99/mo):\n• 10+ Pairs\n• Real-time Alerts\n• Vectorized Backtesting")


# === FETCH DATA & CALCULATE INDICATORS ===
df = fetch_data(selected_pair, selected_interval)

if df.empty:
    st.error("No data. Check connection or API key.")
    st.stop()

def calculate_indicators(df):
    # RSI & SMA (Signal Indicators)
    df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)
    df['sma'] = df['close'].rolling(sma_period).mean()
    
    # MACD (New Confirmation Indicator)
    df['macd_line'], df['macd_signal'], df['macd_hist'] = talib.MACD(
        df['close'], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal
    )
    return df

df = calculate_indicators(df)

# Signal Logic
df['signal'] = 0
# BUY signal: RSI is oversold AND Price is above SMA (Uptrend)
df.loc[(df['rsi'] < alert_rsi_low) & (df['close'] > df['sma']), 'signal'] = 1
# SELL signal: RSI is overbought AND Price is below SMA (Downtrend)
df.loc[(df['rsi'] > alert_rsi_high) & (df['close'] < df['sma']), 'signal'] = -1

df = df.dropna()
if df.empty:
    st.warning("Waiting for sufficient data...")
    st.stop()


# === VECTORIZED BACKTESTING FUNCTION ===
def run_backtest(df_in, initial_capital, risk_per_trade, sl_pips, tp_pips):
    """
    Vectorized trade simulation based on 'signal' column and adjustable SL/TP pips.
    Uses numpy.where for fast conditional execution.
    """
    df = df_in.copy()
    
    # 1. PARAMETER CALCULATION
    PIP_MULTIPLIER = 0.0001
    RISK_PIPS_VALUE = sl_pips * PIP_MULTIPLIER
    REWARD_PIPS_VALUE = tp_pips * PIP_MULTIPLIER
    
    MAX_RISK_USD = initial_capital * risk_per_trade
    REWARD_USD = MAX_RISK_USD * (tp_pips / sl_pips)
    
    # Shift signals backward to align them with the entry bar (i+1)
    df['entry_signal'] = df['signal'].shift(1).fillna(0)
    
    # Filter only rows that have an entry signal
    trade_df = df[df['entry_signal'] != 0].copy()
    
    if trade_df.empty:
        return 0, 0, 0, 0, initial_capital, pd.DataFrame() 

    # 2. CALCULATE TP/SL PRICES (Vectorized)
    trade_df['entry_price'] = trade_df['open']

    # BUY Trades: SL = Entry - Risk, TP = Entry + Reward
    trade_df.loc[trade_df['entry_signal'] == 1, 'stop_loss'] = trade_df['entry_price'] - RISK_PIPS_VALUE
    trade_df.loc[trade_df['entry_signal'] == 1, 'take_profit'] = trade_df['entry_price'] + REWARD_PIPS_VALUE

    # SELL Trades: SL = Entry + Risk, TP = Entry - Reward
    trade_df.loc[trade_df['entry_signal'] == -1, 'stop_loss'] = trade_df['entry_price'] + RISK_PIPS_VALUE
    trade_df.loc[trade_df['entry_signal'] == -1, 'take_profit'] = trade_df['entry_price'] - REWARD_PIPS_VALUE
    
    # 3. DETERMINE TRADE OUTCOME (Vectorized)
    
    trade_df['result'] = 'NEUTRAL'
    trade_df['profit_loss'] = 0.0

    # --- BUY LOGIC (Signal = 1) ---
    buy_trades = trade_df['entry_signal'] == 1
    
    # WIN: High hits TP AND Low stays above SL
    buy_win_condition = (trade_df['high'] >= trade_df['take_profit']) & (trade_df['low'] > trade_df['stop_loss'])
    trade_df.loc[buy_trades & buy_win_condition, 'result'] = 'WIN'
    trade_df.loc[buy_trades & buy_win_condition, 'profit_loss'] = REWARD_USD

    # LOSS: Low hits SL AND High stays below TP
    buy_loss_condition = (trade_df['low'] <= trade_df['stop_loss']) & (trade_df['high'] < trade_df['take_profit'])
    trade_df.loc[buy_trades & buy_loss_condition, 'result'] = 'LOSS'
    trade_df.loc[buy_trades & buy_loss_condition, 'profit_loss'] = -MAX_RISK_USD
    
    # Ambiguous: Both hit (Conservative loss assumption)
    buy_ambiguous_condition = (trade_df['high'] >= trade_df['take_profit']) & (trade_df['low'] <= trade_df['stop_loss'])
    trade_df.loc[buy_trades & buy_ambiguous_condition, 'result'] = 'LOSS'
    trade_df.loc[buy_trades & buy_ambiguous_condition, 'profit_loss'] = -MAX_RISK_USD


    # --- SELL LOGIC (Signal = -1) ---
    sell_trades = trade_df['entry_signal'] == -1
    
    # WIN: Low hits TP AND High stays below SL
    sell_win_condition = (trade_df['low'] <= trade_df['take_profit']) & (trade_df['high'] < trade_df['stop_loss'])
    trade_df.loc[sell_trades & sell_win_condition, 'result'] = 'WIN'
    trade_df.loc[sell_trades & sell_win_condition, 'profit_loss'] = REWARD_USD

    # LOSS: High hits SL AND Low stays above TP
    sell_loss_condition = (trade_df['high'] >= trade_df['stop_loss']) & (trade_df['low'] > trade_df['take_profit'])
    trade_df.loc[sell_trades & sell_loss_condition, 'result'] = 'LOSS'
    trade_df.loc[sell_trades & sell_loss_condition, 'profit_loss'] = -MAX_RISK_USD

    # Ambiguous: Both hit (Conservative loss assumption)
    sell_ambiguous_condition = (trade_df['low'] <= trade_df['take_profit']) & (trade_df['high'] >= trade_df['stop_loss'])
    trade_df.loc[sell_trades & sell_ambiguous_condition, 'result'] = 'LOSS'
    trade_df.loc[sell_trades & sell_ambiguous_condition, 'profit_loss'] = -MAX_RISK_USD

    # Filter out trades that were not resolved (NEUTRAL)
    final_trades = trade_df[trade_df['result'] != 'NEUTRAL'].copy()

    # 4. METRICS CALCULATION
    total_trades = len(final_trades)
    
    if total_trades == 0:
        return 0, 0, 0, 0, initial_capital, pd.DataFrame()

    winning_trades = len(final_trades[final_trades['result'] == 'WIN'])
    total_profit = final_trades['profit_loss'].sum()
    
    win_rate = winning_trades / total_trades
    
    gross_win = final_trades[final_trades['profit_loss'] > 0]['profit_loss'].sum()
    gross_loss = abs(final_trades[final_trades['profit_loss'] < 0]['profit_loss'].sum())
    
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 999.0
    final_capital = initial_capital + total_profit
    
    final_trades['signal'] = final_trades['entry_signal'].apply(lambda x: 'BUY' if x == 1 else 'SELL')

    return total_trades, win_rate, total_profit, profit_factor, final_capital, final_trades[['signal', 'entry_price', 'result', 'profit_loss']]


# === RUN BACKTESTING IF PREMIUM ===
if is_premium:
    total_trades, win_rate, total_profit, profit_factor, final_capital, trade_df = run_backtest(
        df, initial_capital, risk_pct, sl_pips, tp_pips
    )

    # --- BACKTESTING RESULTS DISPLAY ---
    st.markdown("---")
    st.subheader("📊 Backtesting Results (Simulated)")
    st.markdown(f"***Data Tested:*** *{selected_pair}* on *{selected_interval}* interval. *{len(df)}* bars.")
    
    col_t, col_w, col_p, col_f = st.columns(4)

    col_t.metric("Total Trades", total_trades)
    col_w.metric("Win Rate", f"{win_rate:.2%}")
    col_p.metric("Total Profit ($)", f"{total_profit:,.2f}", delta=f"{total_profit/initial_capital:.2%}")
    col_f.metric("Profit Factor", f"{profit_factor:,.2f}")
    
    # Equity Curve
    st.subheader("Equity Curve")
    if not trade_df.empty:
        trade_df['capital_change'] = trade_df['profit_loss']
        trade_df['equity'] = initial_capital + trade_df['capital_change'].cumsum()
        
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(x=trade_df.index, y=trade_df['equity'], 
                                        mode='lines', name='Equity', line=dict(color='#26a69a')))
        equity_fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Account Equity ($)",
            template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
        )
        st.plotly_chart(equity_fig, use_container_width=True)
    else:
        st.info("Not enough data or no signals generated for backtesting.")

    st.subheader("Detailed Trade Log")
    st.dataframe(trade_df, use_container_width=True)


# === CHART & LIVE SIGNAL CHECK ===

# 1. Determine the number of chart rows based on UX toggles
num_rows = 1
row_heights = [0.7]

if show_rsi:
    num_rows += 1
    row_heights.append(0.15)
if show_macd:
    num_rows += 1
    row_heights.append(0.15)

# Calculate which row is which indicator
rsi_row = 0
macd_row = 0
current_row = 2
if show_rsi:
    rsi_row = current_row
    current_row += 1
if show_macd:
    macd_row = current_row
    current_row += 1

# 2. Setup Subplots
st.subheader(f"**{selected_pair}** – **{selected_interval}** – Last {len(df)} Candles")
chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")

fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights)

# Filter data for signals
buy_signals = df[df['signal'] == 1]
sell_signals = df[df['signal'] == -1]

# --- PRICE CHART (ROW 1) ---
if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
else:
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name="Price", line=dict(color="#2196f3")), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['sma'], name=f"SMA({sma_period})", line=dict(color="#ff9800")), row=1, col=1)

# Buy/Sell Signal Markers
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['low'] * 0.9995, mode='markers', marker=dict(symbol='triangle-up', size=10, color='#26a69a'), name='Buy Signal'), row=1, col=1)
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['high'] * 1.0005, mode='markers', marker=dict(symbol='triangle-down', size=10, color='#ef5350'), name='Sell Signal'), row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)

# --- RSI CHART (Conditional Row) ---
if show_rsi:
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name=f"RSI({rsi_period})", line=dict(color="#9c27b0")), row=rsi_row, col=1)
    fig.add_hline(y=alert_rsi_high, line_dash="dash", line_color="#ef5350", annotation_text=f"Overbought ({alert_rsi_high})", row=rsi_row, col=1)
    fig.add_hline(y=alert_rsi_low, line_dash="dash", line_color="#26a69a", annotation_text=f"Oversold ({alert_rsi_low})", row=rsi_row, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#cccccc", row=rsi_row, col=1)
    fig.update_yaxes(title_text=f"RSI({rsi_period})", range=[0, 100], row=rsi_row, col=1)

# --- MACD CHART (Conditional Row) ---
if show_macd:
    # MACD Histogram
    hist_colors = np.where(df['macd_hist'] >= 0, 'rgba(38, 166, 154, 0.7)', 'rgba(239, 83, 80, 0.7)')
    fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name="Hist", marker_color=hist_colors), row=macd_row, col=1)
    # MACD Line
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_line'], name="MACD", line=dict(color='#00bfa5')), row=macd_row, col=1)
    # Signal Line
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name="Signal", line=dict(color='#ff9800')), row=macd_row, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#cccccc", row=macd_row, col=1)
    fig.update_yaxes(title_text=f"MACD", row=macd_row, col=1)


# Update layout
fig.update_layout(height=800, xaxis_rangeslider_visible=False, template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

# Hide x-axis labels for all but the bottom subplot
for i in range(1, num_rows):
    fig.update_xaxes(showticklabels=False, row=i, col=1)


st.plotly_chart(fig, use_container_width=True)

# === LATEST SIGNAL SUMMARY & ALERT CHECK ===
st.markdown("---")
st.subheader("💡 Latest Signal Summary")

# Trigger the alert check
if is_premium:
    check_for_live_signal(df, selected_pair)
    st.markdown("---")

latest_data = df.iloc[-1]
latest_signal = latest_data['signal']
latest_price = latest_data['close']
latest_rsi = latest_data['rsi']
latest_sma = latest_data['sma']
latest_macd = latest_data['macd_line']
latest_macd_signal = latest_data['macd_signal']

if latest_signal == 1:
    signal_text = f"**BUY Signal!** The {selected_pair} is **Oversold** (RSI: {latest_rsi:.2f} < {alert_rsi_low}) and is trading **Above** the SMA({sma_period})."
    signal_style = "buy-signal"
elif latest_signal == -1:
    signal_text = f"**SELL Signal!** The {selected_pair} is **Overbought** (RSI: {latest_rsi:.2f} > {alert_rsi_high}) and is trading **Below** the SMA({sma_period})."
    signal_style = "sell-signal"
else:
    signal_text = "Neutral: Waiting for a clear RSI/SMA crossover signal."
    signal_style = ""

st.markdown(f"""
<div style="padding: 10px; border: 1px solid {'#f0f0f0' if st.session_state.theme == 'dark' else '#212529'}; border-radius: 5px;">
    <p class='{signal_style}'>
        {signal_text}
    </p>
    <ul>
        <li>**Current Price:** {latest_price:.5f}</li>
        <li>**Current RSI({rsi_period}):** {latest_rsi:.2f}</li>
        <li>**Current SMA({sma_period}):** {latest_sma:.5f}</li>
        <li>**MACD Line:** {latest_macd:.5f} | **MACD Signal:** {latest_macd_signal:.5f}</li>
        <li>**Timestamp:** {latest_data.name.strftime('%Y-%m-%d %H:%M:%S')}</li>
    </ul>
</div>
""", unsafe_allow_html=True)
