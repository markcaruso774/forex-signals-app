import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import streamlit.components.v1 as components 
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

# FIXED: fetch_data() — REMOVED SEED + INCREASED VOLATILITY
@st.cache_data(ttl=60)
def fetch_data(symbol, interval):
    """Generates realistic mock data with signals."""
    # np.random.seed(42)  ← REMOVED TO ALLOW REAL VARIATION

    start_date = pd.to_datetime('2024-01-01')
    freq_map = {"1min": 'T', "5min": '5T', "15min": '15T', "30min": '30T', "1h": 'H'}
    
    periods = OUTPUTSIZE
    if interval in ['1min', '5min']: periods = OUTPUTSIZE * 2
    
    timestamps = pd.date_range(start=start_date, periods=periods, freq=freq_map[interval])
    
    base_price = 1.0850
    noise = np.random.randn(periods) * 0.006  # ← INCREASED FROM 0.001 → MORE RSI MOVEMENT
    
    close_array = base_price + np.cumsum(noise)
    close_series = pd.Series(close_array, index=timestamps)
    close = close_series

    open_p = close_series.shift(1).fillna(base_price)
    
    high = np.maximum(open_p, close) + np.abs(np.random.randn(periods) * 0.0005)
    low = np.minimum(open_p, close) - np.abs(np.random.randn(periods) * 0.0005)

    df = pd.DataFrame({'open': open_p, 'high': high, 'low': low, 'close': close}, index=timestamps)
    
    df['high'] = df[['open', 'close']].max(axis=1) + 0.0001
    df['low'] = df[['open', 'close']].min(axis=1) - 0.0001
    
    return df.dropna().tail(OUTPUTSIZE)


# === HELPER FUNCTIONS (Alerts & Calendar) ===

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

# === CUSTOM ECONOMIC CALENDAR FUNCTION (FIXED with Custom Colors) ===
def display_news_calendar():
    st.subheader("Forex Economic Calendar (Today & Tomorrow)")

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
        is_dark = st.session_state.theme == 'dark'
        table_bg_header = '#1f1f1f' if is_dark else '#e9ecef'
        table_color_header = '#ccc' if is_dark else '#212529'
        table_color_text = '#f0f0f0' if is_dark else '#212529'
        table_border_color = '#333' if is_dark else '#dee2e6'

        calendar_html = f"""
        <style>
            .calendar-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
                margin: 10px 0;
                color: {table_color_text};
            }}
            .calendar-table th {{
                background-color: {table_bg_header};
                color: {table_color_header};
                padding: 10px 8px;
                text-align: left;
                font-weight: 600;
            }}
            .calendar-table td {{
                padding: 8px;
                border-bottom: 1px solid {table_border_color};
            }}
            .impact-high {{ color: #FF4136; font-weight: bold; }}
            .impact-medium {{ color: #A0522D; font-weight: bold; }}
            .impact-low {{ color: #A9A9A9; font-weight: bold; }}
        </style>
        <table class="calendar-table">
            <tr>
                <th>Date</th>
                <th>Time (UTC)</th>
                <th>Event</th>
                <th>Impact</th>
            </tr>
        """

        for e in today_events:
            impact_class = {"high": "impact-high", "medium": "impact-medium", "low": "impact-low"}.get(e["impact"], "impact-low")
            display_date = datetime.strptime(e['date'], "%Y-%m-%d").strftime("%b %d")
            calendar_html += f"""
            <tr>
                <td>{display_date}</td>
                <td>{e['time']}</td>
                <td>{e['event']}</td>
                <td><span class='{impact_class}'>• {e['impact'].title()}</span></td>
            </tr>
            """

        calendar_html += "</table>"
        components.html(calendar_html, height=250, scrolling=False) 
    else:
        st.info("No major events scheduled.")


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
    index=3,
    format_func=lambda x: x.replace("min", " minute").replace("1h", "1 hour")
)

# INDICATOR PERIOD CONTROLS
st.sidebar.markdown("---")
st.sidebar.subheader("Indicator Configuration")

show_rsi = st.sidebar.checkbox("Show RSI Chart", value=True)
show_macd = st.sidebar.checkbox("Show MACD Chart", value=True) 

st.sidebar.markdown("**RSI / SMA (Signal)**")
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14, key='rsi_period') 
sma_period = st.sidebar.slider("SMA Period", 10, 50, 20, key='sma_period') 
alert_rsi_low = st.sidebar.slider("Buy RSI <", 20, 40, 35, key='rsi_low') 
alert_rsi_high = st.sidebar.slider("Sell RSI >", 60, 80, 65, key='rsi_high')

if alert_rsi_low >= alert_rsi_high:
    st.sidebar.error("RSI Buy threshold must be lower than Sell threshold.")
    st.stop()

st.sidebar.markdown("**MACD (Confirmation)**")
macd_fast = st.sidebar.slider("MACD Fast Period", 1, 26, 12, key='macd_fast')
macd_slow = st.sidebar.slider("MACD Slow Period", 13, 50, 26, key='macd_slow')
macd_signal = st.sidebar.slider("MACD Signal Period", 1, 15, 9, key='macd_signal')

if macd_fast >= macd_slow:
    st.sidebar.error("MACD Fast Period must be shorter than Slow Period.")
    st.stop()

# === BACKTESTING PARAMETERS ===
st.sidebar.markdown("---")
st.sidebar.subheader("Backtesting Parameters")

initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=10000, key='capital')
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, key='risk_pct') / 100
sl_pips = st.sidebar.number_input("Stop Loss (Pips)", min_value=1, max_value=200, value=50, key='sl_pips')
tp_pips = st.sidebar.number_input("Take Profit (Pips)", min_value=1, max_value=300, value=75, key='tp_pips')

if sl_pips <= 0 or tp_pips <= 0:
    st.sidebar.error("SL and TP must be greater than 0 pips.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.info("Premium ($9.99/mo):\n• 10+ Pairs\n• Real-time Alerts\n• Vectorized Backtesting")


# === FETCH DATA & CALCULATE INDICATORS ===
df = fetch_data(selected_pair, selected_interval)

if df.empty:
    st.error("No data. Check connection or API key.")
    st.stop()

def calculate_indicators(df):
    df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)
    df['sma'] = df['close'].rolling(sma_period).mean()
    df['macd_line'], df['macd_signal'], df['macd_hist'] = talib.MACD(
        df['close'], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal
    )
    return df

df = calculate_indicators(df)

# Signal Logic
df['signal'] = 0
df.loc[(df['rsi'] < alert_rsi_low) & (df['close'] > df['sma']), 'signal'] = 1
df.loc[(df['rsi'] > alert_rsi_high) & (df['close'] < df['sma']), 'signal'] = -1

df = df.dropna()
if df.empty:
    st.warning("Waiting for sufficient data after indicator calculation...")
    st.stop()


# === VECTORIZED BACKTESTING FUNCTION ===
def run_backtest(df_in, initial_capital, risk_per_trade, sl_pips, tp_pips):
    df = df_in.copy()
    PIP_MULTIPLIER = 0.0001
    RISK_PIPS_VALUE = sl_pips * PIP_MULTIPLIER
    REWARD_PIPS_VALUE = tp_pips * PIP_MULTIPLIER
    MAX_RISK_USD = initial_capital * risk_per_trade
    REWARD_USD = MAX_RISK_USD * (tp_pips / sl_pips)
    
    df['entry_signal'] = df['signal'].shift(1).fillna(0)
    trade_df = df[df['entry_signal'] != 0].copy()
    if trade_df.empty:
        return 0, 0, 0, 0, initial_capital, pd.DataFrame() 

    trade_df['entry_price'] = trade_df['open']
    trade_df.loc[trade_df['entry_signal'] == 1, 'stop_loss'] = trade_df['entry_price'] - RISK_PIPS_VALUE
    trade_df.loc[trade_df['entry_signal'] == 1, 'take_profit'] = trade_df['entry_price'] + REWARD_PIPS_VALUE
    trade_df.loc[trade_df['entry_signal'] == -1, 'stop_loss'] = trade_df['entry_price'] + RISK_PIPS_VALUE
    trade_df.loc[trade_df['entry_signal'] == -1, 'take_profit'] = trade_df['entry_price'] - REWARD_PIPS_VALUE
    
    trade_df['result'] = 'NEUTRAL'
    trade_df['profit_loss'] = 0.0

    buy_trades = trade_df['entry_signal'] == 1
    buy_win = (trade_df['high'] >= trade_df['take_profit']) & (trade_df['low'] > trade_df['stop_loss'])
    trade_df.loc[buy_trades & buy_win, 'result'] = 'WIN'
    trade_df.loc[buy_trades & buy_win, 'profit_loss'] = REWARD_USD
    buy_loss = (trade_df['low'] <= trade_df['stop_loss']) & (trade_df['high'] < trade_df['take_profit'])
    trade_df.loc[buy_trades & buy_loss, 'result'] = 'LOSS'
    trade_df.loc[buy_trades & buy_loss, 'profit_loss'] = -MAX_RISK_USD
    buy_amb = (trade_df['high'] >= trade_df['take_profit']) & (trade_df['low'] <= trade_df['stop_loss'])
    trade_df.loc[buy_trades & buy_amb, 'result'] = 'LOSS'
    trade_df.loc[buy_trades & buy_amb, 'profit_loss'] = -MAX_RISK_USD

    sell_trades = trade_df['entry_signal'] == -1
    sell_win = (trade_df['low'] <= trade_df['take_profit']) & (trade_df['high'] < trade_df['stop_loss'])
    trade_df.loc[sell_trades & sell_win, 'result'] = 'WIN'
    trade_df.loc[sell_trades & sell_win, 'profit_loss'] = REWARD_USD
    sell_loss = (trade_df['high'] >= trade_df['stop_loss']) & (trade_df['low'] > trade_df['take_profit'])
    trade_df.loc[sell_trades & sell_loss, 'result'] = 'LOSS'
    trade_df.loc[sell_trades & sell_loss, 'profit_loss'] = -MAX_RISK_USD
    sell_amb = (trade_df['low'] <= trade_df['take_profit']) & (trade_df['high'] >= trade_df['stop_loss'])
    trade_df.loc[sell_trades & sell_amb, 'result'] = 'LOSS'
    trade_df.loc[sell_trades & sell_amb, 'profit_loss'] = -MAX_RISK_USD

    final_trades = trade_df[trade_df['result'] != 'NEUTRAL'].copy()
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

    st.markdown("---")
    st.subheader("Backtesting Results (Simulated)")
    st.markdown(f"***Data Tested:*** *{selected_pair}* on *{selected_interval}* interval. *{len(df)}* bars.")
    
    col_t, col_w, col_p, col_f = st.columns(4)
    col_t.metric("Total Trades", total_trades)
    col_w.metric("Win Rate", f"{win_rate:.2%}")
    col_p.metric("Total Profit ($)", f"{total_profit:,.2f}", delta=f"{total_profit/initial_capital:.2%}")
    col_f.metric("Profit Factor", f"{profit_factor:,.2f}")
    
    st.subheader("Equity Curve")
    if not trade_df.empty:
        trade_df['capital_change'] = trade_df['profit_loss']
        trade_df['equity'] = initial_capital + trade_df['capital_change'].cumsum()
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(x=trade_df.index, y=trade_df['equity'], mode='lines', name='Equity', line=dict(color='#26a69a')))
        equity_fig.update_layout(xaxis_title="Time", yaxis_title="Account Equity ($)", template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white')
        st.plotly_chart(equity_fig, use_container_width=True)
    else:
        st.info("No resolved trades. Adjust RSI/SMA or volatility.")

    st.subheader("Detailed Trade Log")
    st.dataframe(trade_df, use_container_width=True)

# --- NEWS CALENDAR SECTION ---
st.markdown("---")
display_news_calendar()
st.markdown("---")


# === CHART & LIVE SIGNAL CHECK ===
num_rows = 1
row_heights = [0.7]
if show_rsi: num_rows += 1; row_heights.append(0.15)
if show_macd: num_rows += 1; row_heights.append(0.15)

rsi_row = macd_row = 0
current_row = 2
if show_rsi: rsi_row = current_row; current_row += 1
if show_macd: macd_row = current_row

st.subheader(f"**{selected_pair}** – **{selected_interval}** – Last {len(df)} Candles")
chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")

fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights)

buy_signals = df[df['signal'] == 1]
sell_signals = df[df['signal'] == -1]

if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
else:
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name="Price", line=dict(color="#2196f3")), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['sma'], name=f"SMA({sma_period})", line=dict(color="#ff9800")), row=1, col=1)
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['low'] * 0.9995, mode='markers', marker=dict(symbol='triangle-up', size=10, color='#26a69a'), name='Buy Signal'), row=1, col=1)
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['high'] * 1.0005, mode='markers', marker=dict(symbol='triangle-down', size=10, color='#ef5350'), name='Sell Signal'), row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)

if show_rsi:
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name=f"RSI({rsi_period})", line=dict(color="#9c27b0")), row=rsi_row, col=1)
    fig.add_hline(y=alert_rsi_high, line_dash="dash", line_color="#ef5350", annotation_text=f"Overbought ({alert_rsi_high})", row=rsi_row, col=1)
    fig.add_hline(y=alert_rsi_low, line_dash="dash", line_color="#26a69a", annotation_text=f"Oversold ({alert_rsi_low})", row=rsi_row, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#cccccc", row=rsi_row, col=1)
    fig.update_yaxes(title_text=f"RSI({rsi_period})", range=[0, 100], row=rsi_row, col=1)

if show_macd:
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_line'], name='MACD', line=dict(color='#2196f3')), row=macd_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='#ff9800')), row=macd_row, col=1)
    colors = ['#26a69a' if val >= 0 else '#ef5350' for val in df['macd_hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='Histogram', marker_color=colors), row=macd_row, col=1)
    fig.update_yaxes(title_text="MACD", row=macd_row, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#cccccc", row=macd_row, col=1)

fig.update_layout(height=600, template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white',
                  xaxis=dict(rangeslider=dict(visible=False)), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# === LIVE SIGNAL ALERT CHECK ===
check_for_live_signal(df, selected_pair)
