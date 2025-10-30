import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import streamlit.components.v1 as components
import talib
from twelvedata import TDClient

# === CONFIG ===
# !!! IMPORTANT: Replace with your REAL Twelve Data API Key
TD_API_KEY = "e02de9a60165478aaf1da8a7b2096e05" # This is a mock key

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
    "30min": "30min", "1h": "1h"
}
OUTPUTSIZE = 500 # Max 5000 with paid plan, 800 is a good limit

# === PAGE CONFIG ===
st.set_page_config(page_title="PipWizard", page_icon="💹", layout="wide")

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
    theme_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
    if st.button(theme_label, key="theme_toggle", on_click=toggle_theme):
        st.rerun()

# === NEW: ABOUT THE APP SECTION ===
with st.expander("👋 Welcome to PipWizard! Click here to learn about the app."):
    st.markdown(
        """
        ### What is PipWizard?

        PipWizard is a powerful decision-support tool for forex traders. It's designed to help you **find**, **test**, and **act on** trading strategies in real-time.

        It combines a live signal generator, economic news calendar, and a powerful, on-demand backtesting engine.

        ---

        ### How to Use the App

        There is a simple 3-step workflow:

        1.  **Step 1: TEST A STRATEGY (The "Main Backtest")**
            * Use the sidebar to pick a strategy (`RSI Standalone`, `MACD Crossover`, etc.) and set your `Stop Loss` and `Take Profit`.
            * Click the **"Run Backtest"** button.
            * The app will instantly show you a full report, including an **Equity Curve** and **Detailed Trade Log**, so you can see if your settings were profitable on historical data.

        2.  **Step 2: FIND THE BEST STRATEGY (Premium Feature)**
            * Open the **"🚀 Strategy Scanner"** at the bottom of the page.
            * This "heatmap" tool tests *all* strategies across *all* pairs and timeframes at once.
            * It shows you a color-coded table of what is *actually* working in the market right now.

        3.  **Step 3: ACTIVATE LIVE SIGNALS (Premium Feature)**
            * Once you've found a profitable setup (e.g., "RSI Standalone on 1h for EUR/USD"), set those parameters in the sidebar.
            * The app will now run in "live" mode, automatically refreshing and showing you `BUY` (▲) or `SELL` (▼) signals on the chart as they happen.
            * Premium users will also receive an "ALERT SENT" in the sidebar for these signals.

        ---

        ### Feature Tiers: Free vs. Premium

        **🎁 Free Tier (What you have now):**
        * ✅ **Economic News Calendar:** See high-impact news events to plan your trades.
        * ✅ **Full Backtesting Engine:** The main **"Run Backtest"** feature is 100% free.
        * ✅ **All 6 Strategies:** Test all strategies (RSI, MACD, Confluence, etc.).
        * ✅ **All Timeframes:** Test your strategies on any timeframe (1min to 1h).
        * 🔒 **Limited to EUR/USD:** All backtesting and live signals are limited to the **EUR/USD** pair.

        **⭐ Premium Tier (The "All-in-One" Upgrade):**
        * ✅ **Unlock All 10+ Currency Pairs:** Unlock all pairs (GBP/USD, USD/JPY, etc.) for backtesting and live signals.
        * ✅ **🚀 Strategy Scanner:** Get full access to the powerful "heatmap" scanner.
        * ✅ **Live Signal Alerts:** Receive the "ALERT SENT" notifications in your sidebar.
        * *(Includes all Free Tier features, of course!)*

        You can upgrade at any time by clicking the links in the sidebar!
        """
    )
# === END ABOUT SECTION ===


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
    st.sidebar.info("Upgrade to Premium to unlock all pairs and the Strategy Scanner!")

# TIMEFRAME SELECTOR
selected_interval = st.sidebar.selectbox(
    "Timeframe",
    options=list(INTERVALS.keys()),
    index=3,
    format_func=lambda x: x.replace("min", " minute").replace("1h", "1 hour")
)

# === STRATEGY SELECTION ===
st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Selection")
strategy_name = st.sidebar.selectbox(
    "Choose a Strategy",
    [
        "RSI + SMA Crossover",      # Original
        "MACD Crossover",           # Original
        "RSI + MACD (Confluence)",  # New
        "SMA + MACD (Confluence)",  # New
        "RSI Standalone",           # New
        "SMA Crossover Standalone"  # New
    ]
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

# --- Backtest Buttons (Available to ALL users) ---
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
run_backtest_button = col1.button("Run Backtest", type="primary", use_container_width=True)

if 'backtest_results' in st.session_state:
    if col2.button("Clear Results", use_container_width=True):
        del st.session_state.backtest_results
        st.rerun()
# --- End of Buttons ---


st.sidebar.markdown("---")
st.sidebar.info(
    """
    **🎁 Free Tier:**\n
    Full backtesting on EUR/USD only.

    **⭐ Upgrade to Premium:**\n
    • Unlock all pairs\n
    • Unlock Strategy Scanner\n
    • Get Live Signal Alerts
    """
)


# === HELPER FUNCTIONS (Alerts & Calendar) ===
@st.cache_data(ttl=60)
def fetch_data(symbol, interval):
    """Fetches real market data from Twelve Data."""
    td = TDClient(apikey=TD_API_KEY)
    try:
        ts = td.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=OUTPUTSIZE
        ).as_pandas()
        
        if ts is None or ts.empty:
            st.error(f"No data returned for {symbol}. Check API key or symbol.")
            return pd.DataFrame()
        
        df = ts[['open', 'high', 'low', 'close']].copy()
        df.index = pd.to_datetime(df.index)
        return df.iloc[::-1]  # Reverse to chronological order (oldest first)
        
    except Exception as e:
        st.error(f"API Error fetching {symbol}: {str(e)}")
        if "API key" in str(e):
             st.error("Please ensure your Twelve Data API key is correct.")
        return pd.DataFrame()

def send_alert_email(signal_type, price, pair):
    st.sidebar.markdown(f"**ALERT SENT**")
    st.sidebar.warning(f"**{signal_type.upper()}** on {pair} at {price:.5f}")

def check_for_live_signal(df, pair):
    if len(df) < 2:
        return
    latest_bar = df.iloc[-2] # Signal bar
    current_bar = df.iloc[-1] # Action bar
    signal = latest_bar['signal']
    price = current_bar['open'] # Alert at entry price
    
    if 'last_alert_time' not in st.session_state:
        st.session_state.last_alert_time = None

    if signal != 0 and latest_bar.name != st.session_state.last_alert_time:
        st.session_state.last_alert_time = latest_bar.name
        if signal == 1:
            send_alert_email("BUY", price, pair)
        elif signal == -1:
            send_alert_email("SELL", price, pair)

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
        calendar_html = f"""<style>.calendar-table{{width:100%;border-collapse:collapse;font-size:14px;margin:10px 0;color:{table_color_text};}}.calendar-table th{{background-color:{table_bg_header};color:{table_color_header};padding:10px 8px;text-align:left;font-weight:600;}}.calendar-table td{{padding:8px;border-bottom:1px solid {table_border_color};}}.impact-high{{color:#FF4136;font-weight:bold;}}.impact-medium{{color:#A0522D;font-weight:bold;}}.impact-low{{color:#A9A9A9;font-weight:bold;}}</style><table class="calendar-table"><tr><th>Date</th><th>Time (UTC)</th><th>Event</th><th>Impact</th></tr>"""
        for e in today_events:
            impact_class = {"high": "impact-high", "medium": "impact-medium", "low": "impact-low"}.get(e["impact"], "impact-low")
            display_date = datetime.strptime(e['date'], "%Y-%m-%d").strftime("%b %d")
            calendar_html += f"""<tr><td>{display_date}</td><td>{e['time']}</td><td>{e['event']}</td><td><span class='{impact_class}'>• {e['impact'].title()}</span></td></tr>"""
        calendar_html += "</table>"
        components.html(calendar_html, height=250, scrolling=False)
    else:
        st.info("No major events scheduled.")

# === INDICATOR & STRATEGY LOGIC (ACCEPTING PARAMS) ===
def calculate_indicators(df, rsi_p, sma_p, macd_f, macd_sl, macd_sig):
    """Calculates all indicators based on params."""
    df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_p)
    df['sma'] = df['close'].rolling(sma_p).mean()
    df['macd_line'], df['macd_signal'], df['macd_hist'] = talib.MACD(
        df['close'], fastperiod=macd_f, slowperiod=macd_sl, signalperiod=macd_sig
    )
    return df

def apply_strategy(df, strategy_name, rsi_l, rsi_h):
    """Applies the selected trading strategy logic."""
    df['signal'] = 0
    if strategy_name == "RSI + SMA Crossover":
        df.loc[(df['rsi'] < rsi_l) & (df['close'] > df['sma']), 'signal'] = 1
        df.loc[(df['rsi'] > rsi_h) & (df['close'] < df['sma']), 'signal'] = -1
    elif strategy_name == "MACD Crossover":
        buy_cond = (df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
        sell_cond = (df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
        df.loc[buy_cond, 'signal'] = 1
        df.loc[sell_cond, 'signal'] = -1
    elif strategy_name == "RSI + MACD (Confluence)":
        buy_cond_1 = (df['rsi'] < rsi_l)
        buy_cond_2 = (df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
        df.loc[buy_cond_1 & buy_cond_2, 'signal'] = 1
        sell_cond_1 = (df['rsi'] > rsi_h)
        sell_cond_2 = (df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
        df.loc[sell_cond_1 & sell_cond_2, 'signal'] = -1
    elif strategy_name == "SMA + MACD (Confluence)":
        buy_cond_1 = (df['close'] > df['sma'])
        buy_cond_2 = (df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
        df.loc[buy_cond_1 & buy_cond_2, 'signal'] = 1
        sell_cond_1 = (df['close'] < df['sma'])
        sell_cond_2 = (df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
        df.loc[sell_cond_1 & sell_cond_2, 'signal'] = -1
    elif strategy_name == "RSI Standalone":
        buy_cond = (df['rsi'] < rsi_l) & (df['rsi'].shift(1) >= rsi_l)
        sell_cond = (df['rsi'] > rsi_h) & (df['rsi'].shift(1) <= rsi_h)
        df.loc[buy_cond, 'signal'] = 1
        df.loc[sell_cond, 'signal'] = -1
    elif strategy_name == "SMA Crossover Standalone":
        buy_cond = (df['close'] > df['sma']) & (df['close'].shift(1) <= df['sma'].shift(1))
        sell_cond = (df['close'] < df['sma']) & (df['close'].shift(1) >= df['sma'].shift(1))
        df.loc[buy_cond, 'signal'] = 1
        df.loc[sell_cond, 'signal'] = -1
    return df

# === (FIXED) BACKTESTING FUNCTION ===
def run_backtest(df_in, pair_name, initial_capital, risk_per_trade, sl_pips, tp_pips):
    """Runs a bar-by-bar backtest simulation."""
    df = df_in.copy()
    trades = []
    
    if "JPY" in pair_name:
         PIP_MULTIPLIER = 0.01
    else:
         PIP_MULTIPLIER = 0.0001

    RISK_PIPS_VALUE = sl_pips * PIP_MULTIPLIER
    REWARD_PIPS_VALUE = tp_pips * PIP_MULTIPLIER
    MAX_RISK_USD = initial_capital * risk_per_trade
    REWARD_USD = MAX_RISK_USD * (tp_pips / sl_pips)

    signal_bars = df[df['signal'] != 0]

    for i in range(len(signal_bars)):
        signal_row = signal_bars.iloc[i]
        signal_type = signal_row['signal']
        try:
            signal_index = df.index.get_loc(signal_row.name)
        except KeyError:
            continue
        if signal_index + 1 >= len(df):
            continue
        entry_bar = df.iloc[signal_index + 1]
        entry_price = entry_bar['open']
        entry_time = entry_bar.name
        stop_loss, take_profit = 0.0, 0.0
        if signal_type == 1: # Buy
            stop_loss = entry_price - RISK_PIPS_VALUE
            take_profit = entry_price + REWARD_PIPS_VALUE
        else: # Sell
            stop_loss = entry_price + RISK_PIPS_VALUE
            take_profit = entry_price - REWARD_PIPS_VALUE
        result = 'OPEN'
        profit_loss = 0.0
        exit_time = None
        for j in range(signal_index + 2, len(df)):
            future_bar = df.iloc[j]
            if signal_type == 1: # Buy trade
                if future_bar['low'] <= stop_loss:
                    result = 'LOSS'
                    profit_loss = -MAX_RISK_USD
                    exit_time = future_bar.name
                    break
                elif future_bar['high'] >= take_profit:
                    result = 'WIN'
                    profit_loss = REWARD_USD
                    exit_time = future_bar.name
                    break
            elif signal_type == -1: # Sell trade
                if future_bar['high'] >= stop_loss:
                    result = 'LOSS'
                    profit_loss = -MAX_RISK_USD
                    exit_time = future_bar.name
                    break
                elif future_bar['low'] <= take_profit:
                    result = 'WIN'
                    profit_loss = REWARD_USD
                    exit_time = future_bar.name
                    break
        if result == 'OPEN':
            result = 'UNRESOLVED'
            profit_loss = 0.0
            exit_time = df.iloc[-1].name
        trades.append({"entry_time": entry_time, "exit_time": exit_time, "signal": "BUY" if signal_type == 1 else "SELL", "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit, "result": result, "profit_loss": profit_loss})
    if not trades:
        return 0, 0, 0, 0, initial_capital, pd.DataFrame(), pd.DataFrame() 
    trade_log = pd.DataFrame(trades).set_index('entry_time')
    resolved_trades = trade_log[trade_log['result'].isin(['WIN', 'LOSS'])].copy()
    if resolved_trades.empty:
        return 0, 0, 0, 0, initial_capital, trade_log, resolved_trades
    total_trades = len(resolved_trades)
    winning_trades = len(resolved_trades[resolved_trades['result'] == 'WIN'])
    total_profit = resolved_trades['profit_loss'].sum()
    win_rate = winning_trades / total_trades
    gross_win = resolved_trades[resolved_trades['profit_loss'] > 0]['profit_loss'].sum()
    gross_loss = abs(resolved_trades[resolved_trades['profit_loss'] < 0]['profit_loss'].sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 999.0
    final_capital = initial_capital + total_profit
    resolved_trades['equity'] = initial_capital + resolved_trades['profit_loss'].cumsum()
    return total_trades, win_rate, total_profit, profit_factor, final_capital, trade_log, resolved_trades

# === DATA LOADING & MAIN CHART LOGIC ===
with st.spinner(f"Fetching {OUTPUTSIZE} candles for {selected_pair} ({selected_interval})..."):
    df = fetch_data(selected_pair, INTERVALS[selected_interval])
if df.empty:
    st.error("Failed to load data. The API might be down or your key is invalid.")
    st.stop()
with st.spinner("Calculating indicators..."):
    df = calculate_indicators(df, rsi_period, sma_period, macd_fast, macd_slow, macd_signal)
with st.spinner(f"Applying Strategy: {strategy_name}..."):
    df = apply_strategy(df, strategy_name, alert_rsi_low, alert_rsi_high)
df = df.dropna()
if df.empty:
    st.warning("Waiting for sufficient data after indicator calculation...")
    st.stop()

# === RUN MAIN BACKTESTING ON BUTTON CLICK ===
if run_backtest_button:
    with st.spinner("Running backtest on real market data..."):
        total_trades, win_rate, total_profit, profit_factor, final_capital, trade_df, resolved_trades_df = run_backtest(
            df, selected_pair, initial_capital, risk_pct, sl_pips, tp_pips
        )
        st.session_state.backtest_results = {
            "total_trades": total_trades, "win_rate": win_rate, "total_profit": total_profit,
            "profit_factor": profit_factor, "final_capital": final_capital, "trade_df": trade_df,
            "resolved_trades_df": resolved_trades_df, "pair": selected_pair, "interval": selected_interval, "data_len": len(df)
        }
    st.rerun()

# === DISPLAY MAIN BACKTESTING IF RESULTS EXIST ===
if 'backtest_results' in st.session_state:
    results = st.session_state.backtest_results
    st.markdown("---")
    st.subheader("Backtesting Results (Simulated)")
    st.markdown(f"***Data Tested:*** *{results['pair']}* on *{results['interval']}* interval. *{results['data_len']}* bars.")
    col_t, col_w, col_p, col_f = st.columns(4)
    col_t.metric("Total Trades", results['total_trades'])
    col_w.metric("Win Rate", f"{results['win_rate']:.2%}")
    col_p.metric("Total Profit ($)", f"{results['total_profit']:,.2f}", delta=f"{(results['total_profit']/initial_capital):.2%}")
    col_f.metric("Profit Factor", f"{results['profit_factor']:,.2f}")
    st.subheader("Equity Curve")
    if not results['resolved_trades_df'].empty:
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(x=results['resolved_trades_df']['exit_time'], y=results['resolved_trades_df']['equity'], mode='lines', name='Equity', line=dict(color='#26a69a')))
        equity_fig.update_layout(xaxis_title="Time", yaxis_title="Account Equity ($)", template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white', height=300)
        st.plotly_chart(equity_fig, use_container_width=True)
    else:
        st.info("No resolved trades found with these settings.")
    st.subheader("Detailed Trade Log")
    st.dataframe(results['trade_df'], use_container_width=True)
elif not 'backtest_results' in st.session_state:
    st.markdown("---")
    st.info("Set your parameters in the sidebar and click 'Run Backtest' to see results.")

# === MAIN CHART ===
st.markdown("---")
st.subheader(f"**{selected_pair}** – **{selected_interval}** – Last {len(df)} Candles")
chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")
num_rows = 1; row_heights = [0.7]; rsi_row = 0; macd_row = 0
if show_rsi: num_rows += 1; row_heights.append(0.15)
if show_macd: num_rows += 1; row_heights.append(0.15)
current_row = 2
if show_rsi: rsi_row = current_row; current_row += 1
if show_macd: macd_row = current_row
fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights)
buy_signals = df[df['signal'] == 1]; sell_signals = df[df['signal'] == -1]
theme_colors = {"dark": {"increase": "#26a69a", "decrease": "#ef5350", "line": "#2196f3"}, "light": {"increase": "#28a745", "decrease": "#dc3545", "line": "#0d6efd"}}
current_theme = "dark" if st.session_state.theme == "dark" else "light"
if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price", increasing_line_color=theme_colors[current_theme]["increase"], decreasing_line_color=theme_colors[current_theme]["decrease"]), row=1, col=1)
else:
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name="Price", line=dict(color=theme_colors[current_theme]["line"])), row=1, col=1)
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
fig.update_layout(height=600, template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white', xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# === LIVE SIGNAL ALERT CHECK ===
if is_premium: # Only check for alerts if premium
    check_for_live_signal(df, selected_pair)

# --- NEWS CALENDAR SECTION ---
st.markdown("---")
display_news_calendar()
st.markdown("---")

# === STRATEGY SCANNER (PREMIUM FEATURE) ===
if is_premium:
    with st.expander("🚀 Strategy Scanner (Premium Feature)"):
        st.info("Compare all strategies across multiple pairs and timeframes to find the best performers.")
        
        col1, col2, col3 = st.columns(3)
        scan_pairs = col1.multiselect("Select Pairs", PREMIUM_PAIRS, default=["EUR/USD", "GBP/USD", "USD/JPY"])
        scan_intervals = col2.multiselect("Select Timeframes", list(INTERVALS.keys()), default=["15min", "1h"])
        scan_strategies = col3.multiselect("Select Strategies", [ "RSI Standalone", "MACD Crossover"], default=["RSI Standalone", "MACD Crossover"])

        # Default params for scanner
        scan_params = {
            "rsi_p": 14, "sma_p": 20, "macd_f": 12, "macd_sl": 26, "macd_sig": 9,
            "rsi_l": 30, "rsi_h": 70, "capital": 10000, "risk": 0.01, "sl": 50, "tp": 75
        }
        
        if st.button("Run Full Scan", type="primary"):
            if not all([scan_pairs, scan_intervals, scan_strategies]):
                st.error("Please select at least one Pair, Timeframe, and Strategy.")
            else:
                total_jobs = len(scan_pairs) * len(scan_intervals) * len(scan_strategies)
                progress_bar = st.progress(0, text=f"Starting Scan... (0/{total_jobs})")
                
                scan_results = []
                job_count = 0
                
                for pair in scan_pairs:
                    for interval_key in scan_intervals:
                        interval_val = INTERVALS[interval_key]
                        
                        # Fetch data once per pair/interval
                        data = fetch_data(pair, interval_val)
                        if data.empty:
                            st.warning(f"Could not fetch data for {pair} ({interval_key}). Skipping.")
                            total_jobs -= len(scan_strategies)
                            continue
                            
                        for strategy in scan_strategies:
                            job_count += 1
                            progress_bar.progress(job_count / total_jobs, text=f"Testing {strategy} on {pair} ({interval_key})... ({job_count}/{total_jobs})")
                            
                            # 1. Calculate indicators
                            data_with_indicators = calculate_indicators(data.copy(), scan_params["rsi_p"], scan_params["sma_p"], scan_params["macd_f"], scan_params["macd_sl"], scan_params["macd_sig"])
                            # 2. Apply strategy
                            data_with_signal = apply_strategy(data_with_indicators, strategy, scan_params["rsi_l"], scan_params["rsi_h"])
                            data_with_signal = data_with_signal.dropna()
                            
                            if data_with_signal.empty:
                                continue
                                
                            # 3. Run backtest (Passing pair_name as an argument)
                            total_trades, win_rate, total_profit, pf, _, _, _ = run_backtest(
                                data_with_signal, 
                                pair, # Pass the pair name
                                scan_params["capital"], 
                                scan_params["risk"],
                                scan_params["sl"], 
                                scan_params["tp"]
                            )
                            
                            if total_trades > 0:
                                scan_results.append({
                                    "Pair": pair,
                                    "Timeframe": interval_key,
                                    "Strategy": strategy,
                                    "Total Profit ($)": total_profit,
                                    "Win Rate (%)": win_rate * 100,
                                    "Profit Factor": pf,
                                    "Total Trades": total_trades
                                })

                progress_bar.progress(1.0, text="Scan Complete!")
                
                if scan_results:
                    results_df = pd.DataFrame(scan_results).sort_values(by="Total Profit ($)", ascending=False).reset_index(drop=True)
                    
                    # --- Styled "Heatmap" Table ---
                    def style_profit(val):
                        color = '#26a69a' if val > 0 else '#ef5350' if val < 0 else '#f0f0f0'
                        return f'color: {color}; font-weight: bold;'
                    def style_win_rate(val):
                        # Create a color gradient for win rate
                        # Red at 0%, Fades to white at 50%, Green at 100%
                        if val < 50:
                            # Red to White
                            percent = val / 50
                            red = 239; green = 83; blue = 80;
                            alpha = (1 - percent) * 0.5 # 0.5 alpha at 0%, 0 alpha at 50%
                        else:
                            # White to Green
                            percent = (val - 50) / 50
                            red = 38; green = 166; blue = 154;
                            alpha = percent * 0.5 # 0 alpha at 50%, 0.5 alpha at 100%
                        
                        text_color = '#f0f0f0' if st.session_state.theme == 'dark' else '#212529'
                        return f'background-color: rgba({red}, {green}, {blue}, {alpha}); color: {text_color};'

                    def style_profit_factor(val):
                        color = '#26a69a' if val >= 1.0 else '#ef5350'
                        return f'color: {color}; font-weight: bold;'
                        
                    st.dataframe(
                        results_df.style
                            .applymap(style_profit, subset=['Total Profit ($)'])
                            .apply(lambda x: [style_win_rate(v) for v in x], subset=['Win Rate (%)'])
                            .applymap(style_profit_factor, subset=['Profit Factor'])
                            .format({
                                "Total Profit ($)": "${:,.2f}",
                                "Win Rate (%)": "{:.2f}%",
                                "Profit Factor": "{:.2f}"
                            }),
                        use_container_width=True
                    )
                else:
                    st.info("Scan completed, but no trades were found with these settings.")
else:
     st.info("The **🚀 Strategy Scanner** is a Premium feature. Upgrade to compare all strategies at once!")


# === RISK DISCLAIMER ===
st.markdown("---")
st.subheader("⚠️ Risk Disclaimer")
st.warning(
    """
    This is a simulation and not financial advice. All backtest results are based on historical data and do not guarantee future performance. 
    Forex trading is extremely risky and can result in the loss of your entire capital. 
    Always trade responsibly and stick to your risk management plan.
    """
)

# === AUTO-REFRESH COMPONENT ===
components.html("<meta http-equiv='refresh' content='61'>", height=0)
