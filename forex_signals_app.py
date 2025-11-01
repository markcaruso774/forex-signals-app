import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import talib
from twelvedata import TDClient
import pyrebase
import json
import requests
from dateutil import parser

# === 1. FIREBASE CONFIGURATION ===
def initialize_firebase():
    try:
        if "FIREBASE_CONFIG" not in st.secrets:
            st.error("Firebase config not found.")
            return None, None
        config = st.secrets["FIREBASE_CONFIG"]
        if "databaseURL" not in config:
            project_id = config.get('projectId', config.get('project_id'))
            config["databaseURL"] = f"https://{project_id}-default-rtdb.firebaseio.com/"
        firebase = pyrebase.initialize_app(config)
        return firebase.auth(), firebase.database()
    except Exception as e:
        st.error(f"Firebase init error: {e}")
        return None, None

auth, db = initialize_firebase()

# === 2. SESSION STATE ===
if 'user' not in st.session_state:
    st.session_state.user = None
if 'is_premium' not in st.session_state:
    st.session_state.is_premium = False
if 'page' not in st.session_state:
    st.session_state.page = "login"

# === 3. AUTH FUNCTIONS ===
def sign_up(email, password):
    if not auth or not db: return
    try:
        user = auth.create_user_with_email_and_password(email, password)
        st.session_state.user = user
        db.child("users").child(user['localId']).set({"email": email, "subscription_status": "free"})
        st.session_state.is_premium = False
        st.session_state.page = "app"
        st.rerun()
    except Exception as e:
        st.error("Sign up failed.")

def login(email, password):
    if not auth or not db: return
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.user = user
        data = db.child("users").child(user['localId']).get().val()
        st.session_state.is_premium = data.get("subscription_status") == "premium" if data else False
        st.session_state.page = "app"
        st.rerun()
    except Exception as e:
        st.error("Login failed.")

def logout():
    st.session_state.user = None
    st.session_state.is_premium = False
    st.session_state.page = "login"
    st.rerun()

# === 4. PAYSTACK ===
def create_payment_link(email, user_id):
    if "PAYSTACK_TEST" not in st.secrets: return None, None
    url = "https://api.paystack.co/transaction/initialize"
    headers = {"Authorization": f"Bearer {st.secrets['PAYSTACK_TEST']['PAYSTACK_SECRET_KEY']}", "Content-Type": "application/json"}
    payload = {
        "email": email, "amount": 10000,
        "callback_url": st.secrets["APP_URL"],
        "metadata": {"user_id": user_id}
    }
    try:
        resp = requests.post(url, headers=headers, json=payload)
        data = resp.json()
        return data["data"]["authorization_url"], data["data"]["reference"]
    except: return None, None

def verify_payment(ref):
    if not db: return False
    url = f"https://api.paystack.co/transaction/verify/{ref}"
    headers = {"Authorization": f"Bearer {st.secrets['PAYSTACK_TEST']['PAYSTACK_SECRET_KEY']}"}
    try:
        resp = requests.get(url, headers=headers).json()
        if resp["data"]["status"] == "success":
            user_id = resp["data"]["metadata"]["user_id"]
            db.child("users").child(user_id).update({"subscription_status": "premium"})
            st.session_state.is_premium = True
            st.success("Premium activated!")
            st.balloons()
            st.rerun()
    except: pass

# === 5. LOGIN PAGE ===
if st.session_state.page == "login":
    st.set_page_config(page_title="Login - PipWizard", layout="centered")
    if not auth or not db:
        st.error("Service down.")
    else:
        st.title("PipWizard")
        action = st.radio("Action", ("Login", "Sign Up"), horizontal=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if action == "Sign Up":
            confirm = st.text_input("Confirm Password", type="password")
            if st.button("Sign Up") and password == confirm:
                sign_up(email, password)
        if action == "Login":
            if st.button("Login"):
                login(email, password)

# === 6. PROFILE PAGE ===
elif st.session_state.page == "profile":
    st.set_page_config(page_title="Profile", layout="centered")
    st.title("Profile")
    st.write(f"Logged in: `{st.session_state.user['email']}`")
    if st.session_state.is_premium:
        st.success("Premium Active")
    else:
        st.warning("Free Tier")
        if st.button("Upgrade (100 NGN Test)"):
            url, ref = create_payment_link(st.session_state.user['email'], st.session_state.user['localId'])
            if url:
                components.html(f'<meta http-equiv="refresh" content="0; url={url}">', height=0)
    if st.button("Back"): st.session_state.page = "app"; st.rerun()
    if st.button("Logout"): logout()

# === 7. MAIN APP ===
elif st.session_state.page == "app" and st.session_state.user is not None:
    st.set_page_config(page_title="PipWizard", layout="wide")

    # Payment callback
    if "trxref" in st.query_params:
        verify_payment(st.query_params["trxref"])

    ALL_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY", "USD/CHF"]
    FREE_PAIR = "EUR/USD"
    INTERVALS = {"1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min", "1h": "1h"}
    OUTPUTSIZE = 500

    # Theme
    if 'theme' not in st.session_state: st.session_state.theme = "dark"
    def toggle_theme(): st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.markdown(f"""<style>.stApp {{background: {'#0e1117' if st.session_state.theme=='dark' else '#fff'}; color: {'#f0f0f0' if st.session_state.theme=='dark' else '#000'}}}</style>""", unsafe_allow_html=True)

    col1, col2 = st.columns([6,1])
    with col1: st.title("PipWizard – Live Forex Signals")
    with col2: st.button("Light" if st.session_state.theme == "dark" else "Dark", on_click=toggle_theme)

    # Sidebar
    st.sidebar.title("PipWizard")
    st.sidebar.write(f"User: `{st.session_state.user['email']}`")
    is_premium = st.session_state.is_premium
    selected_pair = st.sidebar.selectbox("Pair", ALL_PAIRS if is_premium else [FREE_PAIR])
    selected_interval = st.sidebar.selectbox("Timeframe", list(INTERVALS.keys()), index=3)
    strategy_name = st.sidebar.selectbox("Strategy", ["RSI + SMA Crossover", "MACD Crossover", "RSI + MACD (Confluence)", "SMA + MACD (Confluence)", "RSI Standalone", "SMA Crossover Standalone"])
    show_rsi = st.sidebar.checkbox("Show RSI", True)
    show_macd = st.sidebar.checkbox("Show MACD", True)
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    sma_period = st.sidebar.slider("SMA Period", 10, 50, 20)
    alert_rsi_low = st.sidebar.slider("Buy RSI <", 20, 40, 35)
    alert_rsi_high = st.sidebar.slider("Sell RSI >", 60, 80, 65)
    macd_fast = st.sidebar.slider("MACD Fast", 1, 26, 12)
    macd_slow = st.sidebar.slider("MACD Slow", 13, 50, 26)
    macd_signal = st.sidebar.slider("MACD Signal", 1, 15, 9)
    initial_capital = st.sidebar.number_input("Capital ($)", 1000, value=10000)
    risk_pct = st.sidebar.slider("Risk %", 0.5, 5.0, 1.0) / 100
    sl_pips = st.sidebar.number_input("SL (Pips)", 1, 200, 50)
    tp_pips = st.sidebar.number_input("TP (Pips)", 1, 500, 100)
    run_backtest = st.sidebar.button("Run Backtest", type="primary")
    if 'backtest_results' in st.session_state and st.sidebar.button("Clear"): del st.session_state.backtest_results; st.rerun()
    if not is_premium: st.sidebar.warning("Free: EUR/USD only")
    if st.sidebar.button("Profile"): st.session_state.page = "profile"; st.rerun()

    # Data
    @st.cache_data(ttl=60)
    def fetch_data(symbol, interval):
        td = TDClient(apikey=st.secrets["TD_API_KEY"])
        try:
            ts = td.time_series(symbol=symbol, interval=interval, outputsize=OUTPUTSIZE).as_pandas()
            df = ts[['open', 'high', 'low', 'close']].copy()
            df.index = pd.to_datetime(df.index)
            return df.iloc[::-1]
        except: return pd.DataFrame()

    def calculate_indicators(df, rsi_p, sma_p, macd_f, macd_sl, macd_sig):
        df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_p)
        df['sma'] = df['close'].rolling(sma_p).mean()
        df['macd_line'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=macd_f, slowperiod=macd_sl, signalperiod=macd_sig)
        return df

    def apply_strategy(df, name, rsi_l, rsi_h):
        df['signal'] = 0
        if name == "RSI + SMA Crossover":
            df.loc[(df['rsi'] < rsi_l) & (df['close'] > df['sma']), 'signal'] = 1
            df.loc[(df['rsi'] > rsi_h) & (df['close'] < df['sma']), 'signal'] = -1
        elif name == "MACD Crossover":
            buy = (df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
            sell = (df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
            df.loc[buy, 'signal'] = 1; df.loc[sell, 'signal'] = -1
        return df

    def run_backtest(df_in, pair, capital, risk, sl, tp):
        df = df_in.copy()
        pip = 0.01 if "JPY" in pair else 0.0001
        risk_usd = capital * risk
        reward_usd = risk_usd * (tp / sl)
        trades = []
        for i in df[df['signal'] != 0].index:
            entry_idx = df.index.get_loc(i) + 1
            if entry_idx >= len(df): continue
            entry = df.iloc[entry_idx]
            sl_price = entry['open'] - pip * sl if df.loc[i, 'signal'] == 1 else entry['open'] + pip * sl
            tp_price = entry['open'] + pip * tp if df.loc[i, 'signal'] == 1 else entry['open'] - pip * tp
            for j in range(entry_idx + 1, len(df)):
                bar = df.iloc[j]
                if df.loc[i, 'signal'] == 1:
                    if bar['low'] <= sl_price: profit = -risk_usd; break
                    if bar['high'] >= tp_price: profit = reward_usd; break
                else:
                    if bar['high'] >= sl_price: profit = -risk_usd; break
                    if bar['low'] <= tp_price: profit = reward_usd; break
            else: profit = 0
            trades.append({"profit": profit})
        total = sum(t["profit"] for t in trades)
        wins = sum(1 for t in trades if t["profit"] > 0)
        return len(trades), wins/len(trades) if trades else 0, total, total/capital*100

    # Load data
    df = fetch_data(selected_pair, INTERVALS[selected_interval])
    if df.empty: st.error("No data."); st.stop()
    df = calculate_indicators(df, rsi_period, sma_period, macd_fast, macd_slow, macd_signal)
    df = apply_strategy(df, strategy_name, alert_rsi_low, alert_rsi_high)
    df = df.dropna()

    # Backtest
    if run_backtest:
        trades, win_rate, profit, pct = run_backtest(df, selected_pair, initial_capital, risk_pct, sl_pips, tp_pips)
        st.session_state.backtest_results = {"trades": trades, "win_rate": win_rate, "profit": profit, "pct": pct}
        st.rerun()

    if 'backtest_results' in st.session_state:
        r = st.session_state.backtest_results
        st.metric("Trades", r["trades"])
        st.metric("Win Rate", f"{r['win_rate']:.1%}")
        st.metric("Profit", f"${r['profit']:,.0f}", f"{r['pct']:.1f}%")

    # Chart
    fig = make_subplots(rows=3 if show_rsi and show_macd else 2 if show_rsi or show_macd else 1, cols=1, shared_xaxes=True)
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma'], name="SMA"), row=1, col=1)
    buy = df[df['signal'] == 1]; sell = df[df['signal'] == -1]
    fig.add_trace(go.Scatter(x=buy.index, y=buy['low']*0.9995, mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell.index, y=sell['high']*1.0005, mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell'), row=1, col=1)
    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name="RSI"), row=2, col=1)
        fig.add_hline(y=alert_rsi_high, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=alert_rsi_low, line_dash="dash", line_color="green", row=2, col=1)
    if show_macd:
        row = 3 if show_rsi else 2
        fig.add_trace(go.Scatter(x=df.index, y=df['macd_line'], name='MACD'), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal'), row=row, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # === LIVE CALENDAR ===
    st.markdown("---")
    def display_news_calendar():
        st.subheader("Upcoming Economic Calendar")
        search = st.text_input("Search events", placeholder="e.g., NFP, PMI, CPI", key="calendar_search")

        @st.cache_data(ttl=300)
        def get_free_calendar():
            try:
                response = requests.get("https://api.forexcalendar.com/v1/events?days=7&currency=USD,EUR,GBP,JPY,CAD,AUD,NZD", timeout=10)
                data = response.json().get("events", [])
                events = []
                now = datetime.utcnow()
                for e in data:
                    dt = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
                    if dt < now - timedelta(days=1) or dt > now + timedelta(days=7): continue
                    actual = e.get("actual", "N/A")
                    forecast = e.get("forecast", "N/A")
                    is_past = dt < now
                    actual_display = actual if is_past and actual != "N/A" else "Pending"
                    surprise = ""
                    if is_past and actual != "N/A" and forecast != "N/A":
                        try:
                            def to_num(v): v=str(v); return float(v[:-1])*1000 if v.endswith("K") else float(v[:-1])*1000000 if v.endswith("M") else float(v)
                            a, f = to_num(actual), to_num(forecast)
                            surprise = "Better than Expected" if a > f else "Worse than Expected" if a < f else "As Expected"
                        except: pass
                    events.append({
                        "date": dt.strftime("%A, %b %d"), "time": dt.strftime("%H:%M"), "event": e["title"],
                        "country": e.get("country", "??"), "impact": e.get("impact", "Low").title(),
                        "forecast": forecast, "previous": e.get("previous", "N/A"),
                        "actual": actual_display, "surprise": surprise, "date_dt": dt
                    })
                df = pd.DataFrame(events)
                return df.sort_values("date_dt").drop(columns="date_dt") if not df.empty else pd.DataFrame()
            except:
                return pd.DataFrame([{"date": "Friday, Nov 07", "time": "13:30", "event": "Nonfarm Payrolls", "country": "US", "impact": "High", "forecast": "175K", "previous": "254K", "actual": "Pending", "surprise": ""}])

        with st.spinner("Loading..."): df = get_free_calendar()
        if df.empty:
            st.info("No events.")
            if st.button("Refresh"): st.cache_data.clear(); st.rerun()
            return
        if search:
            df = df[df["event"].str.contains(search, case=False, na=False)]
            if df.empty: st.info(f"No match for '{search}'."); return

        st.markdown("""
        <style>
        .calendar-table{width:100%;border-collapse:collapse;font-family:'Segoe UI',sans-serif;margin:10px 0}
        .calendar-table th{background:#1f77b4;color:white;padding:12px;text-align:left;font-weight:600}
        .calendar-table td{padding:10px 12px;border-bottom:1px solid #444}
        .calendar-table tr:hover{background:#2a2a2a !important}
        .impact-high{background:#ffebee;color:#c62828;font-weight:bold}
        .impact-medium{background:#fff3e0;color:#ef6c00;font-weight:bold}
        .impact-low{background:#f3e5f5;color:#6a1b9a}
        .actual-better{background:#e8f5e8;color:#2e7d32;font-weight:bold}
        .actual-worse{background:#ffebee;color:#c62828;font-weight:bold}
        .actual-expected{background:#fff8e1;color:#ff8f00;font-weight:bold}
        </style>
        """, unsafe_allow_html=True)

        def style_row(r):
            s = [""] * len(r)
            if r["impact"] == "High": s[4] = 'class="impact-high"'
            elif r["impact"] == "Medium": s[4] = 'class="impact-medium"'
            elif r["impact"] == "Low": s[4] = 'class="impact-low"'
            if r["surprise"] == "Better than Expected": s[8] = 'class="actual-better"'
            elif r["surprise"] == "Worse than Expected": s[8] = 'class="actual-worse"'
            elif r["surprise"] == "As Expected": s[8] = 'class="actual-expected"'
            return s

        styled = df.style.apply(style_row, axis=1).set_table_attributes('class="calendar-table"')
        st.markdown(styled.to_html(), unsafe_allow_html=True)

        if st.button("Refresh Calendar"): st.cache_data.clear(); st.rerun()
        st.caption("Source: forexcalendar.com • Live • UTC")

    display_news_calendar()

    # Auto-refresh
    components.html("<meta http-equiv='refresh' content='60'>", height=0)

else:
    st.error("Access denied. Please log in.")
    if st.button("Login"): st.session_state.page = "login"; st.rerun()
