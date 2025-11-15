import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone 
import streamlit.components.v1 as components
import talib
from twelvedata import TDClient
import pyrebase  # For Firebase
import json      # For Firebase
import requests  # For Paystack & Telegram
import uuid      # For unique alert IDs
import yfinance as yf # === NEW: For Unlimited Free Scanning ===

# --- NEW LIBRARY ---
from lightweight_charts.widgets import StreamlitChart

# === 1. FIREBASE CONFIGURATION ===
def initialize_firebase():
    try:
        if "FIREBASE_CONFIG" not in st.secrets:
            st.error("Firebase config not found in Streamlit Secrets.")
            return None, None
        
        config = st.secrets["FIREBASE_CONFIG"]
        
        if "databaseURL" not in config:
            project_id = config.get('projectId', config.get('project_id'))
            if project_id:
                config["databaseURL"] = f"https://{project_id}-default-rtdb.firebaseio.com/"
            else:
                config["databaseURL"] = f"https://{config['authDomain'].split('.')[0]}-default-rtdb.firebaseio.com/"
        
        try:
            firebase = pyrebase.initialize_app(config)
            auth = firebase.auth()
            db = firebase.database()
            return auth, db
        except Exception as e:
            st.error(f"Error initializing Firebase: {e}")
            return None, None
    except Exception as e:
        st.error(f"Error loading Secrets: {e}")
        return None, None

auth, db = initialize_firebase()

# === 2. SESSION STATE ===
if 'user' not in st.session_state: st.session_state.user = None
if 'is_premium' not in st.session_state: st.session_state.is_premium = False
if 'page' not in st.session_state: st.session_state.page = "login" 

# === 3. AUTH FUNCTIONS ===
def sign_up(email, password):
    if auth is None or db is None: return
    try:
        user = auth.create_user_with_email_and_password(email, password)
        st.session_state.user = user
        user_data = {"email": email, "subscription_status": "free"}
        db.child("users").child(user['localId']).set(user_data, user['idToken'])
        st.session_state.is_premium = False
        st.session_state.page = "app"
        st.rerun()
    except Exception as e:
        st.error(f"Sign up failed: {e}")

def login(email, password):
    if auth is None or db is None: return
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.user = user
        user_data = db.child("users").child(user['localId']).get(user['idToken']).val()
        
        if user_data:
            st.session_state.is_premium = (user_data.get("subscription_status") == "premium")
            settings = user_data.get("settings", {}) 
            st.session_state.selected_pair = settings.get("selected_pair", "EUR/USD")
            st.session_state.selected_interval = settings.get("selected_interval", "1h")
            st.session_state.strategy_name = settings.get("strategy_name", "RSI + SMA Crossover")
            st.session_state.rsi_period = settings.get("rsi_period", 14)
            st.session_state.sma_period = settings.get("sma_period", 20)
            st.session_state.alert_rsi_low = settings.get("alert_rsi_low", 35)
            st.session_state.alert_rsi_high = settings.get("alert_rsi_high", 65)
            st.session_state.macd_fast = settings.get("macd_fast", 12)
            st.session_state.macd_slow = settings.get("macd_slow", 26)
            st.session_state.macd_signal = settings.get("macd_signal", 9)
            st.session_state.capital = settings.get("capital", 10000)
            st.session_state.risk_pct = settings.get("risk_pct", 1.0)
            st.session_state.sl_pips = settings.get("sl_pips", 50)
            st.session_state.tp_pips = settings.get("tp_pips", 100)
            st.session_state.telegram_chat_id = settings.get("telegram_chat_id", "")
        else:
            st.session_state.is_premium = False

        st.session_state.page = "app"
        st.rerun()
    except Exception as e:
        st.error(f"Login Failed: {e}")

def logout():
    st.session_state.user = None
    st.session_state.is_premium = False
    st.session_state.page = "login"
    st.rerun()

# === 4. PAYMENT & EXCHANGE RATE LOGIC ===
def get_ngn_exchange_rate():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        return response.json()['rates']['NGN']
    except:
        return 1650.0 

def create_payment_link(email, user_id):
    price_in_usd = 20.00
    exchange_rate = get_ngn_exchange_rate()
    amount_kobo = int(price_in_usd * exchange_rate * 100)
    
    if "PAYSTACK_TEST" in st.secrets: secret_key = st.secrets['PAYSTACK_TEST']['PAYSTACK_SECRET_KEY']
    elif "PAYSTACK_LIVE" in st.secrets: secret_key = st.secrets['PAYSTACK_LIVE']['PAYSTACK_SECRET_KEY']
    else: st.error("Paystack keys missing."); return None, None, 0

    url = "https://api.paystack.co/transaction/initialize"
    headers = {"Authorization": f"Bearer {secret_key}", "Content-Type": "application/json"}
    if "APP_URL" not in st.secrets: st.error("APP_URL missing."); return None, None, 0
    
    payload = {
        "email": email, "amount": amount_kobo, "callback_url": st.secrets["APP_URL"],
        "metadata": {"user_id": user_id, "user_email": email, "description": "PipWizard Premium ($20)"}
    }
    try:
        res = requests.post(url, headers=headers, json=payload).json()
        if res.get("status"): return res["data"]["authorization_url"], res["data"]["reference"], (amount_kobo/100)
        else: st.error(f"Paystack Error: {res.get('message')}"); return None, None, 0
    except Exception as e: st.error(f"Error: {e}"); return None, None, 0

def verify_payment(reference):
    if "PAYSTACK_TEST" in st.secrets: secret_key = st.secrets['PAYSTACK_TEST']['PAYSTACK_SECRET_KEY']
    elif "PAYSTACK_LIVE" in st.secrets: secret_key = st.secrets['PAYSTACK_LIVE']['PAYSTACK_SECRET_KEY']
    else: return False

    try:
        res = requests.get(f"https://api.paystack.co/transaction/verify/{reference}", headers={"Authorization": f"Bearer {secret_key}"}).json()
        if res.get("status") and res["data"]["status"] == "success":
            uid = res["data"]["metadata"].get("user_id")
            if uid and st.session_state.user:
                db.child("users").child(uid).update({"subscription_status": "premium"}, st.session_state.user['idToken'])
                st.session_state.is_premium = True
                st.balloons(); st.success("Premium Activated!")
                try: st.query_params.clear()
                except: pass 
                return True
    except Exception as e: st.error(f"Verify Error: {e}")
    return False

# === 5. LOGIN PAGE ===
if st.session_state.page == "login":
    st.set_page_config(page_title="Login - PipWizard", page_icon="🧙‍♂️", layout="centered")
    if auth is None or db is None: st.error("App failed to start.")
    else:
        st.title("PipWizard 🧙‍♂️ 📈📉")
        if "trxref" in st.query_params: st.info("✅ **Payment Detected!** Log in to activate.")
        else: st.text("Please log in or sign up.")
        action = st.radio("Action:", ("Login", "Sign Up"), horizontal=True, index=0)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if action == "Sign Up":
            if st.button("Sign Up"): sign_up(email, password)
        else:
            if st.button("Login"): login(email, password)

# === 6. PROFILE PAGE ===
elif st.session_state.page == "profile":
    st.set_page_config(page_title="Profile", page_icon="🧙‍♂️", layout="centered")
    st.title("Profile 🧙‍♂️")
    st.write(f"User: `{st.session_state.user['email']}`")
    
    if st.session_state.is_premium:
        st.success("You are a **Premium User**! 🌟")
    else:
        st.warning("Status: **Free Tier**")
        st.markdown("Upgrade to **Premium ($20.00/month)** to unlock everything.")
        if st.button("Upgrade to Premium Now! ($20.00)", type="primary"):
            with st.spinner("Processing..."):
                url, ref, amt = create_payment_link(st.session_state.user['email'], st.session_state.user['localId'])
                if url:
                    st.success(f"Exchange Rate Applied. Charge: **₦{amt:,.2f}**")
                    st.markdown(f'[**Click Here to Pay**]({url})', unsafe_allow_html=True)
                    components.html(f'<meta http-equiv="refresh" content="2; url={url}">', height=0)

    st.markdown("---")
    if st.button("Back to App"): st.session_state.page = "app"; st.rerun()
    if st.button("Logout", type="secondary"): logout()

# === 7. MAIN APP ===
elif st.session_state.page == "app" and st.session_state.user:
    st.set_page_config(page_title="PipWizard", page_icon="🧙‍♂️", layout="wide")
    if "trxref" in st.query_params:
        with st.spinner("Verifying..."): verify_payment(st.query_params["trxref"])

    ALL_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY", "USD/CHF"]
    FREE_PAIR = "EUR/USD"
    PREMIUM_PAIRS = ALL_PAIRS
    INTERVALS = {"1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min", "1h": "1h"}
    OUTPUTSIZE = 500 

    if 'theme' not in st.session_state: st.session_state.theme = "dark"
    def toggle_theme(): st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    
    st.markdown(f"""<style>
        .stApp {{ background-color: {'#0e1117' if st.session_state.theme == 'dark' else '#ffffff'}; color: {'#f0f0f0' if st.session_state.theme == 'dark' else '#000000'}; }}
        section[data-testid="stSidebar"] {{ background-color: {'#0e1117' if st.session_state.theme == 'dark' else '#f0f2f6'}; }}
        .stMarkdown, .stText, p, h1, h2, h3, span, label {{ color: {'#f0f0f0' if st.session_state.theme == 'dark' else '#000000'} !important; }}
        div.stButton > button:first-child {{ background-color: #007bff !important; color: white !important; border: none; }}
        div[data-testid="stTextInput"] input {{ background-color: #ffffff !important; color: #000000 !important; }}
    </style>""", unsafe_allow_html=True)

    col1, col2 = st.columns([6, 1])
    with col1: st.title("PipWizard 🧙‍♂️ 📈📉")
    with col2: 
        if st.button("☀️/🌙", on_click=toggle_theme): st.rerun()

    # === HYBRID DATA FETCHER (THE FIX) ===
    @st.cache_data(ttl=60)
    def fetch_data(symbol, interval, source="TwelveData", output_size=500):
        # 1. YAHOO (For Scanner - Unlimited)
        if source == "Yahoo":
            yf_map = {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "1h": "1h"}
            yf_sym = f"{symbol.replace('/', '')}=X"
            try:
                df = yf.download(yf_sym, interval=yf_map.get(interval, "1h"), period="5d", progress=False)
                if df.empty: return pd.DataFrame()
                df = df.reset_index()
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                df.rename(columns={'date': 'datetime', 'index': 'datetime'}, inplace=True)
                df.set_index('datetime', inplace=True)
                return df[['open', 'high', 'low', 'close']].dropna()
            except: return pd.DataFrame()

        # 2. TWELVE DATA (For Chart - Pretty)
        elif source == "TwelveData":
            if "TD_API_KEY" not in st.secrets: return pd.DataFrame()
            td = TDClient(apikey=st.secrets["TD_API_KEY"])
            try:
                ts = td.time_series(symbol=symbol, interval=interval, outputsize=output_size).as_pandas()
                if ts is None or ts.empty: return pd.DataFrame()
                df = ts[['open', 'high', 'low', 'close']].copy()
                df.index = pd.to_datetime(df.index)
                return df.iloc[::-1]
            except: return pd.DataFrame()
        return pd.DataFrame()

    # ... (Indicator & Strategy functions remain same) ...
    def calculate_indicators(df, rsi_p, sma_p, macd_f, macd_sl, macd_sig):
        df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_p)
        df['sma'] = df['close'].rolling(sma_p).mean()
        df['macd_line'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=macd_f, slowperiod=macd_sl, signalperiod=macd_sig)
        return df

    def apply_strategy(df, strategy_name, rsi_l, rsi_h):
        df['signal'] = 0
        if strategy_name == "RSI + SMA Crossover":
            df.loc[(df['rsi'] < rsi_l) & (df['close'] > df['sma']), 'signal'] = 1
            df.loc[(df['rsi'] > rsi_h) & (df['close'] < df['sma']), 'signal'] = -1
        elif strategy_name == "MACD Crossover":
            buy_cond = (df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
            sell_cond = (df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
            df.loc[buy_cond, 'signal'] = 1; df.loc[sell_cond, 'signal'] = -1
        return df

    # ... (Backtest function remains same) ...
    def run_backtest(df_in, pair_name, initial_capital, risk_per_trade, sl_pips, tp_pips):
        df = df_in.copy(); trades = []
        if "JPY" in pair_name: PIP_MULTIPLIER = 0.01
        else: PIP_MULTIPLIER = 0.0001
        RISK_PIPS_VALUE = sl_pips * PIP_MULTIPLIER
        REWARD_PIPS_VALUE = tp_pips * PIP_MULTIPLIER
        if sl_pips == 0: return 0, 0, 0, 0, initial_capital, pd.DataFrame(), pd.DataFrame() 
        MAX_RISK_USD = initial_capital * risk_per_trade
        REWARD_USD = MAX_RISK_USD * (tp_pips / sl_pips)
        signal_bars = df[df['signal'] != 0]
        for i in range(len(signal_bars)):
            signal_row, signal_type = signal_bars.iloc[i], signal_bars.iloc[i]['signal']
            try: signal_index = df.index.get_loc(signal_row.name)
            except KeyError: continue
            if signal_index + 1 >= len(df): continue 
            entry_bar = df.iloc[signal_index + 1]
            entry_price, entry_time = entry_bar['open'], entry_bar.name
            if signal_type == 1: stop_loss = entry_price - RISK_PIPS_VALUE; take_profit = entry_price + REWARD_PIPS_VALUE
            else: stop_loss = entry_price + RISK_PIPS_VALUE; take_profit = entry_price - REWARD_PIPS_VALUE
            result, profit_loss, exit_time = 'OPEN', 0.0, None
            for j in range(signal_index + 2, len(df)):
                future_bar = df.iloc[j]
                if signal_type == 1: 
                    if future_bar['low'] <= stop_loss: result, profit_loss, exit_time = 'LOSS', -MAX_RISK_USD, future_bar.name; break
                    elif future_bar['high'] >= take_profit: result, profit_loss, exit_time = 'WIN', REWARD_USD, future_bar.name; break
                elif signal_type == -1: 
                    if future_bar['high'] >= stop_loss: result, profit_loss, exit_time = 'LOSS', -MAX_RISK_USD, future_bar.name; break
                    elif future_bar['low'] <= take_profit: result, profit_loss, exit_time = 'WIN', REWARD_USD, future_bar.name; break
            if result == 'OPEN': result, profit_loss, exit_time = 'UNRESOLVED', 0.0, df.iloc[-1].name
            trades.append({"entry_time": entry_time, "exit_time": exit_time, "signal": "BUY" if signal_type == 1 else "SELL", "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit, "result": result, "profit_loss": profit_loss})
        if not trades: return 0, 0, 0, 0, initial_capital, pd.DataFrame(), pd.DataFrame() 
        trade_log = pd.DataFrame(trades).set_index('entry_time')
        resolved_trades = trade_log[trade_log['result'].isin(['WIN', 'LOSS'])].copy()
        if resolved_trades.empty: return 0, 0, 0, 0, initial_capital, trade_log, resolved_trades
        
        resolved_trades.sort_values(by='exit_time', inplace=True)
        
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

    # Sidebar Setup
    st.sidebar.title("PipWizard 🧙‍♂️")
    st.sidebar.write(f"User: {st.session_state.user.get('email')}")
    is_premium = st.session_state.is_premium
    if is_premium:
        selected_pair = st.sidebar.selectbox("Select Pair", PREMIUM_PAIRS, index=0)
        st.sidebar.success("✨ Premium Active")
    else:
        selected_pair = FREE_PAIR
        st.sidebar.warning("Free Tier: EUR/USD Only")

    selected_interval = st.sidebar.selectbox("Timeframe", list(INTERVALS.keys()), index=3)
    st.sidebar.markdown("---")
    strategy_name = st.sidebar.selectbox("Strategy", ["RSI + SMA Crossover", "MACD Crossover"])
    
    # Indicators
    st.sidebar.markdown("**Indicators**")
    rsi_period = st.sidebar.slider("RSI", 5, 30, 14)
    sma_period = st.sidebar.slider("SMA", 10, 50, 20)
    alert_rsi_low = st.sidebar.slider("RSI Buy <", 20, 40, 35)
    alert_rsi_high = st.sidebar.slider("RSI Sell >", 60, 80, 65)
    macd_fast = st.sidebar.slider("MACD Fast", 1, 26, 12)
    macd_slow = st.sidebar.slider("MACD Slow", 13, 50, 26)
    macd_signal = st.sidebar.slider("MACD Signal", 1, 15, 9)
    
    # Backtest Params
    st.sidebar.markdown("**Backtest**")
    initial_capital = st.sidebar.number_input("Capital", 100, 100000, 10000)
    risk_pct = st.sidebar.slider("Risk %", 0.5, 5.0, 1.0) / 100
    sl_pips = st.sidebar.number_input("SL Pips", 10, 200, 50)
    tp_pips = st.sidebar.number_input("TP Pips", 10, 500, 100)
    
    if st.sidebar.button("Run Backtest", type="primary"):
        df = fetch_data(selected_pair, INTERVALS[selected_interval], source="Yahoo") # YAHOO FOR BACKTEST
        if not df.empty:
            df = calculate_indicators(df, rsi_period, sma_period, macd_fast, macd_slow, macd_signal)
            df = apply_strategy(df, strategy_name, alert_rsi_low, alert_rsi_high)
            df_bt = df.dropna()
            res = run_backtest(df_bt, selected_pair, initial_capital, risk_pct, sl_pips, tp_pips)
            st.session_state.backtest_results = res
            st.rerun()

    if not is_premium:
        st.sidebar.info("⭐ Upgrade to Premium ($20) to unlock Scanner")
        if st.sidebar.button("Upgrade Now"): st.session_state.page = "profile"; st.rerun()
        
    # Main Chart
    df = fetch_data(selected_pair, INTERVALS[selected_interval], source="TwelveData") # TWELVE DATA FOR CHART
    if not df.empty:
        df = calculate_indicators(df, rsi_period, sma_period, macd_fast, macd_slow, macd_signal)
        df = apply_strategy(df, strategy_name, alert_rsi_low, alert_rsi_high)
        
        chart = StreamlitChart(width=1000, height=500)
        chart.layout_options = {"backgroundColor": "#0e1117" if st.session_state.theme == 'dark' else "#ffffff", "textColor": "#f0f0f0" if st.session_state.theme == 'dark' else "#000000"}
        
        df_chart = df.reset_index()
        df_chart['time'] = df_chart['datetime'].dt.strftime('%Y-%m-%d %H:%M')
        chart.set(df_chart[['time', 'open', 'high', 'low', 'close']])
        chart.load()
        
        # Backtest Results Display
        if 'backtest_results' in st.session_state and isinstance(st.session_state.backtest_results, tuple):
            res = st.session_state.backtest_results
            if len(res) == 7:
                st.subheader("Backtest Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Profit", f"${res[2]:.2f}")
                c2.metric("Win Rate", f"{res[1]:.2%}")
                c3.metric("Trades", res[0])

    # === SCANNER (UPDATED TO USE YAHOO) ===
    if is_premium:
        st.markdown("---")
        st.subheader("🚀 Strategy Scanner")
        scan_pairs = st.multiselect("Pairs", PREMIUM_PAIRS)
        scan_timeframes = st.multiselect("Timeframes", list(INTERVALS.keys()))
        
        if st.button("Start Scan", type="primary"):
            if scan_pairs and scan_timeframes:
                results = []
                progress = st.progress(0)
                total = len(scan_pairs) * len(scan_timeframes)
                done = 0
                
                for pair in scan_pairs:
                    for tf in scan_timeframes:
                        # === KEY CHANGE: USE YAHOO FOR SCANNING ===
                        data = fetch_data(pair, INTERVALS[tf], source="Yahoo")
                        if not data.empty:
                            data = calculate_indicators(data, rsi_period, sma_period, macd_fast, macd_slow, macd_signal)
                            data = apply_strategy(data, strategy_name, alert_rsi_low, alert_rsi_high)
                            data = data.dropna()
                            res = run_backtest(data, pair, initial_capital, risk_pct, sl_pips, tp_pips)
                            if res[0] > 0: # If trades exist
                                results.append({"Pair": pair, "TF": tf, "Profit": res[2], "Win Rate": res[1]})
                        done += 1
                        progress.progress(done/total)
                
                if results:
                    st.dataframe(pd.DataFrame(results).sort_values("Profit", ascending=False))
                else:
                    st.info("No profitable trades found in scan.")

    # Sidebar Bottom
    st.sidebar.markdown("---")
    if st.sidebar.button("Profile & Logout"): st.session_state.page = "profile"; st.rerun()

# === 8. ERROR HANDLING ===
elif not st.session_state.user:
    st.error("App failed to initialize.")
