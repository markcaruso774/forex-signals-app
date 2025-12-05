import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone 
import streamlit.components.v1 as components
import talib
from twelvedata import TDClient
import pyrebase  # For User Authentication
import firebase_admin # For Secure DB Writes
from firebase_admin import credentials, db as admin_db
import json      
import requests  
import uuid      
import yfinance as yf 
import xml.etree.ElementTree as ET 
from lightweight_charts.widgets import StreamlitChart
import os

# === 1. FIREBASE CONFIGURATION ===
def initialize_firebase():
    try:
        if "FIREBASE_CONFIG" not in st.secrets:
            st.error("Secrets: FIREBASE_CONFIG missing.")
            return None, None
        
        config = st.secrets["FIREBASE_CONFIG"]
        if "databaseURL" not in config:
             project_id = config.get('projectId', config.get('project_id'))
             if project_id:
                 config["databaseURL"] = f"https://{project_id}-default-rtdb.firebaseio.com/"
             else:
                 config["databaseURL"] = f"https://{config['authDomain'].split('.')[0]}-default-rtdb.firebaseio.com/"

        firebase = pyrebase.initialize_app(config)
        auth = firebase.auth()
        db = firebase.database()
    except Exception as e:
        st.error(f"Client Firebase Error: {e}")
        return None, None

    try:
        if not firebase_admin._apps:
            if "FIREBASE_ADMIN" in st.secrets:
                cred_dict = dict(st.secrets["FIREBASE_ADMIN"])
                if "private_key" in cred_dict:
                    cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
                
                db_url = config.get("databaseURL")
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred, {"databaseURL": db_url})
            else:
                st.warning("Secrets: FIREBASE_ADMIN missing.")
    except Exception as e:
        st.warning(f"Admin Init Warning: {e}")

    return auth, db

auth, db = initialize_firebase()

# === 2. SESSION STATE ===
if 'user' not in st.session_state: st.session_state.user = None
if 'is_premium' not in st.session_state: st.session_state.is_premium = False
if 'page' not in st.session_state: st.session_state.page = "login" 

# === HELPER: LOGO LOADER ===
def get_page_icon():
    return "logo.png" if os.path.exists("logo.png") else "🧙‍♂️"

def show_sidebar_logo():
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", width=150) 
    else:
        st.sidebar.title("PipWizard 🧙‍♂️")

# === 3. AUTH FUNCTIONS ===
def sign_up(email, password):
    if auth is None: return
    try:
        user = auth.create_user_with_email_and_password(email, password)
        st.session_state.user = user
        try:
            ref = admin_db.reference(f"users/{user['localId']}")
            ref.set({"email": email, "subscription_status": "free"})
        except:
            db.child("users").child(user['localId']).set({"email": email, "subscription_status": "free"}, user['idToken'])
        st.session_state.is_premium = False
        st.session_state.page = "app"
        st.rerun()
    except Exception as e:
        st.error(f"Sign up failed: {e}")

def login(email, password):
    if auth is None: return
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.user = user
        user_data = None
        try:
            ref = admin_db.reference(f"users/{user['localId']}")
            user_data = ref.get()
        except:
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
            try:
                ref = admin_db.reference(f"users/{user['localId']}")
                ref.set({"email": email, "subscription_status": "free"})
                st.session_state.is_premium = False
            except:
                st.error("Could not create missing profile.")
                return

        st.session_state.page = "app"
        st.rerun()
    except Exception as e:
        st.error(f"Login Failed: {e}")

def logout():
    st.session_state.user = None
    st.session_state.is_premium = False
    st.session_state.page = "login"
    st.rerun()

# === 4. PAYMENT FUNCTIONS ===
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
            if uid:
                try:
                    ref = admin_db.reference(f"users/{uid}")
                    ref.update({"subscription_status": "premium"})
                    if st.session_state.user:
                        st.session_state.is_premium = True
                        st.balloons(); st.success("Premium Activated!")
                        try: st.query_params.clear()
                        except: pass
                    return True
                except Exception as e:
                    st.error(f"DB Update Error: {e}")
    except Exception as e: st.error(f"Verify Error: {e}")
    return False

# === 5. LOGIN PAGE ===
if st.session_state.page == "login":
    st.set_page_config(page_title="Login - PipWizard", page_icon=get_page_icon(), layout="centered")
    if auth is None or db is None: st.error("App failed to start.")
    else:
        if os.path.exists("logo.png"): st.image("logo.png", width=120)
        else: st.title("PipWizard 🧙‍♂️")
        st.markdown("### 👋 Welcome to PipWizard!")
        st.markdown("#### Live Forex Signals & Strategy Tester")
        if "trxref" in st.query_params: st.info("✅ **Payment Detected!** Log in to activate.")
        else: st.text("Please log in or sign up to continue.")
        action = st.radio("Action:", ("Login", "Sign Up"), horizontal=True, index=0)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if action == "Sign Up":
            if st.button("Sign Up"): sign_up(email, password)
        else:
            if st.button("Login"): login(email, password)

# === 6. PROFILE PAGE ===
elif st.session_state.page == "profile":
    st.set_page_config(page_title="Profile", page_icon=get_page_icon(), layout="centered")
    if os.path.exists("logo.png"): st.image("logo.png", width=100)
    st.title("Profile")
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
    st.set_page_config(page_title="PipWizard", page_icon=get_page_icon(), layout="wide")
    if "trxref" in st.query_params:
        with st.spinner("Verifying..."): verify_payment(st.query_params["trxref"])

    ALL_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY", "USD/CHF", "XAU/USD", "BTC/USD"]
    FREE_PAIR = "EUR/USD"
    PREMIUM_PAIRS = ALL_PAIRS
    INTERVALS = {"1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min", "1h": "1h"}
    OUTPUTSIZE = 500 

    if 'theme' not in st.session_state: st.session_state.theme = "dark"
    def toggle_theme(): st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    
    st.markdown(f"""<style>
        .stApp {{ background-color: {'#0e1117' if st.session_state.theme == 'dark' else '#ffffff'}; color: {'#f0f0f0' if st.session_state.theme == 'dark' else '#000000'}; }}
        section[data-testid="stSidebar"] {{ background-color: {'#0e1117' if st.session_state.theme == 'dark' else '#f0f2f6'}; }}
        .block-container {{ padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 100% !important; }}
        div[data-baseweb="select"] > div {{ cursor: pointer !important; }}
        div[data-baseweb="select"] > div > div {{ cursor: pointer !important; }}
        .stMarkdown, .stText, p, h1, h2, h3, span, label {{ color: {'#f0f0f0' if st.session_state.theme == 'dark' else '#000000'} !important; }}
        div.stButton > button:first-child {{ background-color: #007bff !important; color: white !important; border: none; }}
        div[data-testid="stTextInput"] input {{ background-color: #ffffff !important; color: #000000 !important; }}
        
        /* CARD STYLE FOR ALERTS */
        .alert-card {{
            border: 1px solid #444;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
            background-color: #1a1a1a;
        }}
        .alert-header {{ font-weight: bold; display: flex; justify-content: space-between; }}
        .alert-win {{ color: #26a69a; font-weight: bold; float: right; }}
        .alert-loss {{ color: #ef5350; font-weight: bold; float: right; }}
        .alert-details {{ font-size: 0.85em; color: #ccc; margin-top: 4px; }}
    </style>""", unsafe_allow_html=True)

    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("PipWizard 🧙‍♂️ 📈📉 – Live Signals")
        st.markdown("##### 👋 Welcome! Analyze charts, check the calendar, or scan for signals.")
    with col2:
        theme_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
        if st.button(theme_label, key="theme_toggle", on_click=toggle_theme):
            st.rerun()

    with st.expander("📖 User Guide & Telegram Setup"):
        st.markdown(
            """
            ### What is PipWizard?
            PipWizard is a tool to help you **test trading strategies**...
            
            ### 📲 How to Setup Telegram Alerts (New!)
            1.  Search Telegram for **@userinfobot** -> Click "Start" -> Copy your **ID**.
            2.  Paste it in the sidebar below and click **"Save Settings"**.
            """
        )
    
    show_sidebar_logo()
    user_id = st.session_state.user['localId']
    user_email = st.session_state.user.get('email', 'User')
    st.sidebar.write(f"Logged in as: `{user_email}`")
    is_premium = st.session_state.is_premium

    if is_premium:
        selected_pair = st.sidebar.selectbox("Select Pair", PREMIUM_PAIRS, index=0, key="selected_pair")
        st.sidebar.markdown(f"""<div style="background-color: rgba(38, 166, 154, 0.2); border: 1px solid #26a69a; padding: 10px; border-radius: 5px; margin-bottom: 20px;"><p style="color: #26a69a !important; font-weight: bold; margin: 0; text-align: center;">✨ Premium Active</p></div>""", unsafe_allow_html=True)
    else:
        selected_pair = FREE_PAIR
        st.sidebar.warning("Free Tier: EUR/USD Only")

    selected_interval = st.sidebar.selectbox("Timeframe", options=list(INTERVALS.keys()), index=3, format_func=lambda x: x.replace("min", " minute").replace("1h", "1 hour"), key="selected_interval")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Strategy Selection")
    strategies_list = [
        "RSI + SMA Crossover", "MACD Crossover", "RSI + MACD (Confluence)", 
        "SMA + MACD (Confluence)", "RSI Standalone", "SMA Crossover Standalone",
        "EMA Golden Cross", "Bollinger Bands Bounce", "Stochastic Oscillator", "EMA Trend + Price Action"
    ]
    strategy_name = st.sidebar.selectbox("Choose a Strategy", strategies_list, key="strategy_name")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Indicator Configuration")
    # ... (Sliders stay same) ...
    rsi_period = 14; sma_period = 20; macd_fast = 12; macd_slow = 26; macd_signal = 9; ema_short = 50; ema_long = 200; bb_period = 20; bb_std = 2.0; stoch_k = 14; stoch_d = 3
    
    if "RSI" in strategy_name:
        rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14, key='rsi_period')
        alert_rsi_low = st.sidebar.slider("Buy RSI <", 20, 40, 35, key='rsi_low')
        alert_rsi_high = st.sidebar.slider("Sell RSI >", 60, 80, 65, key='rsi_high')
    else: alert_rsi_low, alert_rsi_high = 35, 65
        
    if "MACD" in strategy_name:
        st.sidebar.markdown("**MACD Settings**")
        macd_fast = st.sidebar.slider("Fast", 1, 26, 12, key='macd_fast')
        macd_slow = st.sidebar.slider("Slow", 13, 50, 26, key='macd_slow')
        macd_signal = st.sidebar.slider("Signal", 1, 15, 9, key='macd_signal')

    if "EMA" in strategy_name:
        st.sidebar.markdown("**EMA Settings**")
        ema_short = st.sidebar.slider("Fast EMA", 10, 100, 50, key='ema_short')
        ema_long = st.sidebar.slider("Slow EMA", 100, 300, 200, key='ema_long')

    if "Bollinger" in strategy_name:
        st.sidebar.markdown("**BB Settings**")
        bb_period = st.sidebar.slider("Period", 10, 50, 20, key='bb_period')
        bb_std = st.sidebar.slider("Std Dev", 1.0, 3.0, 2.0, key='bb_std')

    if "Stochastic" in strategy_name:
        st.sidebar.markdown("**Stochastic Settings**")
        stoch_k = st.sidebar.slider("%K", 5, 30, 14, key='stoch_k')
        stoch_d = st.sidebar.slider("%D", 1, 10, 3, key='stoch_d')

    st.sidebar.markdown("---")
    st.sidebar.subheader("Backtesting Parameters")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=100, value=10000, key='capital')
    risk_pct_slider = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, key='risk_pct') 
    risk_pct = risk_pct_slider / 100 
    sl_pips = st.sidebar.number_input("Stop Loss (Pips)", min_value=1, max_value=200, value=50, key='sl_pips')
    tp_pips = st.sidebar.number_input("Take Profit (Pips)", min_value=1, max_value=500, value=100, key='tp_pips') 
    
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    run_backtest_button = col1.button("Run Backtest", type="primary", use_container_width=True)
    if 'backtest_results' in st.session_state:
        if col2.button("Clear Results", use_container_width=True):
            del st.session_state.backtest_results; st.rerun()
    
    if not is_premium:
        st.sidebar.markdown("---"); st.sidebar.info("**⭐ Upgrade to Premium ($20.00/mo):** Unlock all pairs, Scanner & Alerts.")
        if st.sidebar.button("Upgrade to Premium Now!", type="primary", use_container_width=True, key="upgrade_button"):
            st.session_state.page = "profile"; st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Notification Settings")
    telegram_chat_id = st.sidebar.text_input("Your Telegram Chat ID", value=st.session_state.get("telegram_chat_id", ""), key="telegram_chat_id_input")
    
    if st.sidebar.button("Save Telegram ID & Settings", use_container_width=True):
        if st.session_state.user:
            settings_to_save = {"telegram_chat_id": telegram_chat_id}
            try:
                db.child("users").child(user_id).child("settings").set(settings_to_save, st.session_state.user['idToken'])
                st.session_state.telegram_chat_id = telegram_chat_id
                st.sidebar.success("Settings saved!")
            except Exception as e: st.sidebar.error(f"Failed: {e}")

    @st.cache_data(ttl=60)
    def fetch_data(symbol, interval, source="TwelveData", output_size=500):
        if source == "Yahoo":
            yf_map = {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "1h": "1h"}
            if "BTC" in symbol: yf_sym = "BTC-USD"
            elif "XAU" in symbol: yf_sym = "GC=F"
            else: yf_sym = f"{symbol.replace('/', '')}=X"
            try:
                df = yf.download(yf_sym, interval=yf_map.get(interval, "1h"), period="5d", progress=False, auto_adjust=False)
                if df.empty: return pd.DataFrame()
                df = df.reset_index()
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                df.rename(columns={'date': 'datetime', 'index': 'datetime'}, inplace=True)
                df.set_index('datetime', inplace=True)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                # === CRITICAL SORTING FIX ===
                df = df.sort_index()
                return df[['open', 'high', 'low', 'close']].dropna()
            except: return pd.DataFrame()
        return pd.DataFrame()

    def send_telegram_alert(pair, signal_type, entry, tp, sl):
        if "TELEGRAM" not in st.secrets: return
        token = st.secrets["TELEGRAM"].get("BOT_TOKEN")
        chat_id = st.session_state.get("telegram_chat_id")
        if not token or not chat_id: return
        message = f"🚀 *PipWizard Alert*\n*Pair:* {pair}\n*Signal:* {signal_type}\n*Entry:* `{entry}`\n*TP:* `{tp}`\n*SL:* `{sl}`"
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})

    def send_live_alert(pair, signal_type, entry_price, entry_time, tp_price, sl_price):
        if db is None or user_id is None: return
        alert_id = str(uuid.uuid4())
        alert_data = {
            "id": alert_id, "pair": pair, "type": signal_type,
            "entry_price": f"{entry_price:.5f}", "tp_price": f"{tp_price:.5f}", "sl_price": f"{sl_price:.5f}",
            "status": "RUNNING", "entry_time": entry_time.isoformat(),
            "entry_timestamp": int(entry_time.timestamp())
        }
        try:
            db.child("users").child(user_id).child("alerts").child(alert_id).set(alert_data, st.session_state.user['idToken'])
            send_telegram_alert(pair, signal_type, f"{entry_price:.5f}", f"{tp_price:.5f}", f"{sl_price:.5f}")
            st.sidebar.success(f"New {signal_type} Alert on {pair}!")
        except Exception as e: st.sidebar.error(f"Failed: {e}")

    def check_for_live_signal(df, pair, tp_pips, sl_pips):
        if len(df) < 2: return
        latest_bar = df.iloc[-2]
        signal = latest_bar['signal']
        if 'last_alert_time' not in st.session_state: st.session_state.last_alert_time = None
        if signal != 0 and latest_bar.name != st.session_state.last_alert_time:
            st.session_state.last_alert_time = latest_bar.name
            entry_price = df.iloc[-1]['open']
            entry_time = df.iloc[-1].name
            signal_type = "BUY" if signal == 1 else "SELL"
            if "JPY" in pair: PIP = 0.01
            else: PIP = 0.0001
            sl_val = sl_pips * PIP; tp_val = tp_pips * PIP
            if signal_type == "BUY": sl_p = entry_price - sl_val; tp_p = entry_price + tp_val
            else: sl_p = entry_price + sl_val; tp_p = entry_price - tp_val
            send_live_alert(pair, signal_type, entry_price, entry_time, tp_p, sl_p)

    # ... (calculate_indicators & apply_strategy same as before) ...
    def calculate_indicators(df, rsi_p, sma_p, macd_f, macd_sl, macd_sig):
        df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_p)
        df['sma'] = df['close'].rolling(sma_p).mean()
        df['macd_line'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=macd_f, slowperiod=macd_sl, signalperiod=macd_sig)
        df['ema_short'] = talib.EMA(df['close'], timeperiod=ema_short)
        df['ema_long'] = talib.EMA(df['close'], timeperiod=ema_long)
        df['upper_bb'], df['middle_bb'], df['lower_bb'] = talib.BBANDS(df['close'], timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std)
        df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=stoch_k, slowk_period=stoch_d, slowk_matype=0, slowd_period=stoch_d, slowd_matype=0)
        df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
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
            df.loc[buy_cond, 'signal'] = 1; df.loc[sell_cond, 'signal'] = -1
        elif strategy_name == "SMA Crossover Standalone":
            buy_cond = (df['close'] > df['sma']) & (df['close'].shift(1) <= df['sma'].shift(1))
            sell_cond = (df['close'] < df['sma']) & (df['close'].shift(1) >= df['sma'].shift(1))
            df.loc[buy_cond, 'signal'] = 1; df.loc[sell_cond, 'signal'] = -1
        elif strategy_name == "EMA Golden Cross":
            buy_cond = (df['ema_short'] > df['ema_long']) & (df['ema_short'].shift(1) <= df['ema_long'].shift(1))
            sell_cond = (df['ema_short'] < df['ema_long']) & (df['ema_short'].shift(1) >= df['ema_long'].shift(1))
            df.loc[buy_cond, 'signal'] = 1; df.loc[sell_cond, 'signal'] = -1
        elif strategy_name == "Bollinger Bands Bounce":
            df.loc[df['close'] < df['lower_bb'], 'signal'] = 1
            df.loc[df['close'] > df['upper_bb'], 'signal'] = -1
        elif strategy_name == "Stochastic Oscillator":
            buy_cond = (df['slowk'] > df['slowd']) & (df['slowk'].shift(1) <= df['slowd'].shift(1)) & (df['slowk'] < 20)
            sell_cond = (df['slowk'] < df['slowd']) & (df['slowk'].shift(1) >= df['slowd'].shift(1)) & (df['slowk'] > 80)
            df.loc[buy_cond, 'signal'] = 1; df.loc[sell_cond, 'signal'] = -1
        elif strategy_name == "EMA Trend + Price Action":
            uptrend = df['ema_short'] > df['ema_long']
            bullish_pattern = (df['engulfing'] == 100) | (df['hammer'] == 100)
            df.loc[uptrend & bullish_pattern, 'signal'] = 1
            downtrend = df['ema_short'] < df['ema_long']
            bearish_pattern = (df['engulfing'] == -100) | (df['shooting_star'] == -100) 
            df.loc[downtrend & bearish_pattern, 'signal'] = -1
        return df

    def run_backtest(df_in, pair_name, initial_capital, risk_per_trade, sl_pips, tp_pips):
        df = df_in.copy(); trades = []
        if "JPY" in pair_name: PIP = 0.01
        else: PIP = 0.0001
        RISK = sl_pips * PIP; REWARD = tp_pips * PIP
        if sl_pips == 0: return 0, 0, 0, 0, initial_capital, pd.DataFrame(), pd.DataFrame() 
        MAX_RISK = initial_capital * risk_per_trade; REWARD_USD = MAX_RISK * (tp_pips / sl_pips)
        signal_bars = df[df['signal'] != 0]
        for i in range(len(signal_bars)):
            signal_row, signal_type = signal_bars.iloc[i], signal_bars.iloc[i]['signal']
            try: signal_index = df.index.get_loc(signal_row.name)
            except KeyError: continue
            if signal_index + 1 >= len(df): continue 
            entry_bar = df.iloc[signal_index + 1]
            entry_price, entry_time = entry_bar['open'], entry_bar.name
            if signal_type == 1: sl = entry_price - RISK; tp = entry_price + REWARD
            else: sl = entry_price + RISK; tp = entry_price - REWARD
            result, profit_loss, exit_time = 'OPEN', 0.0, None
            for j in range(signal_index + 2, len(df)):
                future_bar = df.iloc[j]
                if signal_type == 1: 
                    if future_bar['low'] <= sl: result, profit_loss, exit_time = 'LOSS', -MAX_RISK, future_bar.name; break
                    elif future_bar['high'] >= tp: result, profit_loss, exit_time = 'WIN', REWARD_USD, future_bar.name; break
                elif signal_type == -1: 
                    if future_bar['high'] >= sl: result, profit_loss, exit_time = 'LOSS', -MAX_RISK, future_bar.name; break
                    elif future_bar['low'] <= tp: result, profit_loss, exit_time = 'WIN', REWARD_USD, future_bar.name; break
            if result == 'OPEN': result, profit_loss, exit_time = 'UNRESOLVED', 0.0, df.iloc[-1].name
            trades.append({"entry_time": entry_time, "exit_time": exit_time, "signal": "BUY" if signal_type == 1 else "SELL", "entry_price": entry_price, "stop_loss": sl, "take_profit": tp, "result": result, "profit_loss": profit_loss})
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

    # === CHART & BACKTEST UI ===
    st.markdown("---"); st.subheader(f"**{selected_pair}** – **{selected_interval}**")
    def show_advanced_chart(symbol):
        if "BTC" in symbol: tv_symbol = "COINBASE:BTCUSD"
        elif "XAU" in symbol: tv_symbol = "OANDA:XAUUSD"
        else: tv_symbol = f"FX:{symbol.replace('/', '')}"
        components.html(f"""<div class="tradingview-widget-container"><div id="tradingview_chart"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{"width": "100%","height": 600,"symbol": "{tv_symbol}","interval": "60","timezone": "Etc/UTC","theme": "dark","style": "1","locale": "en","enable_publishing": false,"allow_symbol_change": true,"studies": ["RSI@tv-basicstudies","MACD@tv-basicstudies"],"container_id": "tradingview_chart"}});</script></div>""", height=620)
    show_advanced_chart(selected_pair)

    with st.spinner(f"Analyzing..."):
         df_analysis = fetch_data(selected_pair, INTERVALS[selected_interval], source="Yahoo")
         if not df_analysis.empty:
             df_ind = calculate_indicators(df_analysis, rsi_period, sma_period, macd_fast, macd_slow, macd_signal)
             df_final = apply_strategy(df_ind, strategy_name, alert_rsi_low, alert_rsi_high)
             check_for_live_signal(df_final, selected_pair, tp_pips, sl_pips)

    if run_backtest_button:
        with st.spinner("Backtesting..."):
            df_bt_data = fetch_data(selected_pair, INTERVALS[selected_interval], source="Yahoo")
            if not df_bt_data.empty:
                df_bt_ind = calculate_indicators(df_bt_data, rsi_period, sma_period, macd_fast, macd_slow, macd_signal)
                df_bt_strat = apply_strategy(df_bt_ind, strategy_name, alert_rsi_low, alert_rsi_high)
                df_backtest = df_bt_strat.dropna()
                total_trades, win_rate, total_profit, pf, final_cap, trade_df, res_df = run_backtest(df_backtest, selected_pair, initial_capital, risk_pct, sl_pips, tp_pips)
                st.session_state.backtest_results = {"total_trades": total_trades, "win_rate": win_rate, "total_profit": total_profit, "profit_factor": pf, "trade_df": trade_df, "resolved_trades_df": res_df, "pair": selected_pair, "interval": selected_interval}
            else: st.error("No data.")
        st.rerun()

    if 'backtest_results' in st.session_state:
        results = st.session_state.backtest_results
        st.markdown("---"); st.subheader("Backtesting Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", results['total_trades'])
        c2.metric("Win Rate", f"{results['win_rate']:.2%}")
        c3.metric("Profit ($)", f"{results['total_profit']:,.2f}")
        c4.metric("Profit Factor", f"{results['profit_factor']:,.2f}")
        if not results['resolved_trades_df'].empty:
            fig = go.Figure(); fig.add_trace(go.Scatter(x=results['resolved_trades_df']['exit_time'], y=results['resolved_trades_df']['equity'], mode='lines', line=dict(color='#26a69a')))
            fig.update_layout(title="Equity Curve", template="plotly_dark"); st.plotly_chart(fig, use_container_width=True)
        st.dataframe(results['trade_df'], width=1000)

    st.markdown("---"); st.subheader("📅 Economic Calendar (Live)")
    components.html("""<div class="tradingview-widget-container"><div class="tradingview-widget-container__widget"></div><script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>{"width": "100%","height": "500","colorTheme": "dark","isTransparent": false,"locale": "en","importanceFilter": "-1,0,1","currencyFilter": "USD,EUR,GBP,JPY,AUD,CAD,CHF,NZD"}</script></div>""", height=520, scrolling=True)

    if is_premium:
        st.markdown("---"); st.subheader("🚀 Strategy Scanner")
        col_scan1, col_scan2, col_scan3 = st.columns(3)
        with col_scan1: scan_pairs = st.multiselect("Pairs", PREMIUM_PAIRS, default=[])
        with col_scan2: scan_intervals = st.multiselect("Timeframes", list(INTERVALS.keys()), default=[])
        with col_scan3: scan_strategies = st.multiselect("Strategies", strategies_list, default=[])
        if st.button("Scan", type="primary", use_container_width=True):
            if not all([scan_pairs, scan_intervals, scan_strategies]): st.error("Select options.")
            else:
                prog = st.progress(0); results = []
                for i, pair in enumerate(scan_pairs):
                    for tf in scan_intervals:
                        data = fetch_data(pair, INTERVALS[tf], source="Yahoo")
                        if data.empty: continue
                        for strat in scan_strategies:
                            ind = calculate_indicators(data.copy(), rsi_period, sma_period, macd_fast, macd_slow, macd_signal)
                            df_s = apply_strategy(ind, strat, alert_rsi_low, alert_rsi_high).dropna()
                            res = run_backtest(df_s, pair, initial_capital, risk_pct, sl_pips, tp_pips)
                            if res[0]>0: results.append({"Pair": pair, "TF": tf, "Strategy": strat, "Profit": res[2], "Win Rate": res[1]})
                    prog.progress((i+1)/len(scan_pairs))
                if results:
                    df_res = pd.DataFrame(results).sort_values("Profit", ascending=False)
                    st.dataframe(df_res.style.format({"Profit": "${:,.2f}", "Win Rate": "{:.2%}"}).background_gradient(subset=['Win Rate'], cmap='RdYlGn'), width=1000)
                else: st.info("No trades found.")

    st.sidebar.markdown("---"); st.sidebar.subheader("Alert History")
    @st.cache_data(ttl=60)
    def load_alerts_from_firebase(uid):
        try: return sorted(db.child("users").child(uid).child("alerts").get(st.session_state.user['idToken']).val().values(), key=lambda x: x['entry_timestamp'], reverse=True)
        except: return []
    
    def update_alert_outcomes(alerts):
        if db is None: return
        for alert in alerts:
            if alert['status'] == 'RUNNING':
                # === LOGIC FIX: SORTING DATA ===
                df_new = fetch_data(alert['pair'], selected_interval, source="Yahoo").sort_index()
                if df_new.empty: continue
                try: 
                    # Ensure timezone awareness match
                    if df_new.index.tz is None: df_new.index = df_new.index.tz_localize('UTC')
                    # Filter only bars AFTER entry
                    entry_time = datetime.fromisoformat(alert['entry_time'])
                    df_future = df_new[df_new.index > entry_time]
                except: continue
                
                new_status = "RUNNING"
                tp = float(alert['tp_price']); sl = float(alert['sl_price'])
                
                for _, bar in df_future.iterrows():
                    if alert['type'] == 'BUY':
                        # STRICT LOGIC CHECK
                        if bar['low'] <= sl: new_status = "LOSS"; break
                        if bar['high'] >= tp: new_status = "WIN"; break
                    elif alert['type'] == 'SELL':
                        if bar['high'] >= sl: new_status = "LOSS"; break
                        if bar['low'] <= tp: new_status = "WIN"; break
                
                if new_status != "RUNNING":
                    db.child("users").child(user_id).child("alerts").child(alert['id']).update({"status": new_status}, st.session_state.user['idToken'])
        st.cache_data.clear()

    alert_list = load_alerts_from_firebase(user_id)
    if st.sidebar.button("Refresh Outcomes", use_container_width=True): update_alert_outcomes(alert_list); st.rerun()
    
    # === CARD UI ===
    if alert_list:
        for alert in alert_list[:10]:
            color = "#26a69a" if alert['type'] == "BUY" else "#ef5350"
            status_color = "#26a69a" if alert['status'] == "WIN" else "#ef5350" if alert['status'] == "LOSS" else "#ff9800"
            st.sidebar.markdown(f"""
            <div class="alert-card">
                <div class="alert-header">
                    <span>{alert['pair']}</span>
                    <span style="color:{color}">{alert['type']}</span>
                    <span style="color:{status_color}">{alert['status']}</span>
                </div>
                <div class="alert-details">
                    Entry: {alert['entry_price']}<br>
                    TP: {alert['tp_price']} | SL: {alert['sl_price']}<br>
                    Time: {datetime.fromisoformat(alert['entry_time']).strftime('%m/%d %H:%M')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else: st.sidebar.info("No alerts yet.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True): logout()

# === 8. ERROR HANDLING ===
elif not st.session_state.user:
    st.set_page_config(page_title="Error", page_icon="🚨", layout="centered")
    st.error("App failed.")
