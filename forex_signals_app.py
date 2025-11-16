import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import talib
from twelvedata import TDClient
import pyrebase  # For Firebase
import json      # For Firebase
import requests  # For Paystack & Telegram
import uuid      # For unique alert IDs
import yfinance as yf # For Unlimited Free Scanning
import xml.etree.ElementTree as ET # For Economic Calendar Parsing

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
        .buy-signal {{ color: #26a69a; }} .sell-signal {{ color: #ef5350; }}
        .alert-history-table {{ width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 0.85em; }}
        .alert-history-table th, .alert-history-table td {{ padding: 4px 2px; text-align: left; border-bottom: 1px solid #444; color: #f0f0f0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .alert-history-table th {{ font-weight: bold; color: #eee; }}
        .alert-status-RUNNING {{ color: #ff9800; font-weight: bold; }}
        .alert-status-PROFIT {{ color: #26a69a; font-weight: bold; }}
        .alert-status-LOSS {{ color: #ef5350; font-weight: bold; }}
        div[data-testid="stMetric"] > label {{ color: #f0f0f0; font-weight: bold; }}
        div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] p {{ color: #f0f0f0; }}
    </style>""", unsafe_allow_html=True)

    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("PipWizard 🧙‍♂️ 📈📉 – Live Signals")
    with col2:
        theme_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
        if st.button(theme_label, key="theme_toggle", on_click=toggle_theme):
            st.rerun()

    with st.expander("👋 Welcome to PipWizard! Click here for a full user guide."):
        st.markdown(
            """
            ### What is PipWizard?
            PipWizard is a tool to help you **test trading strategies** before you use them. 
            
            It is **not** a "get rich quick" bot. It is a decision-support tool that lets you:
            1.  **TEST** your ideas (e.g., "What if I buy when RSI is low?") on *historical data* to see if they would have been profitable.
            2.  **FIND** new strategies by scanning many pairs and timeframes at once.
            3.  **WATCH** your strategy for new signals in real-time.

            ---
            
            ### Tour of the App
            
            **1. The Sidebar (Your Controls)**
            * This is where you set up everything.
            * **Pair & Timeframe:** Choose what you want to analyze.
            * **Strategy:** Pick a strategy from the list (e.g., "RSI Standalone").
            * **Indicator Config:** Set the parameters for your chosen strategy (e.g., RSI Period).
            * **Backtesting Parameters:** Set your risk management rules (Stop Loss, Take Profit, etc.).
            * **Save My Settings:** Click this to save all your sidebar settings to your account.
            * **Alert History:** A new table at the bottom of the sidebar logs all signals and their outcomes.

            **2. The Main Chart (Your "Live" View)**
            * This chart shows you the most recent price data.
            * The "BUY" and "SELL" arrows show you where your **currently selected strategy** has generated signals.
            * **OHLC Data:** Use your mouse crosshair to hover over any candle.

            **3. The Backtesting Report (Your "Test Results")**
            * Click the **"Run Backtest"** button in the sidebar to generate this report.
            * This is the most important feature. It takes your *current* sidebar settings and tests them against the historical data.
            
            **4. The Strategy Scanner (Premium Feature)**
            * Located at the bottom of the page.
            * This is a "backtest of backtests." It uses your **personal sidebar settings** (Capital, SL, TP) to test multiple strategies.

            ---

            ### Feature Tiers
            
            | Feature | 🎁 Free Tier | ⭐ Premium Tier |
            | :--- | :--- | :--- |
            | **Backtesting Engine** | ✅ Yes | ✅ Yes |
            | **All Strategies** | ✅ Yes | ✅ Yes |
            | **Save Settings** | ✅ Yes | ✅ Yes |
            | **Live Signal Alerts** | ✅ EUR/USD Only | ✅ **All Pairs** |
            | **Alert History Log** | ✅ EUR/USD Only | ✅ **All Pairs** |
            | **Currency Pairs** | 🔒 EUR/USD Only | ✅ **All 10+ Pairs** |
            | **🚀 Strategy Scanner**| ❌ No | ✅ **Unlocked** |
            """
        )
    
    st.sidebar.title("PipWizard 🧙‍♂️ 📈📉")
    
    user_id = st.session_state.user['localId']
    user_email = st.session_state.user.get('email', 'User')
    st.sidebar.write(f"Logged in as: `{user_email}`")
    
    is_premium = st.session_state.is_premium

    if is_premium:
        selected_pair = st.sidebar.selectbox("Select Pair", PREMIUM_PAIRS, index=0, key="selected_pair")
        st.sidebar.markdown(f"""
            <div style="background-color: rgba(38, 166, 154, 0.2); border: 1px solid #26a69a; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                <p style="color: #26a69a !important; font-weight: bold; margin: 0; text-align: center;">✨ Premium Active</p>
                <p style="color: #26a69a !important; font-size: 0.8em; margin: 0; text-align: center;">All Features Unlocked</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        selected_pair = FREE_PAIR
        st.sidebar.warning("Free Tier: EUR/USD Only")

    selected_interval = st.sidebar.selectbox("Timeframe", options=list(INTERVALS.keys()), index=3, format_func=lambda x: x.replace("min", " minute").replace("1h", "1 hour"), key="selected_interval")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Strategy Selection")
    strategy_name = st.sidebar.selectbox("Choose a Strategy", ["RSI + SMA Crossover", "MACD Crossover", "RSI + MACD (Confluence)", "SMA + MACD (Confluence)", "RSI Standalone", "SMA Crossover Standalone"], key="strategy_name")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Indicator Configuration")
    # Note: We remove the "Show RSI/MACD" checkboxes because the TradingView widget has them built-in.
    # But we keep the *settings* because they are still used for the Backtester and Scanner.
    
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14, key='rsi_period')
    sma_period = st.sidebar.slider("SMA Period", 10, 50, 20, key='sma_period')
    alert_rsi_low = st.sidebar.slider("Buy RSI <", 20, 40, 35, key='rsi_low')
    alert_rsi_high = st.sidebar.slider("Sell RSI >", 60, 80, 65, key='rsi_high')
    if alert_rsi_low >= alert_rsi_high: st.sidebar.error("RSI Buy threshold must be lower than Sell."); st.stop()
    macd_fast = st.sidebar.slider("MACD Fast Period", 1, 26, 12, key='macd_fast')
    macd_slow = st.sidebar.slider("MACD Slow Period", 13, 50, 26, key='macd_slow')
    macd_signal = st.sidebar.slider("MACD Signal Period", 1, 15, 9, key='macd_signal')
    
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
        st.sidebar.markdown("---")
        st.sidebar.info(
            """
            **⭐ Upgrade to Premium ($20.00/mo):**
            • Unlock all pairs
            • Unlock Strategy Scanner
            • Get Live Signal Alerts
            """
        )
        if st.sidebar.button("Upgrade to Premium Now!", type="primary", use_container_width=True, key="upgrade_button"):
            st.session_state.page = "profile"
            st.rerun()

    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Notification Settings")
    telegram_chat_id = st.sidebar.text_input("Your Telegram Chat ID", 
                                             value=st.session_state.get("telegram_chat_id", ""), 
                                             key="telegram_chat_id_input",
                                             help="Start a chat with @userinfobot on Telegram to get your ID.")
    
    if st.sidebar.button("Save Telegram ID & Settings", use_container_width=True):
        if st.session_state.user:
            settings_to_save = {
                "selected_pair": st.session_state.get("selected_pair", "EUR/USD"),
                "selected_interval": st.session_state.get("selected_interval", "1h"),
                "strategy_name": st.session_state.get("strategy_name", "RSI + SMA Crossover"),
                "rsi_period": st.session_state.get("rsi_period", 14),
                "sma_period": st.session_state.get("sma_period", 20),
                "alert_rsi_low": st.session_state.get("alert_rsi_low", 35),
                "alert_rsi_high": st.session_state.get("alert_rsi_high", 65),
                "macd_fast": st.session_state.get("macd_fast", 12),
                "macd_slow": st.session_state.get("macd_slow", 26),
                "macd_signal": st.session_state.get("macd_signal", 9),
                "capital": st.session_state.get("capital", 10000),
                "risk_pct": st.session_state.get("risk_pct", 1.0), 
                "sl_pips": st.session_state.get("sl_pips", 50),
                "tp_pips": st.session_state.get("tp_pips", 100),
                "telegram_chat_id": telegram_chat_id 
            }
            try:
                db.child("users").child(user_id).child("settings").set(settings_to_save, st.session_state.user['idToken'])
                st.session_state.telegram_chat_id = telegram_chat_id
                st.sidebar.success("Settings saved successfully!")
            except Exception as e:
                st.sidebar.error(f"Failed to save settings: {e}")

    @st.cache_data(ttl=60)
    def fetch_data(symbol, interval, source="TwelveData", output_size=500):
        # NOTE: "TwelveData" source is now technically unused for the main chart, 
        # but we keep it safe in case you want to revert or use it for calculations later.
        if source == "Yahoo":
            yf_map = {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "1h": "1h"}
            yf_sym = f"{symbol.replace('/', '')}=X"
            try:
                df = yf.download(yf_sym, interval=yf_map.get(interval, "1h"), period="5d", progress=False, auto_adjust=False)
                if df.empty: return pd.DataFrame()
                df = df.reset_index()
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                df.rename(columns={'date': 'datetime', 'index': 'datetime'}, inplace=True)
                df.set_index('datetime', inplace=True)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                return df[['open', 'high', 'low', 'close']].dropna()
            except: return pd.DataFrame()
        return pd.DataFrame()

    def send_telegram_alert(pair, signal_type, entry, tp, sl):
        if "TELEGRAM" not in st.secrets: return
        token = st.secrets["TELEGRAM"].get("BOT_TOKEN")
        chat_id = st.session_state.get("telegram_chat_id")
        if not token or not chat_id: return

        emoji = "🟢" if signal_type == "BUY" else "🔴"
        message = f"""
🚀 *PipWizard Alert* 🚀
*Pair:* {pair}
*Signal:* {signal_type} {emoji}
*Entry:* `{entry}`
*TP:* `{tp}`
*SL:* `{sl}`
        """
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        try: requests.post(url, json=payload)
        except Exception as e: print(f"Telegram Error: {e}")

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
        except Exception as e:
            st.sidebar.error(f"Failed to save alert: {e}")

    def check_for_live_signal(df, pair, tp_pips, sl_pips):
        if len(df) < 2: return
        latest_bar = df.iloc[-2]
        signal = latest_bar['signal']
        
        if 'last_alert_time' not in st.session_state: 
            st.session_state.last_alert_time = None
            
        if signal != 0 and latest_bar.name != st.session_state.last_alert_time:
            st.session_state.last_alert_time = latest_bar.name
            entry_price = df.iloc[-1]['open']
            entry_time = df.iloc[-1].name
            signal_type = "BUY" if signal == 1 else "SELL"
            
            if "JPY" in pair: PIP_MULTIPLIER = 0.01
            else: PIP_MULTIPLIER = 0.0001
            sl_value = sl_pips * PIP_MULTIPLIER
            tp_value = tp_pips * PIP_MULTIPLIER

            if signal_type == "BUY":
                sl_price = entry_price - sl_value; tp_price = entry_price + tp_value
            else: 
                sl_price = entry_price + sl_value; tp_price = entry_price - tp_value
            
            send_live_alert(pair, signal_type, entry_price, entry_time, tp_price, sl_price)

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
        return df

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

    # === MAIN CHART (TRADINGVIEW WIDGET) ===
    st.markdown("---")
    st.subheader(f"**{selected_pair}** – **{selected_interval}**")
    
    def show_advanced_chart(symbol):
        # Map format "EUR/USD" -> "FX:EURUSD" for TradingView
        tv_symbol = f"FX:{symbol.replace('/', '')}"
        
        # Advanced Real-Time Chart Widget with RSI and MACD
        html_code = f"""
        <div class="tradingview-widget-container">
          <div id="tradingview_advanced_chart"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
            "width": "100%",
            "height": 600,
            "symbol": "{tv_symbol}",
            "interval": "60",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "studies": [
              "RSI@tv-basicstudies",
              "MACD@tv-basicstudies"
            ],
            "container_id": "tradingview_advanced_chart"
          }}
          );
          </script>
        </div>
        """
        components.html(html_code, height=620)

    # Display the Advanced Widget
    show_advanced_chart(selected_pair)
    
    # We still need to fetch data internally to check for signals (Alert System)
    # But we don't show the static charts anymore.
    # This "invisible" fetch keeps your "Live Alerts" working in the background.
    with st.spinner(f"Analyzing market data for alerts..."):
         df_analysis = fetch_data(selected_pair, INTERVALS[selected_interval], source="Yahoo")
         if not df_analysis.empty:
             df_ind = calculate_indicators(df_analysis, rsi_period, sma_period, macd_fast, macd_slow, macd_signal)
             df_final = apply_strategy(df_ind, strategy_name, alert_rsi_low, alert_rsi_high)
             check_for_live_signal(df_final, selected_pair, tp_pips, sl_pips)

    # === BACKTEST SECTION ===
    if run_backtest_button:
        with st.spinner("Running backtest on real market data..."):
            # USE YAHOO FOR BACKTEST (UNLIMITED)
            df_backtest_data = fetch_data(selected_pair, INTERVALS[selected_interval], source="Yahoo")
            if not df_backtest_data.empty:
                df_bt_ind = calculate_indicators(df_backtest_data, rsi_period, sma_period, macd_fast, macd_slow, macd_signal)
                df_bt_strat = apply_strategy(df_bt_ind, strategy_name, alert_rsi_low, alert_rsi_high)
                df_backtest = df_bt_strat.dropna()
                
                total_trades, win_rate, total_profit, pf, final_cap, trade_df, res_df = run_backtest(
                    df_backtest, selected_pair, initial_capital, risk_pct, sl_pips, tp_pips
                )
                st.session_state.backtest_results = {
                    "total_trades": total_trades, "win_rate": win_rate, "total_profit": total_profit,
                    "profit_factor": pf, "final_capital": final_cap, "trade_df": trade_df,
                    "resolved_trades_df": res_df, "pair": selected_pair, "interval": selected_interval, "data_len": len(df_backtest)
                }
            else:
                 st.error("Could not fetch backtest data from Yahoo.")
        st.rerun()

    # === DISPLAY RESULTS ===
    if 'backtest_results' in st.session_state:
        results = st.session_state.backtest_results
        st.markdown("---"); st.subheader("Backtesting Results (Simulated)")
        st.markdown(f"***Data Tested:*** *{results['pair']}* on *{results['interval']}* interval. *{results['data_len']}* bars.")
        col_t, col_w, col_p, col_f = st.columns(4)
        col_t.metric("Total Trades", results['total_trades'])
        col_w.metric("Win Rate", f"{results['win_rate']:.2%}")
        col_p.metric("Total Profit ($)", f"{results['total_profit']:,.2f}", delta=f"{(results['total_profit']/initial_capital):.2%}")
        col_f.metric("Profit Factor", f"{results['profit_factor']:,.2f}")
        
        st.subheader("Equity Curve")
        resolved_df_key = 'resolved_trades_df' 
        if resolved_df_key in results and not results[resolved_df_key].empty:
            equity_fig = go.Figure()
            equity_fig.add_trace(go.Scatter(x=results[resolved_df_key]['exit_time'], y=results[resolved_df_key]['equity'], mode='lines', name='Equity', line=dict(color='#26a69a')))
            equity_fig.update_layout(xaxis_title="Time", yaxis_title="Account Equity ($)", template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white', height=300)
            st.plotly_chart(equity_fig, use_container_width=True)
        else: st.info("No resolved trades found with these settings.")
        
        st.subheader("Detailed Trade Log")
        trade_df_display = results['trade_df'].copy()
        # Clean Headers
        trade_df_display.columns = [col.replace('_', ' ').title() for col in trade_df_display.columns]
        st.dataframe(trade_df_display, width=1000) 
        
    elif not 'backtest_results' in st.session_state:
        st.markdown("---")
        st.info("Set your parameters in the sidebar and click 'Run Backtest' to see results.")

    # === NEW: NATIVE ECONOMIC CALENDAR (STYLED) ===
    st.markdown("---")
    st.subheader("📅 Economic Calendar (This Week)")
    
    def show_backup_widget():
        calendar_html = """<div class="tradingview-widget-container"><div class="tradingview-widget-container__widget"></div><script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>{ "width": "100%", "height": "500", "colorTheme": "dark", "isTransparent": true, "locale": "en", "importanceFilter": "-1,0,1", "currencyFilter": "USD,EUR,GBP,JPY,AUD,CAD,CHF,NZD" }</script></div>"""
        components.html(calendar_html, height=520, scrolling=True)

    @st.cache_data(ttl=3600)
    def get_native_calendar():
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200: return None
            root = ET.fromstring(response.content)
            data = []
            for event in root.findall('event'):
                country = event.find('country').text
                if country not in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]: continue 
                data.append({
                    "Currency": country,
                    "Event": event.find('title').text,
                    "Impact": event.find('impact').text,
                    "Date": event.find('date').text,
                    "Time": event.find('time').text,
                    "Forecast": event.find('forecast').text if event.find('forecast') is not None else "",
                    "Previous": event.find('previous').text if event.find('previous') is not None else "",
                    "Actual": event.find('actual').text if event.find('actual') is not None else ""
                })
            return pd.DataFrame(data) if data else None
        except Exception: return None

    df_cal = get_native_calendar()
    
    if df_cal is not None and not df_cal.empty:
        # === PANDAS STYLING ===
        def highlight_impact(val):
            if val == 'High': return 'background-color: #ff4b4b; color: white; font-weight: bold;'
            elif val == 'Medium': return 'background-color: #ffa726; color: black; font-weight: bold;'
            elif val == 'Low': return 'background-color: #fff59d; color: black;'
            return ''
        
        # Apply style and render
        st.dataframe(
            df_cal.style.map(highlight_impact, subset=['Impact']),
            column_config={
                "Actual": st.column_config.TextColumn("Actual", help="Released Value"),
            },
            hide_index=True,
            width=1000
        )
        if st.button("Refresh Calendar"): st.cache_data.clear(); st.rerun()
    else:
        show_backup_widget()

    # === STRATEGY SCANNER (PREMIUM) ===
    if is_premium:
        st.markdown(f"""
            <div style="border: 1px solid {'#333' if st.session_state.theme == 'dark' else '#ddd'}; border-radius: 8px; padding: 20px; margin-top: 30px; margin-bottom: 30px; background-color: {'#1a1a1a' if st.session_state.theme == 'dark' else '#fdfdfd'}; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);">
                <h3 style="color: {'#e0e0e0' if st.session_state.theme == 'dark' else '#333'}; margin-top: 0;">
                    🚀 Strategy Scanner <span style="font-size: 0.7em; color: #FFD700;">(Premium Feature)</span>
                </h3>
                <p style="color: {'#bbb' if st.session_state.theme == 'dark' else '#555'}; margin-bottom: 20px;">
                    Uncover the most profitable strategies by backtesting across multiple currency pairs and timeframes, tailored to your risk parameters.
                </p>
        """, unsafe_allow_html=True)

        all_strategies = ["RSI + SMA Crossover", "MACD Crossover", "RSI + MACD (Confluence)", "SMA + MACD (Confluence)", "RSI Standalone", "SMA Crossover Standalone"]
        col_scan1, col_scan2, col_scan3 = st.columns(3)
        with col_scan1: scan_pairs = st.multiselect("Select Currency Pairs", PREMIUM_PAIRS, default=[])
        with col_scan2: scan_intervals = st.multiselect("Select Timeframes", list(INTERVALS.keys()), default=[])
        with col_scan3: scan_strategies = st.multiselect("Select Strategies", all_strategies, default=[])
        
        scan_params = { "rsi_p": rsi_period, "sma_p": sma_period, "macd_f": macd_fast, "macd_sl": macd_slow, "macd_sig": macd_signal, "rsi_l": alert_rsi_low, "rsi_h": alert_rsi_high, "capital": initial_capital, "risk": risk_pct, "sl": sl_pips, "tp": tp_pips }
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Initiate Scan", type="primary", key="scan_button_professional", use_container_width=True): 
            if not all([scan_pairs, scan_intervals, scan_strategies]): st.error("Please select at least one Pair, Timeframe, and Strategy.")
            else:
                total_jobs = len(scan_pairs) * len(scan_intervals) * len(scan_strategies)
                progress_bar = st.progress(0, text=f"Starting Scan... (0/{total_jobs})")
                scan_results = []
                job_count = 0
                results_placeholder = st.empty()
                for pair in scan_pairs:
                    for interval_key in scan_intervals:
                        interval_val = INTERVALS[interval_key]
                        # USE YAHOO FOR SCANNER (UNLIMITED)
                        data = fetch_data(pair, interval_val, source="Yahoo") 
                        if data.empty: total_jobs -= len(scan_strategies); continue
                        for strategy in scan_strategies:
                            job_count += 1
                            progress_bar.progress(job_count / total_jobs, text=f"Analyzing {strategy} on {pair} ({interval_key})... ({job_count}/{total_jobs})")
                            data_with_indicators = calculate_indicators(data.copy(), scan_params["rsi_p"], scan_params["sma_p"], scan_params["macd_f"], scan_params["macd_sl"], scan_params["macd_sig"])
                            data_with_strategy = apply_strategy(data_with_indicators.copy(), strategy, scan_params["rsi_l"], scan_params["rsi_h"])
                            data_clean = data_with_strategy.dropna()
                            if data_clean.empty: continue
                            total_trades, win_rate, total_profit, pf, _, _, _ = run_backtest(data_clean, pair, scan_params["capital"], scan_params["risk"], scan_params["sl"], scan_params["tp"])
                            if total_trades > 0: scan_results.append({"Pair": pair, "Timeframe": interval_key, "Strategy": strategy, "Total Profit ($)": total_profit, "Win Rate (%)": win_rate * 100, "Profit Factor": pf, "Total Trades": total_trades})
                progress_bar.progress(1.0, text="Scan Complete!")
                with results_placeholder.container(): 
                    if scan_results:
                        st.subheader("Scan Results Overview")
                        results_df = pd.DataFrame(scan_results).sort_values(by="Total Profit ($)", ascending=False).reset_index(drop=True)
                        # FIX: Updated width parameter to avoid warning
                        st.dataframe(results_df.style.format({"Total Profit ($)": "${:,.2f}", "Win Rate (%)": "{:.2f}%", "Profit Factor": "{:.2f}"}), width=1000)
                    else: st.info("Scan completed, but no profitable trades were found.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
         st.markdown(f"""
            <div style="border: 1px solid {'#333' if st.session_state.theme == 'dark' else '#ddd'}; border-radius: 8px; padding: 20px; margin-top: 30px; margin-bottom: 30px; background-color: {'#1a1a1a' if st.session_state.theme == 'dark' else '#fdfdfd'}; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);">
                <h3 style="color: {'#e0e0e0' if st.session_state.theme == 'dark' else '#333'}; margin-top: 0;">
                    🚀 Strategy Scanner <span style="font-size: 0.7em; color: #FFD700;">(Premium Feature)</span>
                </h3>
                <p style="color: {'#bbb' if st.session_state.theme == 'dark' else '#555'};">
                    The **Strategy Scanner** is a powerful Premium feature designed to help you discover the most effective trading strategies. Upgrade to unlock!
                </p>
            </div>
         """, unsafe_allow_html=True)
         
         if st.button("Upgrade to Premium Now! (Unlock Scanner)", type="primary", use_container_width=True):
             st.session_state.page = "profile"
             st.rerun()

    # === RISK DISCLAIMER (RESTORED) ===
    st.markdown("---"); st.subheader("⚠️ Risk Disclaimer")
    st.warning(
        """
        **This is a simulation and not financial advice.**
        
        * All backtest results are based on **historical data** and do not guarantee future performance.
        * Forex trading involves substantial risk and is not suitable for every investor.
        * The valuation of currencies may fluctuate, and as a result, clients may lose more than their original investment.
        * This tool is for educational and informational purposes only.
        * Always trade responsibly and use your own risk management plan.
        * Past performance is not indicative of future results.
        """
    )

    # === AUTO-REFRESH ===
    components.html("<meta http-equiv='refresh' content='61'>", height=0)

    # === ALERT HISTORY SECTION (SIDEBAR BOTTOM - RESTORED) ===
    st.sidebar.markdown("---")
    st.sidebar.subheader("Alert History")

    @st.cache_data(ttl=60)
    def load_alerts_from_firebase(user_id):
        try:
            alerts = db.child("users").child(user_id).child("alerts").get(st.session_state.user['idToken']).val()
            if alerts:
                return sorted(alerts.values(), key=lambda x: x['entry_timestamp'], reverse=True)
            return []
        except Exception as e:
            return []

    def update_alert_outcomes(alerts):
        if db is None: return
        with st.spinner("Refreshing alert outcomes..."):
            updated_count = 0
            for alert in alerts:
                if alert['status'] == 'RUNNING':
                    try:
                        alert_time = datetime.fromisoformat(alert['entry_time'])
                        time_diff_seconds = (datetime.now(timezone.utc) - alert_time).total_seconds()
                        interval_map = {"1min": 60, "5min": 300, "15min": 900, "30min": 1800, "1h": 3600}
                        interval_seconds = interval_map.get(selected_interval, 3600)
                        bars_to_fetch = int(time_diff_seconds / interval_seconds) + 2
                        if bars_to_fetch < 2: continue
                        
                        # USE YAHOO FOR OUTCOME CHECK (UNLIMITED)
                        df_new = fetch_data(alert['pair'], selected_interval, source="Yahoo", output_size=bars_to_fetch)
                        if df_new.empty: continue
                        
                        try:
                            if df_new.index.tz is None:
                                df_new.index = df_new.index.tz_localize('UTC')
                            df_future = df_new 
                        except KeyError:
                             continue
                        
                        if df_future.empty: continue
                        new_status = "RUNNING"
                        tp = float(alert['tp_price']); sl = float(alert['sl_price'])
                        for _, bar in df_future.iterrows():
                            if alert['type'] == 'BUY':
                                if bar['low'] <= sl: new_status = "LOSS"; break
                                if bar['high'] >= tp: new_status = "PROFIT"; break
                            elif alert['type'] == 'SELL':
                                if bar['high'] >= sl: new_status = "LOSS"; break
                                if bar['low'] <= tp: new_status = "PROFIT"; break
                        
                        if new_status != "RUNNING":
                            updated_count += 1
                            alert['status'] = new_status
                            db.child("users").child(user_id).child("alerts").child(alert['id']).update({"status": new_status}, st.session_state.user['idToken'])
                    except Exception as e: print(f"Error: {e}")
            if updated_count > 0: st.sidebar.success(f"Updated {updated_count} alert(s)!"); st.cache_data.clear()
            else: st.sidebar.info("No new outcomes found.")

    alert_list = load_alerts_from_firebase(user_id)

    if st.sidebar.button("Refresh Outcomes", use_container_width=True, key="refresh_outcomes_btn"):
        update_alert_outcomes(alert_list)
        st.rerun()

    if alert_list:
        table_html = "<table class='alert-history-table'><tr><th>Time</th><th>Pair</th><th>Type</th><th>Status</th></tr>"
        for alert in alert_list[:10]:
            try: dt = datetime.fromisoformat(alert['entry_time'])
            except ValueError: dt = datetime.now() 
            time_str = dt.strftime('%m/%d %H:%M')
            status_class = f"alert-status-{alert['status']}"
            table_html += f"<tr><td>{time_str}</td><td>{alert['pair']}</td><td>{alert['type']}</td><td><span class='{status_class}'>{alert['status']}</span></td></tr>"
        table_html += "</table>"
        st.sidebar.markdown(table_html, unsafe_allow_html=True)
    else:
        st.sidebar.info("No alerts found yet.")

    # === NEW LOCATION: PROFILE & LOGOUT (AT THE VERY BOTTOM) ===
    st.sidebar.markdown("---")
    if st.sidebar.button("Profile & Logout", use_container_width=True, key="profile_button_bottom"):
        st.session_state.page = "profile"
        st.rerun()

# === 8. ERROR HANDLING ===
elif not st.session_state.user:
    st.set_page_config(page_title="Error - PipWizard", page_icon="🚨", layout="centered")
    st.title("PipWizard 🧙‍♂️ 📈📉")
    st.error("Application failed to initialize.")
