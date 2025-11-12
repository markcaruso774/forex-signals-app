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

# --- NEW LIBRARY ---
from lightweight_charts.widgets import StreamlitChart

# === 1. FIREBASE CONFIGURATION ===
def initialize_firebase():
    """Loads Firebase config from Streamlit Secrets and initializes the app."""
    try:
        if "FIREBASE_CONFIG" not in st.secrets:
            st.error("Firebase config not found in Streamlit Secrets.")
            return None, None
        
        config = st.secrets["FIREBASE_CONFIG"]
        
        if "databaseURL" not in config:
            project_id = config.get('projectId', config.get('project_id'))
            if project_id:
                config["databaseURL"] = f"https{project_id}-default-rtdb.firebaseio.com/"
            else:
                config["databaseURL"] = f"https://{config['authDomain'].split('.')[0]}-default-rtdb.firebaseio.com/"
        
        try:
            firebase = pyrebase.initialize_app(config)
            auth = firebase.auth()
            db = firebase.database()
            return auth, db
        except Exception as e:
            st.error(f"Error initializing Firebase (check config format): {e}")
            return None, None
            
    except Exception as e:
        st.error(f"Error loading Streamlit Secrets: {e}")
        return None, None

auth, db = initialize_firebase()

# === 2. SESSION STATE MANAGEMENT ===
if 'user' not in st.session_state:
    st.session_state.user = None
if 'is_premium' not in st.session_state:
    st.session_state.is_premium = False
if 'page' not in st.session_state:
    st.session_state.page = "login" # Start on 'login'

# === 3. AUTHENTICATION FUNCTIONS ===
def sign_up(email, password):
    """Signs up a new user with Firebase Auth and adds them to Realtime DB."""
    if auth is None or db is None:
        st.error("Auth service not available. Contact support.")
        return
    try:
        user = auth.create_user_with_email_and_password(email, password)
        st.session_state.user = user
        # Create a user profile in the database
        user_data = {"email": email, "subscription_status": "free"}
        db.child("users").child(user['localId']).set(user_data)
        st.session_state.is_premium = False
        st.session_state.page = "app"
        st.rerun()
    except Exception as e:
        error_message = "An unknown error occurred."
        try:
            # Try to parse the JSON error message from Firebase
            error_json = e.args[1]
            error_message = json.loads(error_json).get('error', {}).get('message', error_message)
        except:
            pass # Fallback to default error message
        st.error(f"Failed to create account: {error_message}")

def login(email, password):
    """Logs in an existing user and fetches their subscription status."""
    if auth is None or db is None:
        st.error("Auth service not available. Contact support.")
        return
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.user = user
        
        # Get user's full data from the database
        user_data = db.child("users").child(user['localId']).get().val()
        
        if user_data:
            # 1. Load subscription status
            if user_data.get("subscription_status") == "premium":
                st.session_state.is_premium = True
            else:
                st.session_state.is_premium = False

            # 2. Load saved settings
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
            
            # --- NEW: Load Telegram Chat ID ---
            st.session_state.telegram_chat_id = settings.get("telegram_chat_id", "")
            
        else:
            # Failsafe if user exists in Auth but not DB
            st.session_state.is_premium = False

        st.session_state.page = "app"
        st.rerun()
    except Exception as e:
        error_message = "An unknown error occurred."
        try:
            error_json = e.args[1]
            error_message = json.loads(error_json).get('error', {}).get('message', error_message)
        except:
             pass 
        st.error(f"Login Failed: {error_message}")

def logout():
    """Logs out the user and resets session state."""
    st.session_state.user = None
    st.session_state.is_premium = False
    st.session_state.page = "login"
    st.rerun()

# === 4. PAYSTACK PAYMENT FUNCTIONS ===
def create_payment_link(email, user_id):
    """
    Calls Paystack API to create a one-time payment link.
    """
    test_amount_kobo = 10000 
    
    if "PAYSTACK_TEST" not in st.secrets or "PAYSTACK_SECRET_KEY" not in st.secrets["PAYSTACK_TEST"]:
        st.error("Paystack secret key not configured in Streamlit Secrets.")
        return None, None

    url = "https://api.paystack.co/transaction/initialize"
    headers = {
        "Authorization": f"Bearer {st.secrets['PAYSTACK_TEST']['PAYSTACK_SECRET_KEY']}",
        "Content-Type": "application/json"
    }
    
    if "APP_URL" not in st.secrets or not st.secrets["APP_URL"]:
        st.error("APP_URL is not set in Streamlit Secrets. Cannot create payment link.")
        st.info("Please add `APP_URL = \"https://your-app-name.streamlit.app/\"` to your secrets.")
        return None, None
        
    APP_URL = st.secrets["APP_URL"]

    payload = {
        "email": email,
        "amount": test_amount_kobo, 
        "callback_url": APP_URL, # Redirect back to the main app
        "metadata": {
            "user_id": user_id,
            "user_email": email,
            "description": "PipWizard Premium Subscription (Test)"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        if response_data.get("status"):
            auth_url = response_data["data"]["authorization_url"]
            reference = response_data["data"]["reference"]
            return auth_url, reference
        else:
            st.error(f"Paystack error: {response_data.get('message')}")
            return None, None
    except Exception as e:
        st.error(f"Error creating payment link: {e}")
        return None, None

def verify_payment(reference):
    """
    Calls Paystack to verify a transaction reference.
    """
    if db is None or "PAYSTACK_TEST" not in st.secrets or "PAYSTACK_SECRET_KEY" not in st.secrets["PAYSTACK_TEST"]:
        st.error("Services not initialized.")
        return False

    try:
        url = f"https://api.paystack.co/transaction/verify/{reference}"
        
        headers = {"Authorization": f"Bearer {st.secrets['PAYSTACK_TEST']['PAYSTACK_SECRET_KEY']}"}
        
        response = requests.get(url, headers=headers)
        response_data = response.json()

        if response_data.get("status") and response_data["data"]["status"] == "success":
            st.success("Payment successful! Your account is now Premium.")
            
            metadata = response_data["data"].get("metadata", {})
            user_id = metadata.get("user_id")
            
            if user_id:
                # Update user's status in Firebase
                db.child("users").child(user_id).update({"subscription_status": "premium"})
                st.session_state.is_premium = True
                st.balloons()
                st.session_state.page = "app" # Go back to the main app
                
                try:
                    st.query_params.clear()
                except:
                    pass 
                
                st.rerun()
            else:
                st.error("Could not find user_id in payment metadata. Please contact support.")
            return True
        else:
            st.error("Payment verification failed. Please try again or contact support.")
            return False
            
    except Exception as e:
        st.error(f"Error verifying payment: {e}")
        return False

# === 5. LOGIN/SIGN UP PAGE ===
if st.session_state.page == "login":
    st.set_page_config(page_title="Login - PipWizard", page_icon="💹", layout="centered")

    if auth is None or db is None:
        st.title("PipWizard 💹")
        st.error("Application failed to initialize.")
        st.warning("Could not connect to the authentication service.")
        st.info("This may be due to missing Streamlit Secrets or a Firebase setup issue.")
    else:
        st.title(f"Welcome to PipWizard 💹")
        st.text("Please log in or sign up to continue.")
        action = st.radio("Choose an action:", ("Login", "Sign Up"), horizontal=True, index=1)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if action == "Sign Up":
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.button("Sign Up"):
                if not email or not password or not confirm_password:
                    st.error("Please fill in all fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    sign_up(email, password)
        if action == "Login":
            if st.button("Login"):
                if not email or not password:
                    st.error("Please fill in all fields.")
                else:
                    login(email, password)

# === 6. PROFILE / UPGRADE PAGE ===
elif st.session_state.page == "profile":
    st.set_page_config(page_title="Profile - PipWizard", page_icon="💹", layout="centered")
    
    st.title(f"Profile & Subscription 💹")
    
    if st.session_state.user and 'email' in st.session_state.user:
        st.write(f"Logged in as: `{st.session_state.user['email']}`")
    
    if st.session_state.is_premium:
        st.success("You are a **Premium User**!")
        st.write("All features, pairs, and the Strategy Scanner are unlocked.")
    else:
        st.warning("You are on the **Free Tier**.")
        st.markdown(f"Upgrade to **Premium ($29.99/month)** to unlock all pairs, live alerts, and the Strategy Scanner.")
        
        if st.button("Upgrade to Premium Now! (Test Payment: 100 NGN)", type="primary"):
            with st.spinner("Connecting to Paystack..."):
                user_email = st.session_state.user['email']
                user_id = st.session_state.user['localId']
                
                auth_url, reference = create_payment_link(user_email, user_id) 
                
                if auth_url:
                    st.info("Redirecting you to Paystack to complete your payment...")
                    st.markdown(f'If you are not redirected, [**Click Here to Pay**]({auth_url})', unsafe_allow_html=True)
                    components.html(f'<meta http-equiv="refresh" content="0; url={auth_url}">', height=0)
                else:
                    st.error("Could not initiate payment. Please try again.")

    st.markdown("---")
    if st.button("Back to App"):
        st.session_state.page = "app"
        st.rerun()
        
    if st.button("Logout", type="secondary"):
        logout()

# === 7. MAIN APP PAGE ===
elif st.session_state.page == "app" and st.session_state.user:
    st.set_page_config(page_title="PipWizard", page_icon="💹", layout="wide")

    # --- Check for Payment Callback ---
    query_params = st.query_params
    if "trxref" in query_params:
        reference = query_params["trxref"]
        with st.spinner(f"Verifying your payment ({reference})..."):
            verify_payment(reference)
    # --- End Payment Check ---

    # === CONFIG ===
    ALL_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY", "USD/CHF"]
    FREE_PAIR = "EUR/USD"
    PREMIUM_PAIRS = ALL_PAIRS
    INTERVALS = {"1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min", "1h": "1h"}
    OUTPUTSIZE = 500 # Number of candles to fetch

    # === THEME ===
    if 'theme' not in st.session_state:
        st.session_state.theme = "dark"
    def toggle_theme():
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    def apply_theme():
        dark = st.session_state.theme == "dark"
        # --- CSS FIXES for Metrics, Scanner Labels, and SIDEBAR BACKGROUND ---
        return f"""<style>
            /* FORCE APP & SIDEBAR BACKGROUND TO DARK */
            .stApp {{ background-color: #0e1117; color: #f0f0f0; }}
            section[data-testid="stSidebar"] {{ background-color: #0e1117; }}
            
            /* Fix text colors to ensure visibility */
            .stMarkdown, .stText, p, h1, h2, h3, span, label {{ color: #f0f0f0 !important; }}
            
            .buy-signal {{ color: #26a69a; }} .sell-signal {{ color: #ef5350; }}
            .results-box {{
                border: 1px solid #555;
                border-radius: 5px;
                padding: 10px;
                margin-top: -10px;
                margin-bottom: 10px;
                background-color: #1a1a1a;
            }}
            .results-text {{
                font-size: 0.9em;
                color: #bbb;
            }}
            
            /* Alert History Table Styles - FIXED SPACING & FORMAT */
            .alert-history-table {{
                font-size: 0.85em;
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed; /* Prevents overflow */
            }}
            .alert-history-table th, .alert-history-table td {{
                padding: 4px 2px; /* Reduced padding */
                text-align: left;
                border-bottom: 1px solid #444;
                color: #f0f0f0;
                overflow: hidden;
                text-overflow: ellipsis; /* Handles long text */
                white-space: nowrap;
            }}
            .alert-history-table th {{
                font-weight: bold;
                color: #eee;
            }}
            .alert-status-RUNNING {{ color: #ff9800; font-weight: bold; }}
            .alert-status-PROFIT {{ color: #26a69a; font-weight: bold; }}
            .alert-status-LOSS {{ color: #ef5350; font-weight: bold; }}

            /* Fix for faint metric labels */
            div[data-testid="stMetric"] > label {{
                color: #f0f0f0;
                font-weight: bold;
            }}

            /* Fix for scanner labels */
            div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] p {{
                color: #f0f0f0;
            }}
        </style>"""
    st.markdown(apply_theme(), unsafe_allow_html=True)

    # === HEADER ===
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("PipWizard – Live Forex Signals")
    with col2:
        theme_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
        if st.button(theme_label, key="theme_toggle", on_click=toggle_theme):
            st.rerun()

    # === ABOUT THE APP SECTION (REWRITTEN FOR CLARITY) ===
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
            * **Save My Settings:** Click this to save all your sidebar settings to your account, so they load automatically the next time you log in.
            * **Alert History:** A new table at the bottom of the sidebar that logs all signals and their outcomes. Click **"Refresh Outcomes"** to check the status of running trades.

            **2. The Main Chart (Your "Live" View)**
            * This chart shows you the most recent price data.
            * The "BUY" and "SELL" arrows show you where your **currently selected strategy** has generated signals.
            * **OHLC Data:** Use your mouse crosshair to hover over any candle to see its Open, High, Low, and Close price.
            * This chart automatically refreshes every minute. If a new signal appears, it will be saved to your "Alert History."

            **3. The Backtesting Report (Your "Test Results")**
            * Click the **"Run Backtest"** button in the sidebar to generate this report.
            * This is the most important feature. It takes your *current* sidebar settings and tests them against the historical data.
            * **Note:** The "Data Tested" (e.g., 467 bars) will be less than 500. This is normal. The app correctly removes the first few bars that don't have enough data to calculate indicators (like an SMA).
            * It tells you if your strategy was profitable, its win rate, and shows a full trade-by-trade log.
            * **Use this to test an idea *before* you trust it.**

            **4. The Strategy Scanner (Premium Feature)**
            * Located at the bottom of the page.
            * This is a "backtest of backtests." It now uses your **personal sidebar settings** (Capital, SL, TP) to test multiple strategies, giving you a report that matches your trading style.

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
    
    # === SIDEBAR & CONTROLS ===
    st.sidebar.title("PipWizard")
    
    user_id = st.session_state.user['localId']
    user_email = st.session_state.user.get('email', 'User')
    st.sidebar.write(f"Logged in as: `{user_email}`")
    
    is_premium = st.session_state.is_premium

    if is_premium:
        selected_pair = st.sidebar.selectbox("Select Pair", PREMIUM_PAIRS, index=0, key="selected_pair")
        # Custom Badge for Premium status
        st.sidebar.markdown(f"""
            <div style="background-color: rgba(38, 166, 154, 0.2); border: 1px solid #26a69a; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                <p style="color: #26a69a !important; font-weight: bold; margin: 0; text-align: center;">✨ Premium Active</p>
                <p style="color: #26a69a !important; font-size: 0.8em; margin: 0; text-align: center;">All Features Unlocked</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        selected_pair = FREE_PAIR
        st.sidebar.warning("Free Tier: EUR/USD Only")
        st.sidebar.info("Upgrade to Premium to unlock all pairs and the Strategy Scanner!")

    selected_interval = st.sidebar.selectbox("Timeframe", options=list(INTERVALS.keys()), index=3, format_func=lambda x: x.replace("min", " minute").replace("1h", "1 hour"), key="selected_interval")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Strategy Selection")
    strategy_name = st.sidebar.selectbox("Choose a Strategy", ["RSI + SMA Crossover", "MACD Crossover", "RSI + MACD (Confluence)", "SMA + MACD (Confluence)", "RSI Standalone", "SMA Crossover Standalone"], key="strategy_name")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Indicator Configuration")
    show_rsi = st.sidebar.checkbox("Show RSI Chart", value=True)
    show_macd = st.sidebar.checkbox("Show MACD Chart", value=True)
    st.sidebar.markdown("**RSI / SMA (Signal)**")
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14, key='rsi_period')
    sma_period = st.sidebar.slider("SMA Period", 10, 50, 20, key='sma_period')
    alert_rsi_low = st.sidebar.slider("Buy RSI <", 20, 40, 35, key='rsi_low')
    alert_rsi_high = st.sidebar.slider("Sell RSI >", 60, 80, 65, key='rsi_high')
    if alert_rsi_low >= alert_rsi_high: st.sidebar.error("RSI Buy threshold must be lower than Sell."); st.stop()
    st.sidebar.markdown("**MACD (Confirmation)**")
    macd_fast = st.sidebar.slider("MACD Fast Period", 1, 26, 12, key='macd_fast')
    macd_slow = st.sidebar.slider("MACD Slow Period", 13, 50, 26, key='macd_slow')
    macd_signal = st.sidebar.slider("MACD Signal Period", 1, 15, 9, key='macd_signal')
    if macd_fast >= macd_slow: st.sidebar.error("MACD Fast Period must be shorter than Slow."); st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Backtesting Parameters")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=100, value=10000, key='capital')
    risk_pct_slider = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, key='risk_pct') 
    risk_pct = risk_pct_slider / 100 # Convert to decimal for backtest
    
    sl_pips = st.sidebar.number_input("Stop Loss (Pips)", min_value=1, max_value=200, value=50, key='sl_pips')
    tp_pips = st.sidebar.number_input("Take Profit (Pips)", min_value=1, max_value=500, value=100, key='tp_pips') 
    
    if sl_pips <= 0 or tp_pips <= 0: st.sidebar.error("SL and TP must be greater than 0."); st.stop()
    
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    run_backtest_button = col1.button("Run Backtest", type="primary", use_container_width=True)
    if 'backtest_results' in st.session_state:
        if col2.button("Clear Results", use_container_width=True):
            del st.session_state.backtest_results; st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"""
        **🎁 Free Tier:**\n
        Full backtesting & live alerts on EUR/USD only.

        **⭐ Upgrade to Premium ($29.99/mo):**\n
        • Unlock all pairs\n
        • Unlock Strategy Scanner\n
        • Get Live Signal Alerts (for all pairs)
        """
    )
    if not is_premium:
        if st.sidebar.button("Upgrade to Premium Now!", type="primary", use_container_width=True, key="upgrade_button"):
            st.session_state.page = "profile"
            st.rerun()

    st.sidebar.markdown("---")
    
    # --- NEW: Telegram Chat ID Input ---
    st.sidebar.subheader("Notification Settings")
    telegram_chat_id = st.sidebar.text_input("Your Telegram Chat ID", 
                                             value=st.session_state.get("telegram_chat_id", ""), 
                                             key="telegram_chat_id_input",
                                             help="Start a chat with @userinfobot on Telegram to get your ID.")
    
    if st.button("Save My Settings", use_container_width=True):
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
                # --- NEW: Save the Telegram Chat ID ---
                "telegram_chat_id": telegram_chat_id 
            }
            try:
                db.child("users").child(user_id).child("settings").set(settings_to_save)
                # --- NEW: Update session state immediately ---
                st.session_state.telegram_chat_id = telegram_chat_id
                st.sidebar.success("Settings saved successfully!")
            except Exception as e:
                st.sidebar.error(f"Failed to save settings: {e}")

    if st.sidebar.button("Profile & Logout", use_container_width=True, key="profile_button"):
        st.session_state.page = "profile"
        st.rerun()
    
    # === HELPER FUNCTIONS (Alerts & Data) ===
    
    @st.cache_data(ttl=60) # Cache for 60 seconds
    def fetch_data(symbol, interval, output_size=OUTPUTSIZE):
        """Fetches candle data from Twelve Data."""
        if "TD_API_KEY" not in st.secrets:
            st.error("TD_API_KEY not found in Streamlit Secrets."); return pd.DataFrame()
        td = TDClient(apikey=st.secrets["TD_API_KEY"])
        try:
            ts = td.time_series(symbol=symbol, interval=interval, outputsize=output_size).as_pandas()
            if ts is None or ts.empty:
                st.error(f"No data returned for {symbol}."); return pd.DataFrame()
            df = ts[['open', 'high', 'low', 'close']].copy()
            df.index = pd.to_datetime(df.index)
            return df.iloc[::-1] # Reverse to get ascending time
        except Exception as e:
            st.error(f"API Error fetching {symbol}: {e}"); return pd.DataFrame()

    # --- TELEGRAM ALERT FUNCTION (UPDATED) ---
    def send_telegram_alert(pair, signal_type, entry, tp, sl):
        """Sends a structured alert message to Telegram."""
        if "TELEGRAM" not in st.secrets:
            return # Skip if no Bot Token is configured
            
        token = st.secrets["TELEGRAM"].get("BOT_TOKEN")
        
        # --- NEW: Get the user's specific Chat ID from session state ---
        chat_id = st.session_state.get("telegram_chat_id")
        
        if not token or not chat_id:
            # Don't send if bot or *user's* chat ID is missing
            return

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
        payload = {
            "chat_id": chat_id, 
            "text": message, 
            "parse_mode": "Markdown"
        }
        
        try:
            requests.post(url, json=payload)
        except Exception as e:
            print(f"Telegram Error: {e}") # Log to console, don't interrupt app

    # --- NEW: DETAILED ALERT FUNCTION ---
    def send_live_alert(pair, signal_type, entry_price, entry_time, tp_price, sl_price):
        """Saves a new alert to Firebase and sends via Telegram."""
        if db is None or user_id is None:
            return

        alert_id = str(uuid.uuid4())
        alert_data = {
            "id": alert_id,
            "pair": pair,
            "type": signal_type,
            "entry_price": f"{entry_price:.5f}",
            "tp_price": f"{tp_price:.5f}",
            "sl_price": f"{sl_price:.5f}",
            "status": "RUNNING",
            "entry_time": entry_time.isoformat(),
            "entry_timestamp": int(entry_time.timestamp()) # For sorting
        }
        
        try:
            # Save to Firebase
            db.child("users").child(user_id).child("alerts").child(alert_id).set(alert_data)
            
            # Send to Telegram
            send_telegram_alert(pair, signal_type, f"{entry_price:.5f}", f"{tp_price:.5f}", f"{sl_price:.5f}")
            
            # Show sidebar success
            st.sidebar.success(f"New {signal_type} Alert on {pair}!")
            st.sidebar.markdown(f"""
            - **Entry:** `{entry_price:.5f}`
            - **TP:** `{tp_price:.5f}`
            - **SL:** `{sl_price:.5f}`
            """)
        except Exception as e:
            st.sidebar.error(f"Failed to save alert: {e}")

    # --- NEW: ALERT CHECKER ---
    def check_for_live_signal(df, pair, tp_pips, sl_pips):
        """Checks the latest bar for a new signal and triggers an alert."""
        if len(df) < 2: return
        
        # Check the *second to last* bar for a signal
        latest_bar = df.iloc[-2]
        signal = latest_bar['signal']
        
        if 'last_alert_time' not in st.session_state: 
            st.session_state.last_alert_time = None
            
        # Check if signal is new and hasn't been processed
        if signal != 0 and latest_bar.name != st.session_state.last_alert_time:
            st.session_state.last_alert_time = latest_bar.name
            
            # Get entry price from the *current* bar's open
            entry_price = df.iloc[-1]['open']
            entry_time = df.iloc[-1].name # Timestamp of the current bar
            signal_type = "BUY" if signal == 1 else "SELL"
            
            # Calculate TP/SL
            if "JPY" in pair: PIP_MULTIPLIER = 0.01
            else: PIP_MULTIPLIER = 0.0001
            
            sl_value = sl_pips * PIP_MULTIPLIER
            tp_value = tp_pips * PIP_MULTIPLIER

            if signal_type == "BUY":
                sl_price = entry_price - sl_value
                tp_price = entry_price + tp_value
            else: # SELL
                sl_price = entry_price + sl_value
                tp_price = entry_price - tp_value
            
            # Send the detailed alert
            send_live_alert(pair, signal_type, entry_price, entry_time, tp_price, sl_price)

    # === INDICATOR & STRATEGY LOGIC ===
    
    def calculate_indicators(df, rsi_p, sma_p, macd_f, macd_sl, macd_sig):
        """Calculates all indicators and adds them to the DataFrame."""
        df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_p)
        df['sma'] = df['close'].rolling(sma_p).mean()
        df['macd_line'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=macd_f, slowperiod=macd_sl, signalperiod=macd_sig)
        return df

    def apply_strategy(df, strategy_name, rsi_l, rsi_h):
        """Applies the selected trading strategy logic to the DataFrame."""
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

    # === BACKTESTING FUNCTION ===
    def run_backtest(df_in, pair_name, initial_capital, risk_per_trade, sl_pips, tp_pips):
        """Runs the vector-based backtest on the strategy-applied DataFrame."""
        df = df_in.copy(); trades = []
        
        if "JPY" in pair_name: PIP_MULTIPLIER = 0.01
        else: PIP_MULTIPLIER = 0.0001
        
        RISK_PIPS_VALUE = sl_pips * PIP_MULTIPLIER
        REWARD_PIPS_VALUE = tp_pips * PIP_MULTIPLIER
        
        # Check for division by zero
        if sl_pips == 0: return 0, 0, 0, 0, initial_capital, pd.DataFrame(), pd.DataFrame() 
        
        MAX_RISK_USD = initial_capital * risk_per_trade
        REWARD_USD = MAX_RISK_USD * (tp_pips / sl_pips)
        
        signal_bars = df[df['signal'] != 0]
        
        for i in range(len(signal_bars)):
            signal_row, signal_type = signal_bars.iloc[i], signal_bars.iloc[i]['signal']
            
            try: signal_index = df.index.get_loc(signal_row.name)
            except KeyError: continue
            
            if signal_index + 1 >= len(df): continue # Signal on last bar
            
            entry_bar = df.iloc[signal_index + 1]
            entry_price, entry_time = entry_bar['open'], entry_bar.name
            
            if signal_type == 1: # BUY
                stop_loss = entry_price - RISK_PIPS_VALUE
                take_profit = entry_price + REWARD_PIPS_VALUE
            else: # SELL
                stop_loss = entry_price + RISK_PIPS_VALUE
                take_profit = entry_price - REWARD_PIPS_VALUE
                
            result, profit_loss, exit_time = 'OPEN', 0.0, None
            
            for j in range(signal_index + 2, len(df)):
                future_bar = df.iloc[j]
                if signal_type == 1: # BUY
                    if future_bar['low'] <= stop_loss: 
                        result, profit_loss, exit_time = 'LOSS', -MAX_RISK_USD, future_bar.name; break
                    elif future_bar['high'] >= take_profit: 
                        result, profit_loss, exit_time = 'WIN', REWARD_USD, future_bar.name; break
                elif signal_type == -1: # SELL
                    if future_bar['high'] >= stop_loss: 
                        result, profit_loss, exit_time = 'LOSS', -MAX_RISK_USD, future_bar.name; break
                    elif future_bar['low'] <= take_profit: 
                        result, profit_loss, exit_time = 'WIN', REWARD_USD, future_bar.name; break
                        
            if result == 'OPEN': 
                result, profit_loss, exit_time = 'UNRESOLVED', 0.0, df.iloc[-1].name
                
            trades.append({"entry_time": entry_time, "exit_time": exit_time, "signal": "BUY" if signal_type == 1 else "SELL", "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit, "result": result, "profit_loss": profit_loss})
        
        if not trades: return 0, 0, 0, 0, initial_capital, pd.DataFrame(), pd.DataFrame() 
        
        trade_log = pd.DataFrame(trades).set_index('entry_time')
        resolved_trades = trade_log[trade_log['result'].isin(['WIN', 'LOSS'])].copy()
        
        if resolved_trades.empty: return 0, 0, 0, 0, initial_capital, trade_log, resolved_trades
            
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
        st.error("Failed to load data. The API might be down or your key is invalid."); st.stop()
    
    with st.spinner("Calculating indicators..."):
        df_indicators = calculate_indicators(df, rsi_period, sma_period, macd_fast, macd_slow, macd_signal)
    
    # --- MARKER BUG FIX: Apply strategy to the *full* dataframe (with NaNs) ---
    with st.spinner(f"Applying Strategy: {strategy_name}..."):
        df_final = apply_strategy(df_indicators.copy(), strategy_name, alert_rsi_low, alert_rsi_high)
    # --- End of Bug Fix ---
        
    # === RUN MAIN BACKTESTING ON BUTTON CLICK ===
    if run_backtest_button:
        with st.spinner("Running backtest on real market data..."):
            # We must dropna() *before* backtesting, as it can't handle NaNs
            df_backtest = df_final.dropna()
            total_trades, win_rate, total_profit, pf, final_cap, trade_df, res_df = run_backtest(
                df_backtest, selected_pair, initial_capital, risk_pct, sl_pips, tp_pips
            )
            st.session_state.backtest_results = {
                "total_trades": total_trades, "win_rate": win_rate, "total_profit": total_profit,
                "profit_factor": pf, "final_capital": final_cap, "trade_df": trade_df,
                "resolved_trades_df": res_df, "pair": selected_pair, "interval": selected_interval, "data_len": len(df_backtest)
            }
        st.rerun()

    # === DISPLAY MAIN BACKTESTING IF RESULTS EXIST ===
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
        
        # --- NEW: Capitalize trade log headers ---
        trade_df_display = results['trade_df'].copy()
        trade_df_display.columns = [col.replace('_', ' ').title() for col in trade_df_display.columns]
        st.dataframe(trade_df_display, width='stretch') 
        
    elif not 'backtest_results' in st.session_state:
        st.markdown("---")
        st.info("Set your parameters in the sidebar and click 'Run Backtest' to see results.")

    # === MAIN CHART (LIGHTWEIGHT CHARTS) ===
    st.markdown("---")
    st.subheader(f"**{selected_pair}** – **{selected_interval}** – Last {len(df_final)} Candles")
    
    chart_theme = 'dark' if st.session_state.theme == 'dark' else 'light'
    
    chart = StreamlitChart(width=1000, height=500)
    
    chart.layout_options = {
        "backgroundColor": "#0e1117" if chart_theme == 'dark' else "#ffffff",
        "textColor": "#f0f0f0" if chart_theme == 'dark' else "#212529",
    }
    chart.grid_options = {
        "vertLines": {"color": "#444" if chart_theme == 'dark' else "#ddd"},
        "horzLines": {"color": "#444" if chart_theme == 'dark' else "#ddd"},
    }
    chart.price_scale_options = {"borderColor": "#777"}
    chart.time_scale_options = {"borderColor": "#777"}
    
    # --- Enable crosshair (Note: Does not show OHLC box) ---
    chart.crosshair_options = {
        "mode": 1, # 0=Normal, 1=Magnet
        "vertLine": {"color": "#C0C0C0", "style": 2, "width": 1},
        "horzLine": {"color": "#C0C0C0", "style": 2, "width": 1}
    }

    # 1. PREPARE THE DATA
    # We use df_final here, which has the signals
    df_reset = df_final.reset_index()
    index_col_name = df_reset.columns[0]
    
    # --- RESTORED FIX: Convert back to Unix Timestamp (Int) ---
    # This fixes the "vertical line" chart bug.
    df_reset['time'] = df_reset[index_col_name].apply(lambda x: int(x.timestamp()))
    
    df_chart = df_reset[['time', 'open', 'high', 'low', 'close']]
    sma_data = df_reset[['time', 'sma']].dropna() 
    
    buy_signals = df_final[df_final['signal'] == 1].reset_index()
    sell_signals = df_final[df_final['signal'] == -1].reset_index()
    
    buy_index_col = buy_signals.columns[0]
    sell_index_col = sell_signals.columns[0]
    
    # --- RESTORED FIX: Convert markers to Unix Timestamp (Int) as well ---
    buy_signals['time'] = buy_signals[buy_index_col].apply(lambda x: int(x.timestamp()))
    sell_signals['time'] = sell_signals[sell_index_col].apply(lambda x: int(x.timestamp()))

    buy_markers = [
        {"time": row['time'], "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "BUY"}
        for _, row in buy_signals.iterrows()
    ]
    sell_markers = [
        {"time": row['time'], "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "SELL"}
        for _, row in sell_signals.iterrows()
    ]
    
    # 2. LOAD DATA INTO THE CHART
    chart.set(df_chart)
    
    sma_line = chart.create_line(
        name="sma",  
        color="#ff9800",
        width=2
    )
    sma_line.set(sma_data)
    
    chart.markers = buy_markers + sell_markers

    # 3. RENDER THE CHART
    chart.load()

    # --- SUBPLOTS (RSI / MACD) ---
    fig_subplots = make_subplots(
        rows=2 if show_rsi and show_macd else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1, # <-- COMPRESSED CHART FIX
        row_heights=[0.5, 0.5] if show_rsi and show_macd else [1.0]
    )
    
    # --- COMPRESSED CHART FIX: Calculate num_subplots ---
    num_subplots = (1 if show_rsi else 0) + (1 if show_macd else 0)
    current_row = 1
    if show_rsi:
        # --- FIX: Use df_final, which is the full 500-row df ---
        fig_subplots.add_trace(go.Scatter(x=df_final.index, y=df_final['rsi'], name=f"RSI({rsi_period})", line=dict(color="#9c27b0")), row=current_row, col=1)
        fig_subplots.add_hline(y=alert_rsi_high, line_dash="dash", line_color="#ef5350", annotation_text=f"Overbought ({alert_rsi_high})", row=current_row, col=1)
        fig_subplots.add_hline(y=alert_rsi_low, line_dash="dash", line_color="#26a69a", annotation_text=f"Oversold ({alert_rsi_low})", row=current_row, col=1)
        fig_subplots.add_hline(y=50, line_dash="dot", line_color="#cccccc", row=current_row, col=1)
        fig_subplots.update_yaxes(title_text=f"RSI({rsi_period})", range=[0, 100], row=current_row, col=1)
        current_row += 1
        
    if show_macd:
        # --- FIX: Use df_final, which is the full 500-row df ---
        fig_subplots.add_trace(go.Scatter(x=df_final.index, y=df_final['macd_line'], name='MACD', line=dict(color='#2196f3')), row=current_row, col=1)
        fig_subplots.add_trace(go.Scatter(x=df_final.index, y=df_final['macd_signal'], name='Signal', line=dict(color='#ff9800')), row=current_row, col=1)
        colors = ['#26a69a' if val >= 0 else '#ef5350' for val in df_final['macd_hist']]
        fig_subplots.add_trace(go.Bar(x=df_final.index, y=df_final['macd_hist'], name='Histogram', marker_color=colors), row=current_row, col=1)
        fig_subplots.update_yaxes(title_text="MACD", row=current_row, col=1)
        fig_subplots.add_hline(y=0, line_dash="dot", line_color="#cccccc", row=current_row, col=1)
        
    if show_rsi or show_macd:
        # --- COMPRESSED CHART FIX: Check num_subplots before updating layout ---
        if num_subplots > 0:
            fig_subplots.update_layout(
                # --- COMPRESSED CHART FIX: Use num_subplots * 300 ---
                height=300 * num_subplots, 
                template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white',
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_subplots, use_container_width=True, config={'displayModeBar': False})

    # === LIVE SIGNAL ALERT CHECK ===
    # --- FIX: Pass df_final (full 500 rows) to the check function ---
    check_for_live_signal(df_final, selected_pair, tp_pips, sl_pips)

    # --- ECONOMIC CALENDAR SECTION (REMOVED) ---
    st.markdown("---")
    
    # === STRATEGY SCANNER (PREMIUM FEATURE) ===
    if is_premium:
        # Custom "Classic" Professional Container
        st.markdown(f"""
            <div style="
                border: 1px solid {'#333' if st.session_state.theme == 'dark' else '#ddd'};
                border-radius: 8px;
                padding: 20px;
                margin-top: 30px;
                margin-bottom: 30px;
                background-color: {'#1a1a1a' if st.session_state.theme == 'dark' else '#fdfdfd'};
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            ">
                <h3 style="color: {'#e0e0e0' if st.session_state.theme == 'dark' else '#333'}; margin-top: 0;">
                    🚀 Strategy Scanner <span style="font-size: 0.7em; color: #FFD700;">(Premium Feature)</span>
                </h3>
                <p style="color: {'#bbb' if st.session_state.theme == 'dark' else '#555'}; margin-bottom: 20px;">
                    Uncover the most profitable strategies by backtesting across multiple currency pairs and timeframes, tailored to your risk parameters.
                </p>
        """, unsafe_allow_html=True)

        # --- CRASH FIX: Corrected strategy names ---
        all_strategies = [
            "RSI + SMA Crossover",
            "MACD Crossover",
            "RSI + MACD (Confluence)",
            "SMA + MACD (Confluence)",
            "RSI Standalone",
            "SMA Crossover Standalone"
        ]
        
        # Use columns for a cleaner layout
        col_scan1, col_scan2, col_scan3 = st.columns(3)
        with col_scan1:
            # --- NEW: default=[] ---
            scan_pairs = st.multiselect("Select Currency Pairs", PREMIUM_PAIRS, default=[], help="Choose the forex pairs to include in the scan.")
        with col_scan2:
            # --- NEW: default=[] ---
            scan_intervals = st.multiselect("Select Timeframes", list(INTERVALS.keys()), default=[], help="Select the timeframes for strategy evaluation.")
        with col_scan3:
            # --- NEW: default=[] ---
            scan_strategies = st.multiselect("Select Strategies", all_strategies, default=[], help="Pick the strategies you wish to test.")
        
        # --- NEW: Use sidebar settings for the scan ---
        scan_params = {
            "rsi_p": rsi_period, "sma_p": sma_period, 
            "macd_f": macd_fast, "macd_sl": macd_slow, "macd_sig": macd_signal, 
            "rsi_l": alert_rsi_low, "rsi_h": alert_rsi_high, 
            "capital": initial_capital, "risk": risk_pct, 
            "sl": sl_pips, "tp": tp_pips
        }
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Initiate Scan", type="primary", key="scan_button_professional", use_container_width=True): 
            if not all([scan_pairs, scan_intervals, scan_strategies]):
                st.error("Please select at least one Pair, Timeframe, and Strategy to begin the scan.")
            else:
                total_jobs = len(scan_pairs) * len(scan_intervals) * len(scan_strategies)
                progress_bar = st.progress(0, text=f"Starting Scan... (0/{total_jobs})")
                scan_results = []
                job_count = 0
                
                # Use placeholder for results
                results_placeholder = st.empty()

                for pair in scan_pairs:
                    for interval_key in scan_intervals:
                        interval_val = INTERVALS[interval_key]
                        # Use the main fetch_data function
                        data = fetch_data(pair, interval_val) 
                        if data.empty:
                            with results_placeholder.container():
                                st.warning(f"Could not fetch data for {pair} ({interval_key}). Skipping.")
                            total_jobs -= len(scan_strategies); continue
                        for strategy in scan_strategies:
                            job_count += 1
                            progress_bar.progress(job_count / total_jobs, text=f"Analyzing {strategy} on {pair} ({interval_key})... ({job_count}/{total_jobs})")
                            
                            # Process data
                            data_with_indicators = calculate_indicators(data.copy(), scan_params["rsi_p"], scan_params["sma_p"], scan_params["macd_f"], scan_params["macd_sl"], scan_params["macd_sig"])
                            data_with_strategy = apply_strategy(data_with_indicators.copy(), strategy, scan_params["rsi_l"], scan_params["rsi_h"])
                            
                            # --- BUG FIX: dropna() *after* strategy for backtest ---
                            data_clean = data_with_strategy.dropna()
                            if data_clean.empty: continue
                            # --- End of fix ---
                            
                            total_trades, win_rate, total_profit, pf, _, _, _ = run_backtest(
                                data_clean, pair, scan_params["capital"], scan_params["risk"],
                                scan_params["sl"], scan_params["tp"]
                            )
                            
                            if total_trades > 0:
                                scan_results.append({"Pair": pair, "Timeframe": interval_key, "Strategy": strategy, "Total Profit ($)": total_profit, "Win Rate (%)": win_rate * 100, "Profit Factor": pf, "Total Trades": total_trades})
                
                progress_bar.progress(1.0, text="Scan Complete!")
                
                with results_placeholder.container(): 
                    if scan_results:
                        st.subheader("Scan Results Overview")
                        results_df = pd.DataFrame(scan_results).sort_values(by="Total Profit ($)", ascending=False).reset_index(drop=True)
                        
                        def style_profit(val):
                            color = '#26a69a' if val > 0 else '#ef5350' if val < 0 else ( '#f0f0f0' if st.session_state.theme == 'dark' else '#212529' ); return f'color: {color}; font-weight: bold;'
                        
                        def style_win_rate(val):
                            if pd.isna(val):
                                return 'background-color: #333; color: #888;' 
                            val = max(0, min(100, val)); color = 'white' if st.session_state.theme == 'dark' else 'black'
                            if val < 50: return f'background-color: rgba(239, 83, 80, {0.2 + (1 - (val/50))*0.6}); color: {color};'
                            else: return f'background-color: rgba(38, 166, 154, {0.2 + ((val-50)/50)*0.6}); color: {color};'
                        
                        def style_profit_factor(val):
                            color = '#26a69a' if val >= 1.0 else '#ef5350'; return f'color: {color}; font-weight: bold;'
                        
                        st.dataframe(
                            results_df.style
                                .applymap(style_profit, subset=['Total Profit ($)'])
                                .apply(lambda x: [style_win_rate(v) for v in x], subset=['Win Rate (%)'])
                                .applymap(style_profit_factor, subset=['Profit Factor'])
                                .format({"Total Profit ($)": "${:,.2f}", "Win Rate (%)": "{:.2f}%", "Profit Factor": "{:.2f}"}),
                            width='stretch'
                        )
                    else:
                        st.info("Scan completed, but no profitable trades were found. Consider adjusting your settings.")
        
        st.markdown("</div>", unsafe_allow_html=True) # Close custom container
    else:
         st.markdown(f"""
            <div style="
                border: 1px solid {'#333' if st.session_state.theme == 'dark' else '#ddd'};
                border-radius: 8px;
                padding: 20px;
                margin-top: 30px;
                margin-bottom: 30px;
                background-color: {'#1a1a1a' if st.session_state.theme == 'dark' else '#fdfdfd'};
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            ">
                <h3 style="color: {'#e0e0e0' if st.session_state.theme == 'dark' else '#333'}; margin-top: 0;">
                    🚀 Strategy Scanner <span style="font-size: 0.7em; color: #FFD700;">(Premium Feature)</span>
                </h3>
                <p style="color: {'#bbb' if st.session_state.theme == 'dark' else '#555'};">
                    The **Strategy Scanner** is a powerful Premium feature designed to help you discover the most effective trading strategies.
                </p>
                <p style="color: {'#bbb' if st.session_state.theme == 'dark' else '#555'};">
                    Upgrade to Premium to unlock this feature and supercharge your analysis!
                </p>
                <div style="text-align: center; margin-top: 20px;">
                    <a href="#" onclick="window.parent.location.href = '{st.secrets.get("APP_URL", "")}?page=profile'" target="_self" style="
                        background-color: #007bff;
                        color: white;
                        padding: 10px 20px;
                        border-radius: 5px;
                        text-decoration: none;
                        font-weight: bold;
                    ">Upgrade to Premium Now!</a>
                </div>
            </div>
         """, unsafe_allow_html=True)

    # === RISK DISCLAIMER (REWRITTEN FOR CLARITY) ===
    st.markdown("---")
    st.subheader("⚠️ Risk Disclaimer")
    st.warning(
        """
        **This is a simulation and not financial advice.**
        * All backtest results are based on **historical data** and do not guarantee future performance.
        * Forex trading is extremely risky and can result **in the loss** of your entire capital.
        * This tool is for educational and informational purposes only.
        * Always trade responsibly and use your own risk management plan.
        """
    )

    # === AUTO-REFRESH COMPONENT ===
    components.html("<meta http-equiv='refresh' content='61'>", height=0)

    # === NEW: ALERT HISTORY SECTION (Sidebar Bottom) ===
    st.sidebar.markdown("---")
    st.sidebar.subheader("Alert History")

    @st.cache_data(ttl=60) # Cache alerts for 60s
    def load_alerts_from_firebase(user_id):
        """Loads all alerts for the user from Firebase."""
        try:
            alerts = db.child("users").child(user_id).child("alerts").get().val()
            if alerts:
                # Convert dict of alerts into a sorted list
                alerts_list = sorted(alerts.values(), key=lambda x: x['entry_timestamp'], reverse=True)
                return alerts_list
            return []
        except Exception as e:
            st.error(f"Error loading alerts: {e}")
            return []

    def update_alert_outcomes(alerts):
        """Checks for new outcomes for all 'RUNNING' alerts."""
        if db is None: return
        
        with st.spinner("Refreshing alert outcomes..."):
            updated_count = 0
            for alert in alerts:
                if alert['status'] == 'RUNNING':
                    try:
                        # Fetch *new* data since the alert
                        # We need to know the interval to fetch... this is tricky.
                        # For now, we'll assume the *current* selected interval
                        # This is a limitation we can improve later.
                        
                        # Calculate how many bars have passed
                        alert_time = datetime.fromisoformat(alert['entry_time'])
                        time_diff_seconds = (datetime.now(timezone.utc) - alert_time).total_seconds()
                        
                        # Convert interval to seconds (approx)
                        interval_map = {"1min": 60, "5min": 300, "15min": 900, "30min": 1800, "1h": 3600}
                        interval_seconds = interval_map.get(selected_interval, 3600)
                        
                        # Need at least 2 new bars to check
                        bars_to_fetch = int(time_diff_seconds / interval_seconds) + 2
                        
                        if bars_to_fetch < 2:
                            continue # Not enough time has passed
                        
                        # Fetch just enough new data
                        df_new = fetch_data(alert['pair'], selected_interval, output_size=bars_to_fetch)
                        if df_new.empty:
                            continue
                        
                        # Find the entry bar in the new data
                        try:
                            entry_bar_index = df_new.index.get_loc(alert_time)
                            df_future = df_new.iloc[entry_bar_index + 1:]
                        except KeyError:
                            # Check if data is too old (e.g., entry bar not in the latest 500 candles)
                            if alert_time < df_new.index.min():
                                # Can't determine, data is too old
                                db.child("users").child(user_id).child("alerts").child(alert['id']).update({"status": "EXPIRED"})
                            continue
                        
                        if df_future.empty:
                            continue

                        # Check for TP/SL hit
                        new_status = "RUNNING"
                        tp = float(alert['tp_price'])
                        sl = float(alert['sl_price'])
                        
                        for _, bar in df_future.iterrows():
                            if alert['type'] == 'BUY':
                                if bar['low'] <= sl:
                                    new_status = "LOSS"; break
                                if bar['high'] >= tp:
                                    new_status = "PROFIT"; break
                            elif alert['type'] == 'SELL':
                                if bar['high'] >= sl:
                                    new_status = "LOSS"; break
                                if bar['low'] <= tp:
                                    new_status = "PROFIT"; break
                        
                        if new_status != "RUNNING":
                            updated_count += 1
                            alert['status'] = new_status
                            # Update in Firebase
                            db.child("users").child(user_id).child("alerts").child(alert['id']).update({"status": new_status})

                    except Exception as e:
                        print(f"Error updating outcome for {alert['id']}: {e}")
            
            if updated_count > 0:
                st.sidebar.success(f"Updated {updated_count} alert(s)!")
                # Bust the cache to reload the table
                st.cache_data.clear()
            else:
                st.sidebar.info("No new outcomes found.")


    alert_list = load_alerts_from_firebase(user_id)

    # --- FIX: Sidebar Button Styling ---
    st.sidebar.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #007bff;
            color: white;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #0056b3;
            color: white;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.sidebar.button("Refresh Outcomes", use_container_width=True, key="refresh_outcomes_btn"):
        update_alert_outcomes(alert_list)
        # Rerun to reload the table
        st.rerun()

    if alert_list:
        # Create a clean HTML table
        table_html = "<table class='alert-history-table'><tr><th>Time</th><th>Pair</th><th>Type</th><th>Status</th></tr>"
        
        # Show top 10 most recent alerts
        for alert in alert_list[:10]:
            # Format time to be more readable (e.g. 11/12 20:00)
            try:
                # Try ISO format first
                dt = datetime.fromisoformat(alert['entry_time'])
            except ValueError:
                # Fallback for potential legacy data
                dt = datetime.now() 

            time_str = dt.strftime('%m/%d %H:%M')
            status_class = f"alert-status-{alert['status']}"
            table_html += f"<tr><td>{time_str}</td><td>{alert['pair']}</td><td>{alert['type']}</td><td><span class='{status_class}'>{alert['status']}</span></td></tr>"
        
        table_html += "</table>"
        st.sidebar.markdown(table_html, unsafe_allow_html=True)
    else:
        st.sidebar.info("No alerts found yet.")
    # --- END OF ALERT HISTORY ---


# === 8. Error handling for auth/db init failure ===
elif not st.session_state.user:
    st.set_page_config(page_title="Error - PipWizard", page_icon="🚨", layout="centered")
    st.title("PipWizard 💹")
    st.error("Application failed to initialize.")
    st.warning("Could not connect to the authentication service.")
    st.info("This may be due to missing Streamlit Secrets or. Please contact the administrator.")
    st.code(f"""
    Error Details:
    Auth object: {'Initialized' if auth else 'Failed'}
    DB object:   {'Initialized' if db else 'Failed'}
    User state:  {st.session_state.user}
    Page state:  {st.session_state.page}
    """)


