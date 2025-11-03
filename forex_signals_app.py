import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone # <-- ADDED timezone
import streamlit.components.v1 as components
import talib
from twelvedata import TDClient
import pyrebase  # For Firebase
import json      # For Firebase
import requests  # For Paystack & NEW Calendar

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
    if auth is None or db is None:
        st.error("Auth service not available. Contact support.")
        return
    try:
        user = auth.create_user_with_email_and_password(email, password)
        st.session_state.user = user
        user_data = {"email": email, "subscription_status": "free"}
        db.child("users").child(user['localId']).set(user_data)
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
        st.error(f"Failed to create account: {error_message}")

def login(email, password):
    if auth is None or db is None:
        st.error("Auth service not available. Contact support.")
        return
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.user = user
        user_data = db.child("users").child(user['localId']).get().val()
        if user_data and user_data.get("subscription_status") == "premium":
            st.session_state.is_premium = True
        else:
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
    st.session_state.user = None
    st.session_state.is_premium = False
    st.session_state.page = "login"
    st.rerun()

# === 4. PAYSTACK PAYMENT FUNCTIONS (TYPO FIXED) ===
def create_payment_link(email, user_id):
    """
    Calls Paystack API to create a one-time payment link.
    """
    
    test_amount_kobo = 10000 # 100 NGN * 100 kobo
    
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
    
    # --- THIS IS THE NEW DEBUG LINE ---
    st.write("Loaded Secret Keys:", st.secrets.keys())
    # ---
    
    # --- NEW: Check for Payment Callback ---
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
    OUTPUTSIZE = 500

    # === THEME ===
    if 'theme' not in st.session_state:
        st.session_state.theme = "dark"
    def toggle_theme():
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    def apply_theme():
        dark = st.session_state.theme == "dark"
        return f"""<style>
            .stApp {{ background-color: {'#0e1117' if dark else '#ffffff'}; color: {'#f0f0f0' if dark else '#212529'}; }}
            .buy-signal {{ color: #26a69a; }} .sell-signal {{ color: #ef5350; }}
            .results-box {{
                border: 1px solid {'#555' if dark else '#ddd'};
                border-radius: 5px;
                padding: 10px;
                margin-top: -10px;
                margin-bottom: 10px;
                background-color: {'#1a1a1a' if dark else '#f9f9f9'};
            }}
            .results-text {{
                font-size: 0.9em;
                color: {'#bbb' if dark else '#333'};
            }}
            .actual-good {{ color: #26a69a; font-weight: bold; }}
            .actual-bad {{ color: #ef5350; font-weight: bold; }}
            .actual-neutral {{ color: {'#f0f0f0' if dark else '#212529'}; font-weight: bold; }}
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

    # === ABOUT THE APP SECTION ===
    with st.expander("👋 Welcome to PipWizard! Click here to learn about the app."):
        st.markdown(
            f"""
            ### What is PipWizard?
            
            PipWizard is a powerful decision-support tool for forex traders. It's designed to help you **find**, **test**, and **act on** trading strategies in real-time.
            It combines a live signal generator, economic news calendar, and a powerful, on-demand backtesting engine.

            ### How to Use the App
            1.  **Step 1: TEST A STRATEGY (The "Main Backtest")**
                * Use the sidebar to pick a strategy (`RSI Standalone`, etc.) and set your `Stop Loss` and `Take Profit`.
                * Click the **"Run Backtest"** button to see a full report, including an **Equity Curve** and **Trade Log**.
            2.  **Step 2: FIND THE BEST STRATEGY (Premium Feature)**
                * Open the **"🚀 Strategy Scanner"** at the bottom of the page.
                * This "heatmap" tool tests all strategies across all pairs and timeframes at once.
            3.  **Step 3: ACTIVATE LIVE SIGNALS (Premium Feature)**
                * Set your chosen parameters in the sidebar. The app will run in "live" mode, showing signals on the chart as they happen.
                * Premium users will also receive an "ALERT SENT" in the sidebar.

            ### Feature Tiers: Free vs. Premium
            **🎁 Free Tier (Your Current Plan):**
            * ✅ **Economic News Calendar** (Real-time data from Investing.com)
            * ✅ **Full Backtesting Engine**
            * ✅ **All 6 Strategies** & All Timeframes
            * 🔒 **Limited to EUR/USD** only.

            **⭐ Premium Tier ($29.99/month):**
            Upgrade for **$29.99/month** to unlock every feature:
            * ✅ **Unlock All 10+ Currency Pairs**
            * ✅ **🚀 Strategy Scanner**
            * ✅ **Live Signal Alerts**
            *(Note: Scanner speed is limited by the Twelve Data Free API plan)*
            """
        )
    
    # === SIDEBAR & CONTROLS ===
    st.sidebar.title("PipWizard")
    
    user_email = "User"
    if st.session_state.user and 'email' in st.session_state.user:
        user_email = st.session_state.user['email']
    st.sidebar.write(f"Logged in as: `{user_email}`")
    
    is_premium = st.session_state.is_premium
    if is_premium:
        selected_pair = st.sidebar.selectbox("Select Pair", PREMIUM_PAIRS, index=0)
        st.sidebar.success("Premium Active – All Features Unlocked")
    else:
        selected_pair = FREE_PAIR
        st.sidebar.warning("Free Tier: EUR/USD Only")
        st.sidebar.info("Upgrade to Premium to unlock all pairs and the Strategy Scanner!")
    selected_interval = st.sidebar.selectbox("Timeframe", options=list(INTERVALS.keys()), index=3, format_func=lambda x: x.replace("min", " minute").replace("1h", "1 hour"))
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Strategy Selection")
    strategy_name = st.sidebar.selectbox("Choose a Strategy", ["RSI + SMA Crossover", "MACD Crossover", "RSI + MACD (Confluence)", "SMA + MACD (Confluence)", "RSI Standalone", "SMA Crossover Standalone"])
    
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
    risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, key='risk_pct') / 100
    sl_pips = st.sidebar.number_input("Stop Loss (Pips)", min_value=1, max_value=200, value=50, key='sl_pips')
    
    tp_pips = st.sidebar.number_input("Take Profit (Pips)", min_value=1, max_value=500, value=100, key='tp_pips') # Default is 100
    
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
        Full backtesting on EUR/USD only.
        **⭐ Upgrade to Premium ($29.99/mo):**\n
        • Unlock all pairs\n
        • Unlock Strategy Scanner\n
        • Get Live Signal Alerts
        """
    )
    if not is_premium:
        if st.sidebar.button("Upgrade to Premium Now!", type="primary", use_container_width=True, key="upgrade_button"):
            st.session_state.page = "profile"
            st.rerun()
    st.sidebar.markdown("---")
    if st.sidebar.button("Profile & Logout", use_container_width=True, key="profile_button"):
        st.session_state.page = "profile"
        st.rerun()
    
    # === HELPER FUNCTIONS (Alerts & Calendar) ===
    @st.cache_data(ttl=60)
    def fetch_data(symbol, interval):
        if "TD_API_KEY" not in st.secrets:
            st.error("TD_API_KEY not found in Streamlit Secrets."); return pd.DataFrame()
        td = TDClient(apikey=st.secrets["TD_API_KEY"])
        try:
            ts = td.time_series(symbol=symbol, interval=interval, outputsize=OUTPUTSIZE).as_pandas()
            if ts is None or ts.empty:
                st.error(f"No data returned for {symbol}."); return pd.DataFrame()
            df = ts[['open', 'high', 'low', 'close']].copy()
            df.index = pd.to_datetime(df.index)
            return df.iloc[::-1]
        except Exception as e:
            st.error(f"API Error fetching {symbol}: {e}"); return pd.DataFrame()

    def send_alert_email(signal_type, price, pair):
        st.sidebar.markdown(f"**ALERT SENT**")
        st.sidebar.warning(f"**{signal_type.upper()}** on {pair} at {price:.5f}")

    def check_for_live_signal(df, pair):
        if len(df) < 2: return
        latest_bar, current_bar = df.iloc[-2], df.iloc[-1]
        signal, price = latest_bar['signal'], current_bar['open']
        if 'last_alert_time' not in st.session_state: st.session_state.last_alert_time = None
        if signal != 0 and latest_bar.name != st.session_state.last_alert_time:
            st.session_state.last_alert_time = latest_bar.name
            if signal == 1: send_alert_email("BUY", price, pair)
            elif signal == -1: send_alert_email("SELL", price, pair)

    # --- CALENDAR FUNCTION (NEW: FINNHUB API) ---
    def display_news_calendar():
        st.subheader("Upcoming Economic Calendar")
        search = st.text_input("Search events", placeholder="e.g., NFP, PMI, CPI", key="calendar_search")

        @st.cache_data(ttl=300)
        def get_free_calendar():
            try:
                if "FINNHUB_API_KEY" not in st.secrets:
                    # This check is crucial
                    st.error("Please add FINNHUB_API_KEY to your Streamlit secrets to load the calendar.")
                    return pd.DataFrame()
                    
                token = st.secrets["FINNHUB_API_KEY"]
                now = datetime.now(timezone.utc)
                start_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
                end_date = (now + timedelta(days=7)).strftime("%Y-%m-%d")
                
                url = f"https://finnhub.io/api/v1/calendar/economic?from={start_date}&to={end_date}&token={token}"
                
                response = requests.get(url, timeout=10)
                response.raise_for_status() # Will error if key is bad or API is down
                data = response.json().get("economicCalendar", [])
                
                if not data:
                    return pd.DataFrame() # This will show "No events loaded"
                    
                events = []
                
                # Helper for robust number conversion
                def to_num(v):
                    if v is None: return float('nan')
                    v = str(v).strip().replace(',', '').replace('%', '').replace('$', '')
                    if v.endswith('K'): return float(v[:-1]) * 1000
                    if v.endswith('M'): return float(v[:-1]) * 1000000
                    if v.endswith('B'): return float(v[:-1]) * 1000000000
                    if v == "" or v == "N/A": return float('nan')
                    return float(v)

                for e in data:
                    # Finnhub time is in seconds timestamp
                    event_time = datetime.fromtimestamp(e.get("time"), tz=timezone.utc)
                    
                    # Filter for major currencies
                    if e.get("country") not in ["US", "GB", "EU", "JP", "CA", "AU", "NZ", "CN"]:
                        continue

                    actual = e.get("actual")
                    forecast = e.get("forecast")
                    previous = e.get("prev")
                    
                    actual_display = actual if actual is not None else "Pending"
                    
                    surprise = ""
                    try:
                        a, f = to_num(actual), to_num(forecast)
                        if not (pd.isna(a) or pd.isna(f)):
                            if a > f: surprise = "Better than Expected"
                            elif a < f: surprise = "Worse than Expected"
                            else: surprise = "As Expected"
                    except:
                        surprise = ""
                    
                    # Map Finnhub impact 1,2,3 to Low,Medium,High
                    impact_num = e.get("impact")
                    if impact_num == 3: impact_str = "High"
                    elif impact_num == 2: impact_str = "Medium"
                    else: impact_str = "Low"

                    events.append({
                        "date": event_time.strftime("%A, %b %d"),
                        "time": event_time.strftime("%H:%M"),
                        "event": e.get("event"),
                        "country": e.get("country"),
                        "impact": impact_str,
                        "forecast": forecast if forecast is not None else "N/A",
                        "previous": previous if previous is not None else "N/A",
                        "actual": actual_display,
                        "surprise": surprise,
                        "date_dt": event_time
                    })
                
                df = pd.DataFrame(events)
                return df.sort_values("date_dt").drop(columns="date_dt") if not df.empty else pd.DataFrame()

            except Exception as e:
                # This is the FINAL fallback
                print(f"Error in get_free_calendar (Finnhub): {e}") 
                return pd.DataFrame([
                    {"date": "Friday, Nov 08", "time": "13:30", "event": "Nonfarm Payrolls (Fallback)", "country": "US", "impact": "High", 
                     "forecast": "175K", "previous": "254K", "actual": "Pending", "surprise": ""}
                ])
        # --- END OF REPLACED INNER FUNCTION ---
        
        with st.spinner("Loading live economic calendar..."):
            df = get_free_calendar()
        
        if df.empty:
            st.info("No economic events loaded for the next 7 days.") # Changed message
            if st.button("Refresh Calendar"):
                st.cache_data.clear()
                st.rerun()
            return
            
        if search:
            df = df[df["event"].str.contains(search, case=False, na=False)]
            if df.empty:
                st.info(f"No events matching '{search}'.")
                return
                
        # Your styling code is UNCHANGED
        st.markdown("""
        <style>
        .calendar-table { width: 100%; border-collapse: collapse; font-family: 'Segoe UI', sans-serif; margin: 10px 0; }
        .calendar-table th { background: #1f77b4; color: white; padding: 12px; text-align: left; font-weight: 600; }
        .calendar-table td { padding: 10px 12px; border-bottom: 1px solid #444; }
        .calendar-table tr:hover { background: #2a2a2a !important; }
        .impact-high { background: #ffebee; color: #c62828; font-weight: bold; }
        .impact-medium { background: #fff3e0; color: #ef6c00; font-weight: bold; }
        .impact-low { background: #f3e5f5; color: #6a1b9a; }
        .actual-better { background: #e8f5e8; color: #2e7d32; font-weight: bold; }
        .actual-worse { background: #ffebee; color: #c62828; font-weight: bold; }
        .actual-expected { background: #fff8e1; color: #ff8f00; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
        
        def style_row(row):
            styles = [""] * len(row)
            impact_high_css = "background: #ffebee; color: #c62828; font-weight: bold;"
            impact_medium_css = "background: #fff3e0; color: #ef6c00; font-weight: bold;"
            impact_low_css = "background: #f3e5f5; color: #6a1b9a;"
            actual_better_css = "background: #e8f5e8; color: #2e7d32; font-weight: bold;"
            actual_worse_css = "background: #ffebee; color: #c62828; font-weight: bold;"
            actual_expected_css = "background: #fff8e1; color: #ff8f00; font-weight: bold;"
            
            # Column indices: 0:date, 1:time, 2:event, 3:country, 4:impact, 5:forecast, 6:previous, 7:actual, 8:surprise
            if row["impact"] == "High": styles[4] = impact_high_css
            elif row["impact"] == "Medium": styles[4] = impact_medium_css
            elif row["impact"] == "Low": styles[4] = impact_low_css
            
            if row["surprise"] == "Better than Expected": styles[8] = actual_better_css
            elif row["surprise"] == "Worse than Expected": styles[8] = actual_worse_css
            elif row["surprise"] == "As Expected": styles[8] = actual_expected_css
            
            return styles
            
        styled = df.style.apply(style_row, axis=1).set_table_attributes('class="calendar-table"')
        
        st.markdown(styled.to_html(), unsafe_allow_html=True)
        
        if st.button("Refresh Calendar"):
            st.cache_data.clear()
            st.rerun()
            
        st.caption("Source: Finnhub.io • Live Actuals • Times in UTC")
    # --- END OF CALENDAR FUNCTION ---
    
    # === INDICATOR & STRATEGY LOGIC (ACCEPTING PARAMS) ===
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

    # === (FIXED) BACKTESTING FUNCTION ===
    def run_backtest(df_in, pair_name, initial_capital, risk_per_trade, sl_pips, tp_pips):
        df = df_in.copy(); trades = []
        if "JPY" in pair_name: PIP_MULTIPLIER = 0.01
        else: PIP_MULTIPLIER = 0.0001
        RISK_PIPS_VALUE = sl_pips * PIP_MULTIPLIER; REWARD_PIPS_VALUE = tp_pips * PIP_MULTIPLIER
        MAX_RISK_USD = initial_capital * risk_per_trade; REWARD_USD = MAX_RISK_USD * (tp_pips / sl_pips)
        signal_bars = df[df['signal'] != 0]
        for i in range(len(signal_bars)):
            signal_row, signal_type = signal_bars.iloc[i], signal_bars.iloc[i]['signal']
            try: signal_index = df.index.get_loc(signal_row.name)
            except KeyError: continue
            if signal_index + 1 >= len(df): continue
            entry_bar, entry_price, entry_time = df.iloc[signal_index + 1], df.iloc[signal_index + 1]['open'], df.iloc[signal_index + 1].name
            stop_loss, take_profit = (entry_price - RISK_PIPS_VALUE, entry_price + REWARD_PIPS_VALUE) if signal_type == 1 else (entry_price + RISK_PIPS_VALUE, entry_price - REWARD_PIPS_VALUE)
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
        total_trades, winning_trades = len(resolved_trades), len(resolved_trades[resolved_trades['result'] == 'WIN'])
        total_profit, win_rate = resolved_trades['profit_loss'].sum(), winning_trades / total_trades
        gross_win, gross_loss = resolved_trades[resolved_trades['profit_loss'] > 0]['profit_loss'].sum(), abs(resolved_trades[resolved_trades['profit_loss'] < 0]['profit_loss'].sum())
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
        df = calculate_indicators(df, rsi_period, sma_period, macd_fast, macd_slow, macd_signal)
    with st.spinner(f"Applying Strategy: {strategy_name}..."):
        df = apply_strategy(df, strategy_name, alert_rsi_low, alert_rsi_high)
    df = df.dropna()
    if df.empty:
        st.warning("Waiting for sufficient data after indicator calculation..."); st.stop()

    # === RUN MAIN BACKTESTING ON BUTTON CLICK ===
    if run_backtest_button:
        with st.spinner("Running backtest on real market data..."):
            total_trades, win_rate, total_profit, pf, final_cap, trade_df, res_df = run_backtest(
                df, selected_pair, initial_capital, risk_pct, sl_pips, tp_pips
            )
            st.session_state.backtest_results = {
                "total_trades": total_trades, "win_rate": win_rate, "total_profit": total_profit,
                "profit_factor": pf, "final_capital": final_cap, "trade_df": trade_df,
                "resolved_trades_df": res_df, "pair": selected_pair, "interval": selected_interval, "data_len": len(df)
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
        
        resolved_df_key = 'resolved_trades_df' if 'resolved_trades_df' in results else 'resolved_trades_ _df'
        
        if resolved_df_key in results and not results[resolved_df_key].empty:
            equity_fig = go.Figure()
            equity_fig.add_trace(go.Scatter(x=results[resolved_df_key]['exit_time'], y=results[resolved_df_key]['equity'], mode='lines', name='Equity', line=dict(color='#26a69a')))
            equity_fig.update_layout(xaxis_title="Time", yaxis_title="Account Equity ($)", template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white', height=300)
            st.plotly_chart(equity_fig, use_container_width=True)
        else: st.info("No resolved trades found with these settings.")
        st.subheader("Detailed Trade Log")
        # --- FIX: Replaced use_container_width ---
        st.dataframe(results['trade_df'], width='stretch')
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
    buy_signals, sell_signals = df[df['signal'] == 1], df[df['signal'] == -1]
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
    
    # --- CRASH FIX ---
    fig.update_layout(height=600, template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white', xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # === LIVE SIGNAL ALERT CHECK ===
    if is_premium:
        check_for_live_signal(df, selected_pair)
        
    # --- NEWS CALENDAR SECTION ---
    st.markdown("---")
    display_news_calendar() # <-- This now calls the NEW, upgraded function
    st.markdown("---")
    
    # === STRATEGY SCANNER (PREMIUM FEATURE) ===
    if is_premium:
        with st.expander("🚀 Strategy Scanner (Premium Feature)"):
            st.info("Compare all strategies across multiple pairs and timeframes to find the best performers.")
            
            all_strategies = [
                "RSI + SMA Crossover",
                "MACD Crossover",
                "RSI + MACD (Confluence)",
                "SMA + MACD (Confluence)",
                "RSI Standalone",
                "SMA Crossover Standalone"
            ]
            
            col1, col2, col3 = st.columns(3)
            scan_pairs = col1.multiselect("Select Pairs", PREMIUM_PAIRS, default=["EUR/USD", "GBP/USD", "USD/JPY"])
            scan_intervals = col2.multiselect("Select Timeframes", list(INTERVALS.keys()), default=["15min", "1h"])
            scan_strategies = col3.multiselect("Select Strategies", all_strategies, default=["RSI Standalone", "MACD Crossover"])
            
            # --- THIS IS THE SYNTAX ERROR FIX ---
            scan_params = {"rsi_p": 14, "sma_p": 20, "macd_f": 12, "macd_sl": 26, "macd_sig": 9, "rsi_l": 30, "rsi_h": 70, "capital": 10000, "risk": 0.01, "sl": 50, "tp": 100}
            
            if st.button("Run Full Scan", type="primary", key="scan_button"):
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
                            data = fetch_data(pair, interval_val)
                            if data.empty:
                                st.warning(f"Could not fetch data for {pair} ({interval_key}). Skipping."); total_jobs -= len(scan_strategies); continue
                            for strategy in scan_strategies:
                                job_count += 1
                                progress_bar.progress(job_count / total_jobs, text=f"Testing {strategy} on {pair} ({interval_key})... ({job_count}/{total_jobs})")
                                data_with_indicators = calculate_indicators(data.copy(), scan_params["rsi_p"], scan_params["sma_p"], scan_params["macd_f"], scan_params["macd_sl"], scan_params["macd_sig"])
                                data_with_signal = apply_strategy(data_with_indicators, strategy, scan_params["rsi_l"], scan_params["rsi_h"])
                                data_with_signal = data_with_signal.dropna()
                                if data_with_signal.empty: continue
                                total_trades, win_rate, total_profit, pf, _, _, _ = run_backtest(
                                    data_with_signal, pair, scan_params["capital"], scan_params["risk"],
                                    scan_params["sl"], scan_params["tp"]
                                )
                                if total_trades > 0:
                                    scan_results.append({"Pair": pair, "Timeframe": interval_key, "Strategy": strategy, "Total Profit ($)": total_profit, "Win Rate (%)": win_rate * 100, "Profit Factor": pf, "Total Trades": total_trades})
                    progress_bar.progress(1.0, text="Scan Complete!")
                    if scan_results:
                        results_df = pd.DataFrame(scan_results).sort_values(by="Total Profit ($)", ascending=False).reset_index(drop=True)
                        def style_profit(val):
                            color = '#26a69a' if val > 0 else '#ef5350' if val < 0 else '#f0f0f0'; return f'color: {color}; font-weight: bold;'
                        def style_win_rate(val):
                            val = max(0, min(100, val)); color = 'white'
                            if val < 50: return f'background-color: rgba(239, 83, 80, {1 - (val/50)}); color: {color};'
                            else: return f'background-color: rgba(38, 166, 154, {(val-50)/50}); color: {color};'
                        def style_profit_factor(val):
                            color = '#26a69a' if val >= 1.0 else '#ef5350'; return f'color: {color}; font-weight: bold;'
                        
                        # --- FIX: Replaced use_container_width ---
                        st.dataframe(
                            results_df.style
                                .applymap(style_profit, subset=['Total Profit ($)'])
                                .apply(lambda x: [style_win_rate(v) for v in x], subset=['Win Rate (%)'])
                                .applymap(style_profit_factor, subset=['Profit Factor'])
                                .format({"Total Profit ($)": "${:,.2f}", "Win Rate (%)": "{:.2f}%", "Profit Factor": "{:.2f}"}),
                            width='stretch'
                        )
                    else:
                        st.info("Scan completed, but no trades were found with these settings.")
    else:
         st.info("The **🚀 Strategy Scanner** is a Premium feature. Go to your Profile to upgrade!")

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

# === 6. Error handling for auth/db init failure ===
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
