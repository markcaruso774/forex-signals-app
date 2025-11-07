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
                config["databaseURL"] = f"https://{project_id}-default-rtdb.firebaseio.com/"
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

# === 4. PAYSTACK PAYMENT FUNCTIONS ===
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


import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
from threading import Thread
import websocket
import pyrebase

# Optional indicator libs
try:
    import talib
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False

# -------------------- Basic Config --------------------
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD", "XAU/USD", "BTC/USD"]
DEFAULT_PAIR = "EUR/USD"
UPDATE_INTERVAL = 5  # seconds between UI redraws
PRICE_HISTORY_MAX = 1200  # max ticks to keep

# -------------------- Helpers for indicators --------------------
def simple_sma(series, window):
    return series.rolling(window=window).mean()

def simple_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(window-1), adjust=False).mean()
    ma_down = down.ewm(com=(window-1), adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def simple_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

# -------------------- TwelveData WebSocket streamer --------------------
def build_ws_url(pair, apikey):
    sym = pair.replace("/", "")
    return f"wss://ws.twelvedata.com/v1/quotes/price?symbol={sym}&apikey={apikey}"

def start_price_stream(pair, apikey):
    ws_key = "td_stream_thread"
    if ws_key in st.session_state and st.session_state.get("td_stream_pair") == pair and st.session_state.get(ws_key):
        return

    st.session_state["td_stream_stop"] = True
    time.sleep(0.2)
    st.session_state["td_stream_stop"] = False
    st.session_state["td_stream_pair"] = pair

    url = build_ws_url(pair, apikey)

    def _on_message(ws, message):
        try:
            data = json.loads(message)
            if "price" in data:
                price = float(data["price"])
                now = datetime.utcnow().replace(tzinfo=timezone.utc)
                new = pd.DataFrame({"time":[now], "price":[price]})
                if "price_df" not in st.session_state:
                    st.session_state.price_df = new
                else:
                    st.session_state.price_df = pd.concat([st.session_state.price_df, new])
                    if len(st.session_state.price_df) > PRICE_HISTORY_MAX:
                        st.session_state.price_df = st.session_state.price_df.tail(PRICE_HISTORY_MAX)
        except Exception as e:
            print("ws message parse error:", e)

    def _on_error(ws, err):
        print("WebSocket error:", err)

    def _on_close(ws, close_status_code, close_msg):
        print("WebSocket closed.", close_status_code, close_msg)

    def _on_open(ws):
        print("WebSocket connection opened. Streaming", pair)

    def run_ws():
        while not st.session_state.get("td_stream_stop", False):
            try:
                ws = websocket.WebSocketApp(url, on_message=_on_message, on_error=_on_error, on_close=_on_close, on_open=_on_open)
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                print("WebSocket run error:", e)
            for _ in range(5):
                if st.session_state.get("td_stream_stop", False):
                    break
                time.sleep(1)
        print("Price stream thread exiting.")

    t = Thread(target=run_ws, daemon=True)
    t.start()
    st.session_state["td_stream_thread"] = True

def fetch_last_price_rest(pair, apikey):
    sym = pair.replace("/", "")
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={sym}&interval=1min&outputsize=1&format=json&apikey={apikey}"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        j = r.json()
        if "values" in j and len(j["values"])>0:
            v = j["values"][0]
            price = float(v["close"])
            now = datetime.utcnow().replace(tzinfo=timezone.utc)
            return now, price
    except Exception as e:
        print("REST fallback error:", e)
    return None, None

# -------------------- Streamlit App Layout --------------------
st.set_page_config(page_title="PipWizard", layout="wide")
if 'user' not in st.session_state:
    st.session_state.user = None
if 'is_premium' not in st.session_state:
    st.session_state.is_premium = False
if 'page' not in st.session_state:
    st.session_state.page = "login"

# ===== LOGIN / SIGNUP UI =====
if st.session_state.page == "login":
    st.title("PipWizard 💹")
    st.write("Please log in to continue.")
    
    # Check for payment verification in query params
    query_params = st.query_params
    if "trxref" in query_params and "reference" in query_params:
        reference = query_params["reference"]
        with st.spinner("Verifying your payment, please wait..."):
            verify_payment(reference)

    col1, col2 = st.columns(2)
    with col1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if not email or not password:
                st.error("Please enter email and password.")
            else:
                login(email, password)
    with col2:
        st.write("New here?")
        su_email = st.text_input("Sign up email", key="su_email")
        su_pass = st.text_input("Sign up password", type="password", key="su_pass")
        su_pass2 = st.text_input("Confirm password", type="password", key="su_pass2")
        if st.button("Sign Up"):
            if not su_email or not su_pass or not su_pass2:
                st.error("Fill all fields")
            elif su_pass != su_pass2:
                st.error("Passwords do not match")
            else:
                sign_up(su_email, su_pass)

# ===== MAIN APP AFTER LOGIN: show live chart immediately =====
elif st.session_state.page == "app" and st.session_state.user:
    st.title("PipWizard – Live Dashboard")
    st.sidebar.header("Controls")
    user_email = st.session_state.user.get("email", "User")
    user_id = st.session_state.user.get("localId")
    st.sidebar.write(f"Logged in as: {user_email}")
    is_premium = st.session_state.is_premium

    # --- Premium Upgrade Sidebar ---
    if not is_premium:
        st.sidebar.warning("You are on the Free plan.")
        if st.sidebar.button("Upgrade to Premium"):
            with st.spinner("Creating payment link..."):
                auth_url, ref = create_payment_link(user_email, user_id)
                if auth_url:
                    st.sidebar.markdown(f"**[Click here to pay]({auth_url})**")
                    st.sidebar.info("After paying, return to this page. Your account will be upgraded.")
                else:
                    st.sidebar.error("Could not create payment link. Please try again.")
    else:
        st.sidebar.success("You are a Premium user! ✨")
    
    if st.sidebar.button("Logout"):
        logout()

    # Pair selector and refresh settings
    pair = st.sidebar.selectbox("Select Pair", PAIRS, index=PAIRS.index(DEFAULT_PAIR))
    refresh_choice = st.sidebar.selectbox("Update interval (seconds)", [1,2,5,10], index=2)
    apikey = st.secrets.get("TWELVEDATA", {}).get("API_KEY", None)
    if not apikey:
        st.sidebar.error("TwelveData API key missing. Add TWELVEDATA.API_KEY to Streamlit Secrets.")
        st.stop()

    st.session_state["live_refresh_interval"] = int(refresh_choice)

    # start streamer for selected pair
    if "price_df" not in st.session_state:
        st.session_state.price_df = pd.DataFrame(columns=["time","price"])

    start_price_stream(pair, apikey)

    # ===== FIX 1: Clear old data when the pair changes =====
    if "current_chart_pair" not in st.session_state or st.session_state.current_chart_pair != pair:
        st.session_state.price_df = pd.DataFrame(columns=["time","price"]) # Clear old data!
        st.session_state.current_chart_pair = pair # Set the new pair
    # ========================================================

    # seed via REST if WS slow
    seed_wait_seconds = 3
    if len(st.session_state.price_df) == 0:
        t0 = time.time()
        with st.spinner(f"Connecting to {pair} stream..."):
            while time.time() - t0 < seed_wait_seconds and len(st.session_state.price_df) == 0:
                now, price = fetch_last_price_rest(pair, apikey)
                if price is not None:
                    st.session_state.price_df = pd.DataFrame({"time":[now], "price":[price]})
                    break
                time.sleep(0.5)

    # Live chart placeholder
    placeholder = st.empty()

    # Main live update loop
    while True:
        df = st.session_state.price_df.copy()
        if df.empty:
            placeholder.info("Waiting for live ticks...")
            time.sleep(st.session_state.get("live_refresh_interval", UPDATE_INTERVAL))
            continue

        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').drop_duplicates(subset='time').reset_index(drop=True)

        # ===== FIX 2: Only calculate indicators if we have enough data =====
        # MACD (26) is the slowest, so we use that as the threshold.
        if len(df) > 27:
            df['sma20'] = simple_sma(df['price'], 20)
            if _HAS_TALIB:
                try:
                    df['rsi'] = talib.RSI(df['price'].values, timeperiod=14)
                    macd_line, macd_signal, macd_hist = talib.MACD(df['price'].values, fastperiod=12, slowperiod=26, signalperiod=9)
                    df['macd_line'] = macd_line
                    df['macd_signal'] = macd_signal
                    df['macd_hist'] = macd_hist
                except Exception:
                    df['rsi'] = simple_rsi(df['price'], 14)
                    df['macd_line'], df['macd_signal'], df['macd_hist'] = simple_macd(df['price'])
            else:
                df['rsi'] = simple_rsi(df['price'], 14)
                df['macd_line'], df['macd_signal'], df['macd_hist'] = simple_macd(df['price'])

            buys = df[df['rsi'] < 30]
            sells = df[df['rsi'] > 70]
        
        # If we don't have enough data, create empty columns/DataFrames
        # so the rest of the plotting code doesn't fail
        else:
            df['sma20'] = np.nan
            df['rsi'] = np.nan
            df['macd_line'] = np.nan
            df['macd_signal'] = np.nan
            df['macd_hist'] = np.nan
            buys = pd.DataFrame()
            sells = pd.DataFrame()
        # ===================================================================

        # build figure
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6,0.2,0.2], vertical_spacing=0.03,
                            specs=[[{"type":"scatter"}],[{"type":"scatter"}],[{"type":"bar"}]])
        
        # --- Plot 1: Price + SMA + Signals ---
        fig.add_trace(go.Scatter(x=df['time'], y=df['price'], mode='lines', name='Price', line=dict(color='#00bcd4')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['sma20'], mode='lines', name='SMA20', line=dict(color='#ff9800')), row=1, col=1)
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys['time'], y=buys['price'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='#26a69a', size=10)), row=1, col=1)
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells['time'], y=sells['price'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='#ef5350', size=10)), row=1, col=1)

        # --- Plot 2: RSI ---
        fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], mode='lines', name='RSI(14)', line=dict(color='#9c27b0')), row=2, col=1)
        fig.add_hline(y=70, line_dash='dash', line_color='#ef5350', row=2, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='#26a69a', row=2, col=1)
        fig.update_yaxes(range=[0,100], row=2, col=1)

        # --- Plot 3: MACD ---
        colors = ['#26a69a' if v>=0 else '#ef5350' for v in df['macd_hist'].fillna(0)]
        fig.add_trace(go.Bar(x=df['time'], y=df['macd_hist'], name='MACD Histogram', marker_color=colors), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['macd_signal'], mode='lines', name='MACD Signal', line=dict(color='#ff9800')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['macd_line'], mode='lines', name='MACD', line=dict(color='#2196f3')), row=3, col=1)

        fig.update_layout(template='plotly_dark', showlegend=True, height=700, margin=dict(l=10,r=10,t=40,b=10))
        fig.update_xaxes(rangeslider_visible=False)

        placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # The redundant pair-switching check at the end of the loop has been removed.
        # Streamlit's natural rerun behavior handles this.

        time.sleep(st.session_state.get("live_refresh_interval", UPDATE_INTERVAL))

# ===== If user not logged in properly =====
else:
    st.title("PipWizard 💹")
    st.error("You must be logged in to view the dashboard. Please log in.")
    # Check for payment verification in query params on login page too
    query_params = st.query_params
    if "trxref" in query_params and "reference" in query_params:
        reference = query_params["reference"]
        with st.spinner("Verifying your payment, please wait..."):
            verify_payment(reference)
            if st.session_state.page == "app": # if verify_payment logs us in
                st.rerun() # Rerun to show the app
