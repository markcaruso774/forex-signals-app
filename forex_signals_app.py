import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import streamlit.components.v1 as components
import talib
from twelvedata import TDClient
import pyrebase
import json
import requests
from lightweight_charts.widgets import StreamlitChart

# ==============================================================
# 1. FIREBASE
# ==============================================================
def initialize_firebase():
    try:
        if "FIREBASE_CONFIG" not in st.secrets:
            st.error("Firebase config not found.")
            return None, None
        cfg = st.secrets["FIREBASE_CONFIG"]
        if "databaseURL" not in cfg:
            pid = cfg.get('projectId', cfg.get('project_id'))
            if pid:
                cfg["databaseURL"] = f"https://{pid}-default-rtdb.firebaseio.com/"
            else:
                cfg["databaseURL"] = f"https://{cfg['authDomain'].split('.')[0]}-default-rtdb.firebaseio.com/"
        firebase = pyrebase.initialize_app(cfg)
        return firebase.auth(), firebase.database()
    except Exception as e:
        st.error(f"Firebase error: {e}")
        return None, None

auth, db = initialize_firebase()

# ==============================================================
# 2. SESSION STATE
# ==============================================================
for key, default in [
    ("user", None), ("is_premium", False), ("page", "login"),
    ("theme", "dark"), ("backtest_results", None), ("scanner_results", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==============================================================
# 3. AUTH
# ==============================================================
def sign_up(email, pwd):
    if not (auth and db): st.error("Auth unavailable."); return
    try:
        user = auth.create_user_with_email_and_password(email, pwd)
        st.session_state.user = user
        db.child("users").child(user["localId"]).set({"email": email, "subscription_status": "free"})
        st.session_state.is_premium = False
        st.session_state.page = "app"
        st.rerun()
    except Exception as e:
        msg = json.loads(e.args[1]).get("error", {}).get("message", "Unknown")
        st.error(f"Sign-up failed: {msg}")

def login(email, pwd):
    if not (auth and db): st.error("Auth unavailable."); return
    try:
        user = auth.sign_in_with_email_and_password(email, pwd)
        st.session_state.user = user
        data = db.child("users").child(user["localId"]).get().val()
        st.session_state.is_premium = data and data.get("subscription_status") == "premium"
        st.session_state.page = "app"
        st.rerun()
    except Exception as e:
        msg = json.loads(e.args[1]).get("error", {}).get("message", "Unknown")
        st.error(f"Login failed: {msg}")

def logout():
    st.session_state.user = None
    st.session_state.is_premium = False
    st.session_state.page = "login"
    st.rerun()

# ==============================================================
# 4. PAYSTACK
# ==============================================================
def create_payment_link(email, uid):
    if "PAYSTACK_TEST" not in st.secrets: st.error("Paystack not configured."); return None, None
    if "APP_URL" not in st.secrets: st.error("APP_URL missing."); return None, None
    url = "https://api.paystack.co/transaction/initialize"
    headers = {"Authorization": f"Bearer {st.secrets['PAYSTACK_TEST']['PAYSTACK_SECRET_KEY']}",
               "Content-Type": "application/json"}
    payload = {"email": email, "amount": 10000, "callback_url": st.secrets["APP_URL"],
               "metadata": {"user_id": uid, "user_email": email}}
    try:
        r = requests.post(url, headers=headers, json=payload).json()
        if r.get("status"): return r["data"]["authorization_url"], r["data"]["reference"]
        st.error(r.get("message")); return None, None
    except Exception as e:
        st.error(e); return None, None

def verify_payment(ref):
    if not (db and "PAYSTACK_TEST" in st.secrets): st.error("Service unavailable."); return False
    url = f"https://api.paystack.co/transaction/verify/{ref}"
    headers = {"Authorization": f"Bearer {st.secrets['PAYSTACK_TEST']['PAYSTACK_SECRET_KEY']}"}
    try:
        r = requests.get(url, headers=headers).json()
        if r.get("status") and r["data"]["status"] == "success":
            uid = r["data"]["metadata"].get("user_id")
            if uid:
                db.child("users").child(uid).update({"subscription_status": "premium"})
                st.session_state.is_premium = True
                st.success("Premium activated!"); st.balloons()
                st.session_state.page = "app"
                st.query_params.clear()
                st.rerun()
            return True
        st.error("Payment failed."); return False
    except Exception as e:
        st.error(e); return False

# ==============================================================
# 5. LOGIN PAGE
# ==============================================================
if st.session_state.page == "login":
    st.set_page_config(page_title="Login – PipWizard", page_icon="Chart", layout="centered")
    if not (auth and db): st.title("PipWizard"); st.error("Init failed."); st.stop()
    st.title("Welcome to PipWizard")
    action = st.radio("Action", ("Login", "Sign Up"), horizontal=True, index=1)
    email = st.text_input("Email")
    pwd = st.text_input("Password", type="password")
    if action == "Sign Up":
        confirm = st.text_input("Confirm Password", type="password")
        if st.button("Sign Up"):
            if pwd != confirm: st.error("Passwords don’t match.")
            elif not email or not pwd: st.error("Fill all fields.")
            else: sign_up(email, pwd)
    if action == "Login":
        if st.button("Login"):
            if not email or not pwd: st.error("Fill all fields.")
            else: login(email, pwd)

# ==============================================================
# 6. PROFILE PAGE
# ==============================================================
elif st.session_state.page == "profile":
    st.set_page_config(page_title="Profile – PipWizard", page_icon="Chart", layout="centered")
    st.title("Profile & Subscription")
    if st.session_state.user: st.write(f"Logged in as: `{st.session_state.user['email']}`")
    if st.session_state.is_premium: st.success("Premium Active")
    else:
        st.warning("Free Tier")
        if st.button("Upgrade (100 NGN test)", type="primary"):
            with st.spinner("Connecting…"):
                url, _ = create_payment_link(st.session_state.user["email"], st.session_state.user["localId"])
                if url:
                    st.markdown(f"[Pay Now]({url})", unsafe_allow_html=True)
                    components.html(f'<meta http-equiv="refresh" content="0; url={url}">', height=0)
    if st.button("Back to App"): st.session_state.page = "app"; st.rerun()
    if st.button("Logout"): logout()

# ==============================================================
# 7. MAIN APP
# ==============================================================
elif st.session_state.page == "app" and st.session_state.user:
    st.set_page_config(page_title="PipWizard", page_icon="Chart", layout="wide")

    # ---- payment callback ----
    if "trxref" in st.query_params:
        with st.spinner("Verifying…"):
            verify_payment(st.query_params["trxref"])

    # ---- config ----
    ALL_PAIRS = ["EUR/USD","GBP/USD","USD/JPY","USD/CAD","AUD/USD","NZD/USD",
                 "EUR/GBP","EUR/JPY","GBP/JPY","USD/CHF"]
    FREE_PAIR = "EUR/USD"
    INTERVALS = {"1min":"1min","5min":"5min","15min":"15min","30min":"30min","1h":"1h"}
    OUTPUTSIZE = 500

    # ---- theme ----
    def toggle_theme():
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.markdown(f"""<style>
        .stApp {{background:#{'0e1117' if st.session_state.theme=='dark' else 'ffffff'};
                 color:#{'f0f0f0' if st.session_state.theme=='dark' else '212529'};}}
    </style>""", unsafe_allow_html=True)

    col1, col2 = st.columns([6,1])
    with col1: st.title("PipWizard – Live Forex Signals")
    with col2:
        if st.button("Light" if st.session_state.theme=="dark" else "Dark"):
            toggle_theme(); st.rerun()

    # ---- sidebar ----
    st.sidebar.title("PipWizard")
    st.sidebar.write(f"User: `{st.session_state.user['email']}`")
    premium = st.session_state.is_premium
    pair = st.sidebar.selectbox("Pair", ALL_PAIRS if premium else [FREE_PAIR])
    interval = st.sidebar.selectbox("Timeframe", list(INTERVALS.keys()), index=3)
    strategy = st.sidebar.selectbox("Strategy", [
        "RSI + SMA Crossover","MACD Crossover","RSI + MACD (Confluence)",
        "SMA + MACD (Confluence)","RSI Standalone","SMA Crossover Standalone"
    ])
    show_rsi = st.sidebar.checkbox("Show RSI", True)
    show_macd = st.sidebar.checkbox("Show MACD", True)

    rsi_p = st.sidebar.slider("RSI Period",5,30,14)
    sma_p = st.sidebar.slider("SMA Period",10,50,20)
    rsi_low = st.sidebar.slider("Buy RSI <",20,40,35)
    rsi_high = st.sidebar.slider("Sell RSI >",60,80,65)
    macd_f = st.sidebar.slider("MACD Fast",1,26,12)
    macd_s = st.sidebar.slider("MACD Slow",13,50,26)
    macd_sig = st.sidebar.slider("MACD Signal",1,15,9)

    capital = st.sidebar.number_input("Capital ($)",1000,value=10000)
    risk_pct = st.sidebar.slider("Risk %",0.5,5.0,1.0)/100
    sl_pips = st.sidebar.number_input("SL (pips)",1,200,50)
    tp_pips = st.sidebar.number_input("TP (pips)",1,500,100)

    col_run, col_scan, col_clr = st.sidebar.columns(3)
    run_bt = col_run.button("Backtest", type="primary")
    run_scan = col_scan.button("Scan All", type="secondary")
    if st.session_state.backtest_results and col_clr.button("Clear"):
        st.session_state.backtest_results = None; st.rerun()

    # ---- data fetch ----
    @st.cache_data(ttl=60)
    def fetch_data(sym, intv):
        if "TD_API_KEY" not in st.secrets: return pd.DataFrame()
        td = TDClient(apikey=st.secrets["TD_API_KEY"])
        try:
            ts = td.time_series(symbol=sym, interval=intv, outputsize=OUTPUTSIZE).as_pandas()
            if ts.empty: return pd.DataFrame()
            df = ts[['open','high','low','close']].copy()
            df.index = pd.to_datetime(df.index)
            return df.iloc[::-1]
        except: return pd.DataFrame()

    # ---- indicators & strategy ----
    def add_indicators(df):
        df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_p)
        df['sma'] = df['close'].rolling(sma_p).mean()
        df['macd'], df['macd_sig'], df['macd_hist'] = talib.MACD(df['close'],
                fastperiod=macd_f, slowperiod=macd_s, signalperiod=macd_sig)
        return df

    def apply_strategy(df):
        df['signal'] = 0
        if strategy == "RSI + SMA Crossover":
            df.loc[(df['rsi'] < rsi_low) & (df['close'] > df['sma']), 'signal'] = 1
            df.loc[(df['rsi'] > rsi_high) & (df['close'] < df['sma']), 'signal'] = -1
        elif strategy == "MACD Crossover":
            df.loc[(df['macd'] > df['macd_sig']) & (df['macd'].shift(1) <= df['macd_sig'].shift(1)), 'signal'] = 1
            df.loc[(df['macd'] < df['macd_sig']) & (df['macd'].shift(1) >= df['macd_sig'].shift(1)), 'signal'] = -1
        elif strategy == "RSI + MACD (Confluence)":
            df.loc[(df['rsi'] < rsi_low) & (df['macd'] > df['macd_sig']) &
                   (df['macd'].shift(1) <= df['macd_sig'].shift(1)), 'signal'] = 1
            df.loc[(df['rsi'] > rsi_high) & (df['macd'] < df['macd_sig']) &
                   (df['macd'].shift(1) >= df['macd_sig'].shift(1)), 'signal'] = -1
        elif strategy == "SMA + MACD (Confluence)":
            df.loc[(df['close'] > df['sma']) & (df['macd'] > df['macd_sig']) &
                   (df['macd'].shift(1) <= df['macd_sig'].shift(1)), 'signal'] = 1
            df.loc[(df['close'] < df['sma']) & (df['macd'] < df['macd_sig']) &
                   (df['macd'].shift(1) >= df['macd_sig'].shift(1)), 'signal'] = -1
        elif strategy == "RSI Standalone":
            df.loc[(df['rsi'] < rsi_low) & (df['rsi'].shift(1) >= rsi_low), 'signal'] = 1
            df.loc[(df['rsi'] > rsi_high) & (df['rsi'].shift(1) <= rsi_high), 'signal'] = -1
        elif strategy == "SMA Crossover Standalone":
            df.loc[(df['close'] > df['sma']) & (df['close'].shift(1) <= df['sma'].shift(1)), 'signal'] = 1
            df.loc[(df['close'] < df['sma']) & (df['close'].shift(1) >= df['sma'].shift(1)), 'signal'] = -1
        return df

    # ---- backtest ----
    def backtest(df_in, pair, capital, risk, sl, tp):
        df = df_in.copy(); trades = []
        pip = 0.01 if "JPY" in pair else 0.0001
        risk_usd = capital * risk
        reward_usd = risk_usd * (tp / sl)
        for idx, row in df[df['signal'] != 0].iterrows():
            entry_i = df.index.get_loc(idx) + 1
            if entry_i >= len(df): continue
            entry_price = df.iloc[entry_i]['open']
            sl_price = entry_price - sl*pip if row['signal']==1 else entry_price + sl*pip
            tp_price = entry_price + tp*pip if row['signal']==1 else entry_price - tp*pip
            result, pl, exit = 'OPEN', 0.0, None
            for j in range(entry_i+1, len(df)):
                bar = df.iloc[j]
                if row['signal']==1:
                    if bar['low'] <= sl_price: result, pl, exit = 'LOSS', -risk_usd, bar.name; break
                    if bar['high'] >= tp_price: result, pl, exit = 'WIN', reward_usd, bar.name; break
                else:
                    if bar['high'] >= sl_price: result, pl, exit = 'LOSS', -risk_usd, bar.name; break
                    if bar['low'] <= tp_price: result, pl, exit = 'WIN', reward_usd, bar.name; break
            if result == 'OPEN': result, pl, exit = 'UNRESOLVED', 0.0, df.index[-1]
            trades.append({"entry": df.iloc[entry_i].name, "exit": exit,
                           "type": "BUY" if row['signal']==1 else "SELL",
                           "entry_price": entry_price, "sl": sl_price, "tp": tp_price,
                           "result": result, "pl": pl})
        if not trades: return 0,0,0,0,capital,pd.DataFrame(),pd.DataFrame()
        log = pd.DataFrame(trades).set_index('entry')
        res = log[log['result'].isin(['WIN','LOSS'])].copy()
        if res.empty: return 0,0,0,0,capital,log,res
        wins = len(res[res['result']=='WIN'])
        total = len(res)
        profit = res['pl'].sum()
        gross_win = res[res['pl']>0]['pl'].sum()
        gross_loss = abs(res[res['pl']<0]['pl'].sum())
        pf = gross_win/gross_loss if gross_loss else 999.0
        final = capital + profit
        res['equity'] = capital + res['pl'].cumsum()
        return total, wins/total, profit, pf, final, log, res

    # ---- SCANNER ----
    def run_scanner():
        results = []
        for p in ALL_PAIRS:
            df = fetch_data(p, INTERVALS[interval])
            if df.empty or len(df) < 50: continue
            df = add_indicators(df)
            df = apply_strategy(df)
            df = df.dropna()
            if df.empty: continue
            signals = df[df['signal'] != 0]
            if signals.empty: continue
            last_sig = signals.iloc[-1]
            sig_type = "BUY" if last_sig['signal'] == 1 else "SELL"
            results.append({
                "Pair": p,
                "Signal": sig_type,
                "Time": last_sig.name.strftime("%H:%M"),
                "RSI": f"{last_sig['rsi']:.1f}",
                "Price": f"{last_sig['close']:.5f}"
            })
        return pd.DataFrame(results).sort_values("Pair")

    # ---- load main pair ----
    with st.spinner("Fetching candles…"):
        df = fetch_data(pair, INTERVALS[interval])
    if df.empty: st.error("No data."); st.stop()

    df = add_indicators(df)
    df = apply_strategy(df)
    df = df.dropna()
    if df.empty: st.warning("Not enough data."); st.stop()

    # ==============================================================
    # MAIN CANDLESTICK CHART – FIXED (NO .set(), NO .marker())
    # ==============================================================
    st.markdown("---")
    st.subheader(f"**{pair}** – **{interval}** – {len(df)} candles")

    # Prepare OHLC
    df_lw = df.reset_index()
    ts_col = df_lw.columns[0]
    df_lw['time'] = (df_lw[ts_col].astype('int64') // 1_000_000_000).astype(int)
    ohlc = df_lw[['time','open','high','low','close']].to_dict('records')

    # Create chart with data directly
    chart = StreamlitChart(
        ohlc,
        width=1000,
        height=500,
        layout={
            "backgroundColor": "#0e1117" if st.session_state.theme == "dark" else "#ffffff",
            "textColor": "#f0f0f0" if st.session_state.theme == "dark" else "#212529"
        },
        grid={
            "vertLines": {"color": "#444" if st.session_state.theme == "dark" else "#ddd"},
            "horzLines": {"color": "#444" if st.session_state.theme == "dark" else "#ddd"}
        },
        time_scale={"timeVisible": True, "secondsVisible": False}
    )

    # Buy/Sell markers via Plotly overlay (lightweight-charts has no .marker())
    buy = df[df['signal'] == 1].reset_index()
    sell = df[df['signal'] == -1].reset_index()
    buy['time'] = (buy.iloc[:, 0].astype('int64') // 1_000_000_000).astype(int)
    sell['time'] = (sell.iloc[:, 0].astype('int64') // 1_000_000_000).astype(int)

    overlay = go.Figure()
    if not buy.empty:
        overlay.add_scatter(x=buy['time'], y=buy['low'] * 0.999, mode='markers',
                            marker=dict(symbol='arrow-up', size=14, color='#26a69a'), name='BUY')
    if not sell.empty:
        overlay.add_scatter(x=sell['time'], y=sell['high'] * 1.001, mode='markers',
                            marker=dict(symbol='arrow-down', size=14, color='#ef5350'), name='SELL')
    overlay.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=500, margin=dict(l=0,r=0,t=0,b=0)
    )
    st.plotly_chart(overlay, use_container_width=True, config={'displayModeBar': False})

    # ==============================================================
    # RSI & MACD – Plotly
    # ==============================================================
    if show_rsi:
        st.markdown("---"); st.subheader("RSI")
        fig_rsi = go.Figure()
        fig_rsi.add_scatter(x=df.index, y=df['rsi'], name="RSI", line=dict(color="#ff9800"))
        fig_rsi.add_hline(y=rsi_high, line_dash="dash", line_color="#ef5350")
        fig_rsi.add_hline(y=rsi_low, line_dash="dash", line_color="#26a69a")
        fig_rsi.update_layout(height=200,
                              template='plotly_dark' if st.session_state.theme=="dark" else 'plotly_white')
        st.plotly_chart(fig_rsi, use_container_width=True)

    if show_macd:
        st.markdown("---"); st.subheader("MACD")
        fig_macd = go.Figure()
        fig_macd.add_scatter(x=df.index, y=df['macd'], name="MACD", line=dict(color="#2196f3"))
        fig_macd.add_scatter(x=df.index, y=df['macd_sig'], name="Signal", line=dict(color="#ff9800"))
        fig_macd.add_bar(x=df.index, y=df['macd_hist'], name="Histogram",
                         marker_color=np.where(df['macd_hist']>=0, '#26a69a', '#ef5350'))
        fig_macd.update_layout(height=200,
                              template='plotly_dark' if st.session_state.theme=="dark" else 'plotly_white')
        st.plotly_chart(fig_macd, use_container_width=True)

    # ==============================================================
    # SCANNER
    # ==============================================================
    if run_scan:
        with st.spinner("Scanning all pairs..."):
            scanner_df = run_scanner()
            st.session_state.scanner_results = scanner_df
        st.rerun()

    if st.session_state.scanner_results is not None:
        st.markdown("---")
        st.subheader("Strategy Scanner – Live Signals")
        if st.session_state.scanner_results.empty:
            st.info("No signals across all pairs.")
        else:
            st.dataframe(
                st.session_state.scanner_results.style.apply(
                    lambda row: ['background: #d4edda' if row['Signal']=='BUY' else 'background: #f8d7da'], axis=1
                ),
                use_container_width=True
            )

    # ==============================================================
    # BACKTEST
    # ==============================================================
    if run_bt:
        with st.spinner("Backtesting…"):
            total, wr, profit, pf, final, log, res = backtest(df, pair, capital, risk_pct, sl_pips, tp_pips)
            st.session_state.backtest_results = {
                "total":total, "wr":wr, "profit":profit, "pf":pf, "final":final,
                "log":log, "res":res, "pair":pair, "intv":interval
            }
        st.rerun()

    if st.session_state.backtest_results:
        r = st.session_state.backtest_results
        st.markdown("---"); st.subheader("Backtest Results")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Trades", r["total"])
        c2.metric("Win Rate", f"{r['wr']:.1%}")
        c3.metric("Profit $", f"{r['profit']:,.0f}")
        c4.metric("Profit Factor", f"{r['pf']:.2f}")
        if not r["res"].empty:
            eq = go.Figure()
            eq.add_scatter(x=r["res"].index, y=r["res"]['equity'], mode='lines', line=dict(color='#26a69a'))
            eq.update_layout(height=300,
                             template='plotly_dark' if st.session_state.theme=="dark" else 'plotly_white')
            st.plotly_chart(eq, use_container_width=True)
        st.dataframe(r["log"])

    # ---- auto-refresh ----
    components.html("<meta http-equiv='refresh' content='61'>", height=0)

# ==============================================================
# FALLBACK
# ==============================================================
else:
    st.error("Please log in.")
