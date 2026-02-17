import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. Page Config
st.set_page_config(page_title="Gold AI Bot", layout="wide")
st.title("🚀 XAUUSD Smart Money AI")

# 2. Fetch Data (Alignment-Proof)
df = yf.download("GC=F", period="60d", interval="1h", multi_level_index=False)

# 3. Institutional Logic
df['Vol_Avg'] = df['Volume'].rolling(20).mean()
df['OB'] = df['Volume'] > (df['Vol_Avg'] * 3.0) # 300% Spike

# 4. Dashboard Visuals
st.subheader("Gold Price Action (1H)")
st.line_chart(df['Close'].tail(50))

# 5. Signal Detection
latest_ob = df[df['OB']].tail(1)
if not latest_ob.empty:
    st.success(f"🎯 INSTITUTIONAL ORDER BLOCK AT: {latest_ob['Low'].values[0]:.2f}")
    st.info("Strategy: Wait for price to return to this zone for a high-probability entry.")
else:
    st.warning("⏳ Market is quiet. No major Smart Money volume detected.")
