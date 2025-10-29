import pandas as pd
import numpy as np
import talib
import math

# --- 1. DATA LOADING AND PREPARATION ---

def load_data():
    """Mocks loading historical EURUSD H4 data."""
    # Create mock data for demonstration
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='4H'),
        'open': np.random.uniform(1.0800, 1.0900, 100).round(5),
        'high': np.random.uniform(1.0900, 1.1000, 100).round(5),
        'low': np.random.uniform(1.0700, 1.0800, 100).round(5),
        'close': np.random.uniform(1.0850, 1.0950, 100).round(5),
        'volume': np.random.randint(1000, 5000, 100)
    }
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Ensure High > Close/Open and Low < Close/Open for realism
    df['high'] = df[['open', 'close']].max(axis=1) + 0.0005
    df['low'] = df[['open', 'close']].min(axis=1) - 0.0005
    
    return df

# --- 2. INDICATOR AND SIGNAL LOGIC ---

def generate_signals(df):
    """Calculates SMA, RSI, and generates BUY/SELL signals."""
    # Parameters
    SMA_PERIOD = 20
    RSI_PERIOD = 14
    
    # Calculate Indicators
    df['SMA'] = talib.SMA(df['close'], timeperiod=SMA_PERIOD)
    df['RSI'] = talib.RSI(df['close'], timeperiod=RSI_PERIOD)
    
    # Generate Signal:
    # 1 (BUY) if RSI crosses above 50 AND price is above SMA
    # -1 (SELL) if RSI crosses below 50 AND price is below SMA
    df['signal'] = 0
    
    # BUY logic
    df.loc[(df['RSI'].shift(1) < 50) & (df['RSI'] > 50) & (df['close'] > df['SMA']), 'signal'] = 1
    
    # SELL logic
    df.loc[(df['RSI'].shift(1) > 50) & (df['RSI'] < 50) & (df['close'] < df['SMA']), 'signal'] = -1
    
    return df

# --- 3. REFINED BACKTESTING FUNCTION ---

def run_backtest(df_in, initial_capital=10000, risk_per_trade=0.01):
    """
    Simulates trades based on 'signal' column and calculates performance metrics.
    Uses realistic candlestick high/low checks for SL/TP execution.
    """
    df = df_in.copy()
    
    # PARAMETERS
    RISK_PIPS_VALUE = 0.0050 # Approx. 50 pips (e.g., for EUR/USD)
    REWARD_PIPS_VALUE = 0.0075 # Approx. 75 pips (1.5x Risk/Reward)
    
    # Simplified Profit/Loss in USD (Risk 1% of $10,000 = $100)
    MAX_RISK_USD = initial_capital * risk_per_trade
    REWARD_USD = MAX_RISK_USD * 1.5
    
    trades = []
    
    # TRADE SIMULATION
    # We iterate up to the second to last row because we need the data from i+1
    for i in range(len(df) - 1): 
        signal = df.iloc[i]['signal']
        
        if signal != 0:
            # Entry occurs at the Open of the NEXT candle
            entry_price = df.iloc[i + 1]['open']
            next_high = df.iloc[i + 1]['high']
            next_low = df.iloc[i + 1]['low']
            
            result = 'NEUTRAL'
            profit_loss = 0.0

            if signal == 1: # BUY Signal
                stop_loss = entry_price - RISK_PIPS_VALUE
                take_profit = entry_price + REWARD_PIPS_VALUE
                
                # SCENARIO 1: TP was hit first (High >= TP AND Low > SL)
                if next_high >= take_profit and next_low > stop_loss:
                    result = 'WIN'
                    profit_loss = REWARD_USD
                
                # SCENARIO 2: SL was hit first (Low <= SL AND High < TP)
                elif next_low <= stop_loss and next_high < take_profit:
                    result = 'LOSS'
                    profit_loss = -MAX_RISK_USD
                
                # SCENARIO 3: Both were hit (Tie-breaker: conservative assumption of SL hit)
                elif next_high >= take_profit and next_low <= stop_loss:
                    result = 'LOSS'
                    profit_loss = -MAX_RISK_USD
            
            elif signal == -1: # SELL Signal
                stop_loss = entry_price + RISK_PIPS_VALUE
                take_profit = entry_price - REWARD_PIPS_VALUE
                
                # SCENARIO 1: TP was hit first (Low <= TP AND High < SL)
                if next_low <= take_profit and next_high < stop_loss:
                    result = 'WIN'
                    profit_loss = REWARD_USD
                
                # SCENARIO 2: SL was hit first (High >= SL AND Low > TP)
                elif next_high >= stop_loss and next_low > take_profit:
                    result = 'LOSS'
                    profit_loss = -MAX_RISK_USD

                # SCENARIO 3: Both were hit (Tie-breaker: conservative assumption of SL hit)
                elif next_low <= take_profit and next_high >= stop_loss:
                    result = 'LOSS'
                    profit_loss = -MAX_RISK_USD
            
            # Record trade only if a clear outcome occurred on the next bar
            if result != 'NEUTRAL':
                trades.append({
                    'entry_time': df.iloc[i+1].name,
                    'signal': 'BUY' if signal == 1 else 'SELL',
                    'entry_price': entry_price,
                    'result': result,
                    'profit_loss': profit_loss
                })

    # METRICS CALCULATION
    trade_df = pd.DataFrame(trades)
    
    if trade_df.empty:
        return 0, 0, 0, 0, initial_capital, trade_df

    total_trades = len(trade_df)
    winning_trades = len(trade_df[trade_df['result'] == 'WIN'])
    total_profit = trade_df['profit_loss'].sum()
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    profit_factor = trade_df[trade_df['profit_loss'] > 0]['profit_loss'].sum() / \
                    abs(trade_df[trade_df['profit_loss'] < 0]['profit_loss'].sum()) if trade_df[trade_df['profit_loss'] < 0]['profit_loss'].sum() != 0 else 999.0
    
    final_capital = initial_capital + total_profit

    return total_trades, win_rate, total_profit, profit_factor, final_capital, trade_df

# --- 4. NEW: REAL-TIME ALERTING FUNCTIONS ---

def send_alert_email(signal_type, price):
    """Mocks sending a real-time email alert to a premium user."""
    print("\n" + "="*50)
    print(f"!!! PREMIUM ALERT: {signal_type.upper()} SIGNAL DETECTED !!!")
    print(f"Instrument: EURUSD (H4)")
    print(f"Price: {price:.5f}")
    print(f"Time: {pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')}")
    print("Email sent to user@premium.com")
    print("="*50 + "\n")

def check_for_live_signal(df):
    """Checks the latest bar of the DataFrame for a BUY or SELL signal."""
    # Assume the last row in the DataFrame is the most recent bar
    latest_bar = df.iloc[-1]
    signal = latest_bar['signal']
    price = latest_bar['close']

    if signal == 1:
        send_alert_email("BUY", price)
    elif signal == -1:
        send_alert_email("SELL", price)
    else:
        print("No new signal detected on the latest bar.")

# --- 5. MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    # Load Data
    historical_data = load_data()
    
    # Generate Signals
    df_with_signals = generate_signals(historical_data)
    
    # Run Backtest
    initial_cap = 10000
    trades_count, win_rate, total_pnl, profit_factor, final_cap, trades_log = run_backtest(df_with_signals, initial_capital=initial_cap)
    
    # Display Backtest Results
    print("\n" + "#"*50)
    print("## BACKTESTING RESULTS")
    print("#"*50)
    print(f"Initial Capital: ${initial_cap:,.2f}")
    print(f"Total Trades: {trades_count}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total Profit/Loss (USD): ${total_pnl:,.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Final Capital: ${final_cap:,.2f}")
    print("#"*50)
    
    # Run Live Signal Check
    print("## LIVE SIGNAL CHECK (New Feature)")
    check_for_live_signal(df_with_signals)
    
    # Display Trade Log (optional for full detail)
    if not trades_log.empty:
        print("\n## Detailed Trade Log (First 5 Trades)")
        print(trades_log.head())
