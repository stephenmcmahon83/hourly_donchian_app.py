import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Hourly Donchian + RSI(2) Engine", layout="wide")

st.title("⚡ Hourly Donchian + Cumulative RSI(2) Platform")
st.caption(
    "Quantitative Hourly Trading Engine: Donchian Structural Boundaries + Cumulative RSI(2) Washout Timing. "
    "Execution strictly on the close of 1-hour bars over the full 2-year maximum data window."
)

# ==========================================
# 1. DATA LOADER & INDICATOR CALCULATIONS
# ==========================================

def calculate_wilders_rsi(prices, period=2):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)

@st.cache_data(show_spinner=False)
def fetch_hourly_data(symbol):
    """Pulls the maximum allowable 2 years (~730 days) of 1-hour bars with timezone normalization."""
    df = None
    # 730d is Yahoo's hard ceiling; fallback to 728d if current intraday hour breaches boundary
    for p in ["730d", "728d", "720d"]:
        try:
            df = yf.download(symbol, period=p, interval="1h", auto_adjust=False, progress=False)
            if df is not None and not df.empty:
                break
        except Exception:
            continue

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Convert to Eastern Time and strip timezone to tz-naive (prevents timezone comparison errors)
    if df.index.tz is not None:
        try:
            df.index = df.index.tz_convert('America/New_York').tz_localize(None)
        except Exception:
            df.index = df.index.tz_localize(None)

    return df.dropna()

def prepare_hourly_indicators(df, donchian_window=20, trend_sma=100, rsi_smooth=3):
    df = df.copy()

    # 1. Donchian Channels (Shifted by 1 to prevent lookahead bias)
    df['Donchian_High'] = df['High'].shift(1).rolling(window=donchian_window).max()
    df['Donchian_Low'] = df['Low'].shift(1).rolling(window=donchian_window).min()
    df['Donchian_Mid'] = (df['Donchian_High'] + df['Donchian_Low']) / 2.0

    # 2. Macro Trend Filter
    df['SMA_Trend'] = df['Close'].rolling(window=trend_sma).mean()

    # 3. Wilder's RSI(2) & Cumulative Rolling Smoothed RSI
    df['RSI2'] = calculate_wilders_rsi(df['Close'], period=2)
    df['CumRSI'] = df['RSI2'].rolling(window=rsi_smooth).mean()

    return df.dropna()

# ==========================================
# 2. BAR-CLOSE EXECUTION BACKTEST ENGINE
# ==========================================

def run_hourly_backtest(df, rsi_entry_thresh=15, exit_mode="Donchian Midline", 
                       use_rsi_exit=True, max_hold_bars=21, init_cash=10000):
    trades = []
    in_trade = False

    entry_price = 0.0
    entry_date = None
    bars_held = 0

    for i in range(len(df)):
        current_bar = df.iloc[i]
        date = df.index[i]
        close = current_bar['Close']

        if in_trade:
            bars_held += 1
            exit_hit = False
            exit_reason = ""

            # 1. Stop Loss: Close below Lower Donchian Band
            if close < current_bar['Donchian_Low']:
                exit_hit = True
                exit_reason = "Stop: Donchian Low"

            # 2. Target Exit
            elif exit_mode == "Donchian Midline" and close >= current_bar['Donchian_Mid']:
                exit_hit = True
                exit_reason = "Target: Midline"
            elif exit_mode == "Donchian Upper Band" and close >= current_bar['Donchian_High']:
                exit_hit = True
                exit_reason = "Target: Upper Band"

            # 3. Momentum Snapback Exit
            elif use_rsi_exit and current_bar['RSI2'] >= 70:
                exit_hit = True
                exit_reason = "Momentum: RSI(2) >= 70"

            # 4. Time Stop
            elif bars_held >= max_hold_bars:
                exit_hit = True
                exit_reason = f"Time Stop ({max_hold_bars}h)"

            if exit_hit:
                trade_ret = (close - entry_price) / entry_price * 100
                trades.append({
                    "Entry Date": entry_date.strftime('%Y-%m-%d %H:%M'),
                    "Entry Price": round(entry_price, 2),
                    "Exit Date": date.strftime('%Y-%m-%d %H:%M'),
                    "Exit Price": round(close, 2),
                    "Exit Reason": exit_reason,
                    "Return (%)": round(trade_ret, 2),
                    "Hours Held": bars_held,
                    "Result": "Win" if trade_ret > 0 else "Loss"
                })
                in_trade = False

        if not in_trade:
            # Entry Conditions (Evaluated at bar close)
            is_uptrend = close > current_bar['SMA_Trend']
            is_discounted = close < current_bar['Donchian_Mid']
            is_oversold = current_bar['CumRSI'] <= rsi_entry_thresh

            if is_uptrend and is_discounted and is_oversold:
                in_trade = True
                entry_price = close
                entry_date = date
                bars_held = 0

    trades_df = pd.DataFrame(trades)
    bh_ret = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100

    if trades_df.empty:
        eq = pd.Series([init_cash], index=[df.index[0]])
        return trades_df, {
            "Trades": 0, "Win Rate (%)": 0.0, "Total Return (%)": 0.0,
            "B&H Return (%)": round(bh_ret, 2), "Profit Factor": np.nan, "Max DD (%)": 0.0,
            "Avg Hold (Hours)": 0.0, "Total Bars": len(df),
            "Start Date": df.index[0].strftime('%Y-%m-%d'), "End Date": df.index[-1].strftime('%Y-%m-%d')
        }, eq, eq

    win_rate = (len(trades_df[trades_df['Result'] == 'Win']) / len(trades_df)) * 100
    comp_ret = ((1 + trades_df['Return (%)'] / 100).prod() - 1) * 100

    eq_series = init_cash * (1 + trades_df['Return (%)'] / 100).cumprod()
    eq_series.index = pd.to_datetime(trades_df['Exit Date'])
    drawdowns = (eq_series - eq_series.cummax()) / eq_series.cummax()
    max_dd = drawdowns.min() * 100

    gross_win = trades_df[trades_df['Result'] == 'Win']['Return (%)'].sum()
    gross_loss = abs(trades_df[trades_df['Result'] == 'Loss']['Return (%)'].sum())
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else np.nan

    metrics = {
        "Trades": len(trades_df),
        "Win Rate (%)": round(win_rate, 1),
        "Total Return (%)": round(comp_ret, 2),
        "B&H Return (%)": round(bh_ret, 2),
        "Profit Factor": pf,
        "Max DD (%)": round(max_dd, 1),
        "Avg Hold (Hours)": round(trades_df['Hours Held'].mean(), 1),
        "Total Bars": len(df),
        "Start Date": df.index[0].strftime('%Y-%m-%d'),
        "End Date": df.index[-1].strftime('%Y-%m-%d')
    }
    return trades_df, metrics, eq_series, drawdowns

# ==========================================
# 3. STREAMLIT INTERFACE
# ==========================================

st.sidebar.header("1. Asset & Indicator Setup")
ticker = st.sidebar.text_input("Ticker Symbol", value="QQQ").upper().strip()
donchian_n = st.sidebar.slider("Donchian Lookback (Hours)", min_value=10, max_value=50, value=20)
trend_sma_n = st.sidebar.slider("Trend SMA Filter (Hours)", min_value=50, max_value=200, value=100)
rsi_smooth_n = st.sidebar.slider("RSI(2) Smoothing Window", min_value=1, max_value=5, value=3)

st.sidebar.header("2. Execution Rules")
rsi_thresh = st.sidebar.number_input("CumRSI Entry Threshold", min_value=5.0, max_value=30.0, value=15.0, step=1.0)
target_mode = st.sidebar.selectbox("Primary Profit Target", ["Donchian Midline", "Donchian Upper Band"])
use_rsi_snap = st.sidebar.checkbox("Momentum Exit (RSI(2) >= 70)", value=True)
max_hold = st.sidebar.number_input("Max Hold Time-Stop (Hours)", min_value=7, max_value=70, value=21, step=7)

chart_bars = st.sidebar.slider("Chart View Lookback (Recent Hours)", min_value=100, max_value=700, value=350, step=50)

if st.sidebar.button("Run Hourly Simulation", type="primary", use_container_width=True):
    with st.spinner(f"Downloading maximum 2-year hourly dataset for {ticker}..."):
        raw_hourly = fetch_hourly_data(ticker)

    if raw_hourly is None or len(raw_hourly) < trend_sma_n + 10:
        st.error(f"Could not load sufficient hourly history for {ticker}.")
    else:
        df_proc = prepare_hourly_indicators(raw_hourly, donchian_n, trend_sma_n, rsi_smooth_n)
        trades_df, metrics, eq_series, dd_series = run_hourly_backtest(
            df_proc, rsi_thresh, target_mode, use_rsi_snap, max_hold
        )

        st.info(
            f"📅 **Backtest Sample Loaded:** **{metrics['Total Bars']:,} Hourly Bars** across the full 2-year data window "
            f"(**{metrics['Start Date']}** to **{metrics['End Date']}**)."
        )

        # Performance Summary Cards
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Trades", metrics['Trades'])
        c2.metric("Win Rate", f"{metrics['Win Rate (%)']}%")
        c3.metric("Strategy Return", f"{metrics['Total Return (%)']:,.2f}%")
        c4.metric("B&H Return", f"{metrics['B&H Return (%)']:,.2f}%")
        c5.metric("Profit Factor", f"{metrics['Profit Factor']}")
        c6.metric("Max Drawdown", f"{metrics['Max DD (%)']:.1f}%")

        # --- Interactive Plotly Chart ---
        st.subheader(f"📈 {ticker} Hourly Execution Chart (Last {chart_bars} Hours)")
        plot_df = df_proc.iloc[-chart_bars:]

        fig = go.Figure()

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
            low=plot_df['Low'], close=plot_df['Close'], name=f"{ticker} 1H",
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        ))

        # Donchian Bands
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Donchian_High'], mode='lines', 
                                 line=dict(color='rgba(0, 229, 255, 0.5)', width=1), name="Donchian Upper"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Donchian_Mid'], mode='lines', 
                                 line=dict(color='rgba(255, 255, 255, 0.4)', width=1, dash='dot'), name="Midline"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Donchian_Low'], mode='lines', 
                                 line=dict(color='rgba(255, 82, 82, 0.5)', width=1), name="Donchian Lower"))

        # Trend SMA
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_Trend'], mode='lines', 
                                 line=dict(color='#FFD700', width=1.5), name=f"{trend_sma_n}H SMA Trend"))

        # Render Trades that fall within this recent window
        if not trades_df.empty:
            entry_dts = pd.to_datetime(trades_df['Entry Date'])
            recent_trades = trades_df[entry_dts >= plot_df.index[0]]

            if not recent_trades.empty:
                # Trade Vectors
                for _, tr in recent_trades.iterrows():
                    vec_col = 'rgba(0, 255, 102, 0.6)' if tr['Result'] == 'Win' else 'rgba(255, 51, 102, 0.6)'
                    fig.add_trace(go.Scatter(
                        x=[pd.to_datetime(tr['Entry Date']), pd.to_datetime(tr['Exit Date'])],
                        y=[tr['Entry Price'], tr['Exit Price']],
                        mode='lines',
                        line=dict(color=vec_col, width=2, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                # Entry Markers
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(recent_trades['Entry Date']), y=recent_trades['Entry Price'],
                    mode='markers', name='Entry (Bar Close)',
                    marker=dict(symbol='triangle-up', size=11, color='#00FF66', line=dict(color='white', width=1)),
                    hovertext=[f"Buy: ${p:.2f}<br>{d}" for p, d in zip(recent_trades['Entry Price'], recent_trades['Entry Date'])],
                    hoverinfo='text'
                ))

                # Exit Markers
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(recent_trades['Exit Date']), y=recent_trades['Exit Price'],
                    mode='markers', name='Exit (Bar Close)',
                    marker=dict(symbol='triangle-down', size=11, color='#FF3366', line=dict(color='white', width=1)),
                    hovertext=[f"Exit: ${p:.2f}<br>{r} ({ret:+.2f}%)" for p, r, ret in zip(recent_trades['Exit Price'], recent_trades['Exit Reason'], recent_trades['Return (%)'])],
                    hoverinfo='text'
                ))

        fig.update_layout(template="plotly_dark", height=520, xaxis_rangeslider_visible=False,
                          margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Equity Curve & Drawdown
        col_eq1, col_eq2 = st.columns(2)
        with col_eq1:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=eq_series.index, y=eq_series.values, mode='lines', line=dict(color='#00FF66')))
            fig_eq.update_layout(title="Equity Compounding ($)", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_eq, use_container_width=True)
        with col_eq2:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values * 100, fill='tozeroy', mode='lines', line=dict(color='#FF3366')))
            fig_dd.update_layout(title="Drawdown Profile (%)", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_dd, use_container_width=True)

        # Trade Log Table
        st.subheader("Hourly Trade Audit Log")
        st.dataframe(
            trades_df.sort_values(by="Entry Date", ascending=False),
            column_config={
                "Entry Price": st.column_config.NumberColumn(format="$%.2f"),
                "Exit Price": st.column_config.NumberColumn(format="$%.2f"),
                "Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "Hours Held": st.column_config.NumberColumn(format="%d h")
            },
            use_container_width=True
        )
