import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Hourly Mean-Reversion Donchian Engine", layout="wide")

st.title("⚡ Hourly Mean-Reversion Donchian Close Platform")
st.caption(
    "Long-Only Mean-Reversion Engine: Buy X-Hour Lowest Close (Dip Flush), Sell Y-Hour Highest Close (Snapback Bounce). "
    "Execution strictly on the close of 1-hour bars (Max 2-Year Yahoo Dataset)."
)

# ==========================================
# 1. DATA LOADER & INDICATOR CALCULATIONS
# ==========================================

@st.cache_data(show_spinner=False)
def fetch_hourly_data(symbol):
    """Pulls the maximum allowable 2 years (~730 days) of 1-hour bars with timezone normalization."""
    df = None
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

def prepare_reversed_donchian_channels(df, entry_window=20, exit_window=5, trend_sma=0):
    df = df.copy()

    # Shift by 1 bar to strictly eliminate lookahead bias
    df['Entry_Lower_Close'] = df['Close'].shift(1).rolling(window=entry_window).min()
    df['Exit_Upper_Close'] = df['Close'].shift(1).rolling(window=exit_window).max()

    if trend_sma > 0:
        df['SMA_Trend'] = df['Close'].rolling(window=trend_sma).mean()
    else:
        df['SMA_Trend'] = np.nan

    return df.dropna(subset=['Entry_Lower_Close', 'Exit_Upper_Close'])

# ==========================================
# 2. BAR-CLOSE EXECUTION BACKTEST ENGINE
# ==========================================

def run_reversed_donchian_backtest(df, use_trend_filter=False, max_hold_bars=0, stop_loss_pct=0.0, init_cash=10000):
    trades = []
    in_trade = False

    entry_price = 0.0
    entry_date = None
    bars_held = 0

    for i in range(len(df)):
        current_bar = df.iloc[i]
        date = df.index[i]
        close = current_bar['Close']
        entry_trigger = current_bar['Entry_Lower_Close']
        exit_trigger = current_bar['Exit_Upper_Close']

        if in_trade:
            bars_held += 1
            exit_hit = False
            exit_reason = ""

            # 1. Primary Exit: Hourly Close prints above prior Y-hour highest close
            if close > exit_trigger:
                exit_hit = True
                exit_reason = "Highest Close Target"

            # 2. Optional Emergency Stop Loss (% Drawdown)
            elif stop_loss_pct > 0 and (close - entry_price) / entry_price * 100 <= -stop_loss_pct:
                exit_hit = True
                exit_reason = f"Stop Loss (-{stop_loss_pct:.1f}%)"

            # 3. Optional Time Stop (Max Holding Bars)
            elif max_hold_bars > 0 and bars_held >= max_hold_bars:
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
            # Trend Filter Check
            trend_passed = True
            if use_trend_filter and not pd.isna(current_bar['SMA_Trend']):
                if close <= current_bar['SMA_Trend']:
                    trend_passed = False

            # Reversed Entry Rule: Buy when close flushes below prior X-hour lowest close
            if close < entry_trigger and trend_passed:
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
# 3. STREAMLIT USER INTERFACE
# ==========================================

st.sidebar.header("1. Asset & Lookback")
ticker = st.sidebar.text_input("Ticker Symbol", value="QQQ").upper().strip()

st.sidebar.header("2. Mean-Reversion Close Channels")
entry_window = st.sidebar.number_input(
    "Entry Lookback: X-Hour Lowest Close", 
    min_value=2, max_value=200, value=20, step=1,
    help="Triggers a Long Buy when the hourly close drops below the lowest close of the prior X hours."
)

exit_window = st.sidebar.number_input(
    "Exit Lookback: Y-Hour Highest Close", 
    min_value=2, max_value=200, value=5, step=1,
    help="Triggers an Exit when the hourly close pushes above the highest close of the prior Y hours."
)

with st.sidebar.expander("⚙️ Advanced Risk & Trend Filters", expanded=True):
    use_trend_filter = st.checkbox(
        "Require Close > Trend SMA (Bull Filter)", 
        value=True, 
        help="Only buy washouts when the stock is in an overall secular uptrend."
    )
    trend_sma = st.number_input("Trend SMA Lookback (Hours)", min_value=20, max_value=500, value=150, step=10) if use_trend_filter else 0

    use_stop_loss = st.checkbox("Enable Protective Stop-Loss (%)", value=False)
    stop_loss_pct = st.number_input("Stop Loss Percentage", min_value=0.5, max_value=20.0, value=3.0, step=0.5) if use_stop_loss else 0.0

    use_time_stop = st.checkbox("Enable Max Hold Time-Stop", value=False)
    max_hold_bars = st.number_input("Max Hold Duration (Hours)", min_value=7, max_value=150, value=35, step=7) if use_time_stop else 0

chart_bars = st.sidebar.slider("Chart View (Recent Hours)", min_value=100, max_value=800, value=350, step=50)

if st.sidebar.button("Run Reversed Donchian Simulation", type="primary", use_container_width=True):
    with st.spinner(f"Downloading 2-year hourly dataset for {ticker}..."):
        raw_hourly = fetch_hourly_data(ticker)

    if raw_hourly is None or len(raw_hourly) < max(entry_window, exit_window, trend_sma) + 10:
        st.error(f"Could not load sufficient hourly history for {ticker}.")
    else:
        df_proc = prepare_reversed_donchian_channels(raw_hourly, entry_window, exit_window, trend_sma)
        trades_df, metrics, eq_series, dd_series = run_reversed_donchian_backtest(
            df_proc, use_trend_filter, max_hold_bars, stop_loss_pct
        )

        st.info(
            f"📅 **Backtest Sample Scope:** **{metrics['Total Bars']:,} Hourly Bars** evaluated "
            f"from **{metrics['Start Date']}** to **{metrics['End Date']}**."
        )

        # Performance Summary Cards
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Trades", metrics['Trades'])
        c2.metric("Win Rate", f"{metrics['Win Rate (%)']}%")
        c3.metric("Strategy Return", f"{metrics['Total Return (%)']:,.2f}%")
        c4.metric("B&H Return", f"{metrics['B&H Return (%)']:,.2f}%")
        c5.metric("Profit Factor", f"{metrics['Profit Factor']}")
        c6.metric("Max Drawdown", f"{metrics['Max DD (%)']:.1f}%")

        # --- Interactive Execution Chart ---
        st.subheader(f"📈 {ticker} Mean-Reversion Execution Chart (Last {chart_bars} Hours)")
        plot_df = df_proc.iloc[-chart_bars:]

        fig = go.Figure()

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
            low=plot_df['Low'], close=plot_df['Close'], name=f"{ticker} 1H",
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        ))

        # Entry Trigger Line: Lowest Close
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['Entry_Lower_Close'], mode='lines',
            line=dict(color='#00E5FF', width=1.5),
            name=f"Entry Trigger ({entry_window}h Lowest Close)"
        ))

        # Exit Target Line: Highest Close
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['Exit_Upper_Close'], mode='lines',
            line=dict(color='#FFA726', width=1.5, dash='dash'),
            name=f"Exit Target ({exit_window}h Highest Close)"
        ))

        # Trend SMA
        if use_trend_filter and 'SMA_Trend' in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df['SMA_Trend'], mode='lines',
                line=dict(color='#FFD700', width=1.5),
                name=f"{trend_sma}h Trend SMA"
            ))

        # Render Trades
        if not trades_df.empty:
            entry_dts = pd.to_datetime(trades_df['Entry Date'])
            recent_trades = trades_df[entry_dts >= plot_df.index[0]]

            if not recent_trades.empty:
                # Trade Vectors (Connecting Entry to Exit)
                for _, tr in recent_trades.iterrows():
                    vec_col = 'rgba(0, 255, 102, 0.7)' if tr['Result'] == 'Win' else 'rgba(255, 51, 102, 0.7)'
                    fig.add_trace(go.Scatter(
                        x=[pd.to_datetime(tr['Entry Date']), pd.to_datetime(tr['Exit Date'])],
                        y=[tr['Entry Price'], tr['Exit Price']],
                        mode='lines',
                        line=dict(color=vec_col, width=2, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                # Entry Markers (Long Buy at the flush)
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(recent_trades['Entry Date']), y=recent_trades['Entry Price'],
                    mode='markers', name='Buy (Washout Close)',
                    marker=dict(symbol='triangle-up', size=11, color='#00FF66', line=dict(color='white', width=1)),
                    hovertext=[f"Buy Washout: ${p:.2f}<br>{d}" for p, d in zip(recent_trades['Entry Price'], recent_trades['Entry Date'])],
                    hoverinfo='text'
                ))

                # Exit Markers (Sell into strength)
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(recent_trades['Exit Date']), y=recent_trades['Exit Price'],
                    mode='markers', name='Sell (Bounce Close)',
                    marker=dict(symbol='triangle-down', size=11, color='#FF3366', line=dict(color='white', width=1)),
                    hovertext=[f"Exit: ${p:.2f}<br>{r} ({ret:+.2f}%)" for p, r, ret in zip(recent_trades['Exit Price'], recent_trades['Exit Reason'], recent_trades['Return (%)'])],
                    hoverinfo='text'
                ))

        fig.update_layout(template="plotly_dark", height=520, xaxis_rangeslider_visible=False,
                          margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Equity Curve & Drawdown Charts
        col_eq1, col_eq2 = st.columns(2)
        with col_eq1:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=eq_series.index, y=eq_series.values, mode='lines', line=dict(color='#00FF66')))
            fig_eq.update_layout(title="Compounded Account Equity ($)", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_eq, use_container_width=True)
        with col_eq2:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values * 100, fill='tozeroy', mode='lines', line=dict(color='#FF3366')))
            fig_dd.update_layout(title="Peak-to-Trough Drawdown (%)", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
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
