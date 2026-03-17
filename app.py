import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from scipy.stats import norm

# Настройка страницы
st.set_page_config(page_title="Sigma HFT | Risk Radar", layout="wide", initial_sidebar_state="collapsed")

# Авторефреш страницы каждые 60 секунд (60000 мс)
st_autorefresh(interval=60000, limit=None, key="data_refresh")

# Константы
RISK_FREE_RATE = 0.04
MIN_T_DAYS = 1 / 365.0

# --- БЛОК ЛОГИКИ (Воркер) ---

def filter_rth_today(df):
    if df.empty: return df
    ny_tz = 'America/New_York'
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df.tz_convert(ny_tz)
    today = pd.Timestamp.now(tz=ny_tz).date()
    df = df[df.index.date == today]
    return df[(df.index.hour > 9) | ((df.index.hour == 9) & (df.index.minute >= 30))]

def calculate_greeks(S, K, T, r, sigma, opt_type):
    T = np.maximum(T, MIN_T_DAYS)
    sigma = np.maximum(sigma, 0.001)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    delta = np.where(opt_type == 'call', norm.cdf(d1), norm.cdf(d1) - 1)
    return gamma, delta

# Критически важный декоратор. Обеспечивает выживание под нагрузкой.
@st.cache_data(ttl=60, show_spinner=False)
def fetch_and_calculate_backend(ticker_symbol="^SPX", depth=1):
    sys_time_ny = pd.Timestamp.now(tz='America/New_York').strftime('%H:%M:%S')
    
    # 1. Сбор базового актива
    ticker = yf.Ticker(ticker_symbol)
    try:
        hist_daily = ticker.history(period="5d")
        hist_1m = ticker.history(period="1d", interval="1m")
        prev_val = hist_daily['Close'].iloc[-2] if len(hist_daily) >= 2 else 0
            
        spot_rth = filter_rth_today(hist_1m)
        if not spot_rth.empty:
            spot_val = spot_rth['Close'].iloc[-1]
            open_val = spot_rth['Open'].iloc[1] if len(spot_rth) > 1 else spot_rth['Open'].iloc[0]
        else:
            spot_val = hist_daily['Close'].iloc[-1] if not hist_daily.empty else 0
            open_val = hist_daily['Open'].iloc[-1] if not hist_daily.empty else 0
    except Exception:
        return pd.DataFrame(), 0, 0, 0, sys_time_ny
        
    # 2. Сбор опционов
    expirations = ticker.options
    if not expirations:
        return pd.DataFrame(), spot_val, open_val, prev_val, sys_time_ny
        
    target_exps = expirations[:depth]
    all_chains = []
    today = datetime.date.today()
    
    for i, exp in enumerate(target_exps):
        chain = ticker.option_chain(exp)
        calls, puts = chain.calls, chain.puts
        calls['type'], puts['type'] = 'call', 'put'
        
        T = np.maximum((datetime.datetime.strptime(exp, "%Y-%m-%d").date() - today).days, 0) / 365.0
        df_exp = pd.concat([calls, puts], ignore_index=True)
        df_exp = df_exp[(df_exp['openInterest'] > 0) & (df_exp['impliedVolatility'] > 0.01)].copy()
        
        if df_exp.empty: continue
            
        gamma, delta = calculate_greeks(spot_val, df_exp['strike'], T, RISK_FREE_RATE, df_exp['impliedVolatility'], df_exp['type'])
        
        direction = np.where(df_exp['type'] == 'call', 1, -1)
        notional_mult = spot_val * 100
        
        df_exp['gex'] = gamma * df_exp['openInterest'] * notional_mult * direction
        df_exp['dex'] = delta * df_exp['openInterest'] * notional_mult 
        all_chains.append(df_exp)
        
    final_df = pd.concat(all_chains, ignore_index=True) if all_chains else pd.DataFrame()
    return final_df, spot_val, open_val, prev_val, sys_time_ny

# --- БЛОК ОТОБРАЖЕНИЯ (Фронтенд) ---

# Параметры фиксированы для максимальной скорости (0DTE, SPX)
current_ticker = "^SPX"

raw_df, ticker_spot, ticker_open, ticker_prev, fetch_time_ny = fetch_and_calculate_backend(ticker_symbol=current_ticker, depth=1)

if raw_df.empty:
    st.warning("Нет ликвидности или ожидание данных RTH...")
    st.stop()

# Агрегация уровней
net_gex = raw_df['gex'].sum()
net_dex = raw_df['dex'].sum()

gex_profile = raw_df.groupby('strike')['gex'].sum().reset_index()
dealer_magnet = gex_profile.loc[gex_profile['gex'].idxmax()]['strike'] if not gex_profile.empty else ticker_spot

local_flip = gex_profile[(gex_profile['strike'] >= ticker_spot * 0.95) & (gex_profile['strike'] <= ticker_spot * 1.05)]
gamma_flip = local_flip.loc[local_flip['gex'].abs().idxmin()]['strike'] if not local_flip.empty else ticker_spot

# Интерфейс
st.markdown(f"### Сводка (Обновлено: {fetch_time_ny} NY)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Spot", f"{ticker_spot:.2f}", f"{(ticker_spot - ticker_prev)/ticker_prev * 100:.2f}%")
col2.metric("Net GEX", f"${net_gex/1e9:+.1f}B", "STICKY" if net_gex > 0 else "TRENDING", delta_color="off")
col3.metric("Dealer Magnet", f"{dealer_magnet:.1f}")
col4.metric("Gamma Flip", f"{gamma_flip:.1f}")

# Отрисовка профиля
range_val = 100  # Фиксированный зум для интрадея
min_y = ticker_spot - range_val
max_y = ticker_spot + range_val
mask = (gex_profile['strike'] >= min_y) & (gex_profile['strike'] <= max_y)
plot_df = gex_profile[mask]

fig = go.Figure()

colors = ['#FF453A' if x < 0 else '#32D74B' for x in plot_df['gex']]
fig.add_trace(go.Bar(
    x=plot_df['gex'], 
    y=plot_df['strike'], 
    orientation='h',
    marker_color=colors,
    name='GEX'
))

fig.add_hline(y=ticker_spot, line_color="#00C7FF", line_width=2, annotation_text="Spot")
fig.add_hline(y=dealer_magnet, line_color="#00C7FF", line_dash="dash", annotation_text="Magnet")
fig.add_hline(y=gamma_flip, line_color="#FFD60A", line_width=2, annotation_text="Gamma Flip")

fig.update_layout(
    template="plotly_dark",
    title=f"{current_ticker} 0DTE GEX Profile",
    xaxis_title="Notional GEX ($)",
    yaxis_title="Strike",
    height=750,
    margin=dict(l=0, r=0, t=40, b=0),
    showlegend=False,
    hovermode="y unified"
)

st.plotly_chart(fig, use_container_width=True)