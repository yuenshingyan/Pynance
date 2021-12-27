# Import Dependencies
from datetime import date
import datetime
import numpy as np
import streamlit as st
import pandas as pd
import yfinance as yf
from helper_functions import get_adj_close_prices, port_opt, regime_detection, var

st.set_page_config(
    page_title="Pynance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {1450}px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )

for key in ["", "Watchlist", "Portfolios"]:
    if key not in st.session_state:
        st.session_state[key] = {}


today = date.today() - datetime.timedelta(1)
add_ticker = st.sidebar.text_input(label="Add To Watchlist", value="Type a stock symbol", key="add_ticker")
if add_ticker not in st.session_state['Watchlist']:
    if add_ticker != "Type a stock symbol":
        days = datetime.timedelta(2)
        three_day_ago = today - days
        close_prices = yf.download(add_ticker, start=three_day_ago, end=today)['Adj Close']
        st.session_state['Watchlist'][add_ticker] = round(close_prices.pct_change(-1).dropna()[-1] * 100, 2)

else:
    st.session_state['Watchlist'].pop(add_ticker)

st.sidebar.text('Watchlist\n')
watchlist_str = "\n".join(
    ["\t" + ticker + "\t" + str(ret) + "%" for ticker, ret in st.session_state['Watchlist'].items()])
st.sidebar.text(watchlist_str)


st.title('Pynance')

# Regime Detection
st.header("Regime Detection")
cols_regime_detection = st.columns([0.005, 0.005, 0.005])
cols_regime_detection2 = st.columns(7)
cols_regime_detection3 = st.columns([1.02, .5, 1.01, .5, .75, .5, 2])
one_year_ago = today - datetime.timedelta(365)

ticker = cols_regime_detection[0].text_input(label="Please type in a stock symbol.", value="AAPL")
start_date = cols_regime_detection[1].date_input("From", one_year_ago)
end_date = cols_regime_detection[2].date_input("To", today, max_value=today)

one_month = cols_regime_detection2[0].button("1 Month")
three_months = cols_regime_detection2[1].button("3 Months")
six_months = cols_regime_detection2[2].button("6 Months")
one_year = cols_regime_detection2[3].button("1 Year")
three_years = cols_regime_detection2[4].button("3 Years")
five_years = cols_regime_detection2[5].button("5 Years")
ten_years = cols_regime_detection2[6].button("10 Years")

SMA = cols_regime_detection3[0].select_slider('Simple Moving Average', options=['No', 'Yes'], value="No")
BB = cols_regime_detection3[2].select_slider('Bollingers Band', options=['No', 'Yes'], value="No")
IKH = cols_regime_detection3[4].select_slider('Ichimoku Kinko Hyo', options=['No', 'Yes'], value="No")
sub_view = cols_regime_detection3[6].selectbox('Sub View', options=['Volitility',
                                                                    'Relative Strength Index',
                                                                    'On-Balance Volume',
                                                                    'Stochastic Oscillator',
                                                                    'Money Flow Index'],
                                               index=0)

buttons = [one_month, three_months, six_months, one_year, three_years, five_years, ten_years]
buttons_val = [30, 90, 180, 365, 1095, 1825, 3650]
for b, bv in zip(buttons, buttons_val):
    if b:
        start_date = today - datetime.timedelta(bv)
        end_date = today

# Regime Detection Inputs
if ticker.isupper() and len(ticker) <= 5:
    try:
        p, returns_high_volatility, returns_low_volatility = regime_detection(ticker, start_date, end_date, SMA, BB,
                                                                              IKH, sub_view)

        if p != None:
            st.bokeh_chart(p, use_container_width=False)

        else:
            st.write("Less than two regime detected, please try a longer date range.")

    except:
        st.write("Selected date range must be greater than one day, try a longer period.")

# ----------------------------------------------------------------Portfolio Optimization---------------------------------------------------------------------------
st.header("Portfolio Optimization")
cols_tickers_from_to_capital = st.columns(3)

default_tickers = "FB, AAPL, AMZN, NFLX, GOOG, MSFT, MSI, AMD, NVDA, F"
tickers = st.text_input(label="Please type in a portfolio", value=default_tickers)
capital = cols_tickers_from_to_capital[0].number_input('Capital', value=10000)
start_date_port_opt = cols_tickers_from_to_capital[1].date_input("From", one_year_ago, key="port_opt")
end_date_port_opt = cols_tickers_from_to_capital[2].date_input("To", today, key="port_opt")

acp, warning = get_adj_close_prices(tickers.split(","), start_date_port_opt, end_date_port_opt)

if warning != []:
    st.write(f"Ticker: {' '.join(warning)} cannot be found.")

cleaned_weights_min_volatility, cleaned_weights_max_sharpe, performance_stats_min_volatility, performance_stats_max_sharpe = port_opt(
    acp)

# Rounding
cleaned_weights = np.hstack([cleaned_weights_min_volatility, cleaned_weights_max_sharpe])
performance_stats = np.vstack([performance_stats_min_volatility, performance_stats_max_sharpe])

cleaned_weights_performance_stats = np.vstack([cleaned_weights, performance_stats.T])

cleaned_weights_performance_stats = pd.DataFrame(cleaned_weights_performance_stats,
                                                 columns=['Min Volatility (%)', 'Max Sharpe (%)'],
                                                 index=tickers.split(",") + ["Expected annual return",
                                                                             "Annual volatility", "Sharpe Ratio"])

cleaned_weights_performance_stats.loc[:, 'Min Volatility (%)'] = cleaned_weights_performance_stats.loc[:,
                                                                 'Min Volatility (%)'] * 100
cleaned_weights_performance_stats.loc[:, 'Max Sharpe (%)'] = cleaned_weights_performance_stats.loc[:,
                                                             'Max Sharpe (%)'] * 100
cleaned_weights_performance_stats.loc[:, 'Min Volatility'] = cleaned_weights_performance_stats.loc[:,
                                                             'Min Volatility (%)'] * capital / (100 * 7.8)
cleaned_weights_performance_stats.loc[:, 'Max Sharpe'] = cleaned_weights_performance_stats.loc[:,
                                                         'Max Sharpe (%)'] * capital / (100 * 7.8)

cleaned_weights_performance_stats.iloc[:-2, :2] = cleaned_weights_performance_stats.iloc[:-2, :2] / 100
cleaned_weights_performance_stats.iloc[:-2, 2:4] = cleaned_weights_performance_stats.iloc[:-2, 2:4] / capital * (100 * 7.8)

cols_port_opt_var = st.columns([6, 4])
cols_port_opt_var[0].subheader("Optimized Portfolio")
cols_port_opt_var[0].dataframe(cleaned_weights_performance_stats.style.format("{:,.2f}"))

# -------------------------------------------------------------Value At Risk---------------------------------------------------------------------------------------
investing_period = end_date_port_opt - start_date_port_opt
cols_port_opt_var[1].subheader("Value At Risk")
choose_condidence_lvl = cols_port_opt_var[1].slider("Confidence Level", .05, .5)

value_at_risk = var(acp.pct_change(-1).dropna(), cleaned_weights_min_volatility, capital, choose_condidence_lvl)
cols_port_opt_var[1].text(
    f"{(1 - choose_condidence_lvl) * 100}% confidence that your portfolio of ${capital}\nwill not exceed losses greater than ${round(value_at_risk, 2)}\nover a {investing_period.days} day period.")
