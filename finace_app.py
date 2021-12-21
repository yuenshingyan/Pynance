# Import Dependencies
import time
from datetime import date
import datetime
import streamlit as st

import math
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from hmmlearn import hmm
import yfinance as yf
import pandas as pd
import numpy as np

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Helper Functions
def get_datetime(past_days=365):
  today = date.today()
  days = datetime.timedelta(past_days)
  one_year_ago = today - days

  return str(one_year_ago), str(today)

# Adjusted Close Prices
@st.cache(suppress_st_warning=True)
def get_adj_close_prices(ticks, one_year_ago, today):
  close_prices = {}
  for t in ticks:
    close_prices[t] = yf.download(t, start=one_year_ago, end=today)['Adj Close']

  close_prices = pd.DataFrame(close_prices)

  return close_prices

def port_opt(acp):
  # Calculate expected returns and sample covariance
  mu = expected_returns.mean_historical_return(acp)
  S = risk_models.sample_cov(acp)

  # Optimize for maximal Sharpe ratio
  ef_min_volatility = EfficientFrontier(mu, S)
  ef_max_sharpe = EfficientFrontier(mu, S)

  # Set Constriants
  raw_weights_min_volatility = ef_min_volatility.min_volatility()
  raw_weights_max_sharpe = ef_max_sharpe.max_sharpe()

  # Store weights
  cleaned_weights_min_volatility = ef_min_volatility.clean_weights()
  cleaned_weights_max_sharpe = ef_max_sharpe.clean_weights()

  # Turn Weights Into Pandas Dataframes
  cleaned_weights_min_volatility = pd.DataFrame(
      cleaned_weights_min_volatility.values(), 
      index=cleaned_weights_min_volatility, 
      columns=['Min Volatility'])
  
  cleaned_weights_max_sharpe = pd.DataFrame(
      cleaned_weights_max_sharpe.values(), 
      index=cleaned_weights_max_sharpe, 
      columns=['Max Sharpe'])

  # Store Performance Stats
  performance_stats_min_volatility = ef_min_volatility.portfolio_performance()
  performance_stats_max_sharpe = ef_max_sharpe.portfolio_performance()

  return cleaned_weights_min_volatility, cleaned_weights_max_sharpe, performance_stats_min_volatility, performance_stats_max_sharpe

def regime_detection(historical_price, ticker):
  log_ret = np.log1p(historical_price['Adj Close'].pct_change(-1))

  model = hmm.GaussianHMM(n_components=2, covariance_type='diag')
  X = log_ret.dropna().to_numpy().reshape(-1, 1)
  model.fit(X) # Viterbi Algo is used to find the max proba, mean and variance
  Z = model.predict(X)
  Z_Close = np.append(Z, False)

  Z2 = pd.DataFrame(Z, index=log_ret.dropna().index, columns=['state'])
  Z2_Close = pd.DataFrame(Z_Close, index=log_ret.index, columns=['state'])

  # dying the close prices
  close_high_volatility = historical_price[Z_Close == 0]
  close_low_volatility = historical_price[Z_Close == 1]

  # dying the returns
  returns_high_volatility = np.empty(len(Z))
  returns_low_volatility = np.empty(len(Z))

  returns_high_volatility[:] = np.nan
  returns_low_volatility[:] = np.nan

  returns_high_volatility[Z == 0] = log_ret.dropna()[Z == 0]
  returns_low_volatility[Z == 1] = log_ret.dropna()[Z == 1]

  w = 12 * 60 * 60 * 1000 # half day in ms

  TOOLS = "pan, wheel_zoom, box_zoom, reset, save"

  title = ticker + ' Historical Price'

  p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1300, title = title)

  p.xaxis.major_label_orientation = math.pi/4

  p.grid.grid_line_alpha=0.3

  inc_high_volatility = close_high_volatility["Adj Close"] > close_high_volatility["Open"]
  dec_high_volatility = close_high_volatility["Open"] > close_high_volatility["Adj Close"]

  p.segment(close_high_volatility.index, 
            close_high_volatility["High"], 
            close_high_volatility.index, 
            close_high_volatility["Low"], 
            color="black", 
            line_width=0.1)
  
  p.vbar(close_high_volatility.index[inc_high_volatility], 
        w, 
        close_high_volatility["Open"][inc_high_volatility], 
        close_high_volatility["Adj Close"][inc_high_volatility], 
        fill_color="#FFEE49",
        line_color="#FFEE49", 
        line_width=0.1, 
        legend_label="High Volatility (Inc)")
  
  p.vbar(close_high_volatility.index[dec_high_volatility], 
        w, 
        close_high_volatility["Open"][dec_high_volatility], 
        close_high_volatility["Adj Close"][dec_high_volatility], 
        fill_color="#49C9FF",
        line_color="#49C9FF", 
        line_width=0.1, 
        legend_label="High Volatility (Dec)")
  
  inc_low_volatility = close_low_volatility["Adj Close"] > close_low_volatility["Open"]
  dec_low_volatility = close_low_volatility["Open"] > close_low_volatility["Adj Close"]
  
  p.segment(close_low_volatility.index, 
            close_low_volatility["High"], 
            close_low_volatility.index, 
            close_low_volatility["Low"], 
            color="black", 
            line_width=0.1)
  
  p.vbar(close_low_volatility.index[inc_low_volatility], 
        w, 
        close_low_volatility["Open"][inc_low_volatility], 
        close_low_volatility["Adj Close"][inc_low_volatility], 
        fill_color="#D5E1DD",
        line_color="#D5E1DD", 
        line_width=0.1, 
        legend_label="Low Volatility (Inc)")
  
  p.vbar(close_low_volatility.index[dec_low_volatility], 
        w, 
        close_low_volatility["Open"][dec_low_volatility], 
        close_low_volatility["Adj Close"][dec_low_volatility], 
        fill_color="#F2583E",
        line_color="#F2583E", 
        line_width=0.1, 
        legend_label="Low Volatility (Dec)")

  p.xaxis.axis_label = 'Date'
  p.yaxis.axis_label = 'Price (USD)'
  p.add_layout(p.legend[0], 'left')
    
  return p 

def sidebar_plot(adj_price):
  if adj_price.iloc[-1] > adj_price.iloc[0]:
    color = "Green"

  elif adj_price.iloc[-1] < adj_price.iloc[0]:
    color = "Red"

  fig, ax = plt.subplots(figsize=(1, 1))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.plot(adj_price, color=color)

  return fig

st.set_page_config(
    page_title="Pynance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Main
st.title('Pynance')

# Regime Detection 
st.header("Regime Detection")
cols_name = st.columns(3)

ticker = cols_name[0].text_input(label="Please type in a stock symbol.", value="AAPL")

today = date.today()
days = datetime.timedelta(365)
one_year_ago = today - days

start_date = cols_name[1].date_input("From", one_year_ago)
end_date = cols_name[2].date_input("To", today)  

historical_price = yf.download(ticker, start=start_date, end=end_date)

p = regime_detection(historical_price, ticker)
st.bokeh_chart(p, use_container_width=True)

# Portfolio Optimization
st.header("Portfolio Optimization")
cols_name2 = st.columns(4)
default_tickers = "FB, AAPL, AMZN, NFLX, GOOG"
tickers = cols_name2[0].text_input(label="Please type in a portfolio", value=default_tickers)
start_date_port_opt = cols_name2[1].date_input("From", one_year_ago, key="port_opt")
end_date_port_opt = cols_name2[2].date_input("To", today, key="port_opt")
capital = cols_name2[3].number_input('Capital', value=10000)

acp = get_adj_close_prices(tickers.split(","), start_date, end_date)
cleaned_weights_min_volatility, cleaned_weights_max_sharpe, performance_stats_min_volatility, performance_stats_max_sharpe = port_opt(acp)

display_format = st.radio("", ('Percentages', 'Fractions Of Capital'))

# Rounding
cleaned_weights_min_volatility_pct = round(cleaned_weights_min_volatility * 100, 2)
cleaned_weights_max_sharpe_pct = round(cleaned_weights_max_sharpe * 100, 2)
port_max_sharpe_pct = np.hstack([cleaned_weights_min_volatility_pct, cleaned_weights_max_sharpe_pct])
port_max_sharpe_pct = pd.DataFrame(port_max_sharpe_pct, columns=["Min Volatility", "Max Sharpe"], index=tickers.split(","))

cleaned_weights_min_volatility_capital = round(cleaned_weights_min_volatility * capital, 2)
cleaned_weights_max_sharpe_capital = round(cleaned_weights_max_sharpe * capital, 2)
port_max_sharpe_capital = np.hstack([cleaned_weights_min_volatility_capital, cleaned_weights_max_sharpe_capital])
port_max_sharpe_capital = pd.DataFrame(port_max_sharpe_capital, columns=["Min Volatility", "Max Sharpe"], index=tickers.split(","))

performance_stats = pd.DataFrame([performance_stats_min_volatility, performance_stats_max_sharpe], 
             index=['Min Volatility', 'Max Sharpe'], 
             columns=["Expected annual return", "Annual volatility", "Sharpe Ratio"]).T

cols_name3 = st.columns(3)

if 'Watchlist' not in st.session_state:
    st.session_state['Watchlist'] = []  
    
if display_format == "Percentages":
    performance_stats.iloc[0, :] = performance_stats.iloc[0, :] * 100
    cols_name3[0].subheader("Optimized Portfolio")
    cols_name3[1].subheader("Performance Stats")
    cols_name3[0].dataframe(port_max_sharpe_pct)
    cols_name3[1].dataframe(performance_stats)
    
elif display_format == "Fractions Of Capital":
    performance_stats.iloc[0, :] = performance_stats.iloc[0, :] * capital
    cols_name3[0].subheader("Optimized Portfolio")
    cols_name3[1].subheader("Performance Stats")
    cols_name3[0].dataframe(port_max_sharpe_capital)
    cols_name3[1].dataframe(performance_stats)
 
add_ticker = st.sidebar.text_input(label="Add To Watchlist", value="Type a stock symbol", key="add_ticker")    
if add_ticker not in st.session_state['Watchlist']:
  if add_ticker != "Type a stock symbol":
    st.session_state['Watchlist'].append(add_ticker)
    
else:
  st.session_state['Watchlist'].remove(add_ticker)
    
st.sidebar.text('Watchlist\n')
st.sidebar.text("\n".join(st.session_state['Watchlist']))

for t in st.session_state['Watchlist']:
  st.sidebar.pyplot(sidebar_plot(t["Adj Close"]))
