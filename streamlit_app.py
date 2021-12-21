import time

import streamlit as st

import math
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from hmmlearn import hmm
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Stock Regime Detection APP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title('Stock Regime Detection APP')

ticker = st.text_input(label="Please type in a stock symbol.", value="AAPL")

historical_price = yf.download(ticker, start="2011-12-20", end="2021-12-20")

def regime_detection(historical_price):
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

  title = "AAPL" + ' Chart'

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
  p.yaxis.axis_label = 'Adjusted Close Price (USD)'
  p.add_layout(p.legend[0], 'left')
    
  return p 

p = regime_detection(historical_price)
st.bokeh_chart(p, use_container_width=True)

# @st.cache(suppress_st_warning=True)
# st.write("Result：", 10)
