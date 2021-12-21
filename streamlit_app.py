import time

import streamlit as st

import math
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
from hmmlearn import hmm
import yfinance as yf
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="自定義網頁標題",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.title('Stock Regime Detection APP')

ticker = st.text_input(label="Please type in a stock symbol.", value="AAPL")

historical_price = yf.download(ticker, start="2011-12-20", end="2021-12-20")

st.write(ticker)


# @st.cache(suppress_st_warning=True)
# st.write("Result：", 10)
