import time

import streamlit as st

import math
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
output_notebook()

import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm

st.set_page_config(
    page_title="自定義網頁標題",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.title('My first App')

@st.cache(suppress_st_warning=True)

a = st.slider("選擇一個數字", 0, 10)
result = a
st.write("結果：", result)

