import time

import streamlit as st

import numpy as np
import pandas as pd

st.set_page_config(
    page_title="自定義網頁標題",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.title('My first App')

@st.cache(suppress_st_warning=True)
def expensive_computation(a):
    st.write(f"沒有快取：expensive_computation({a})")
    time.sleep(2)
    return a * 2

a = st.slider("選擇一個數字", 0, 10)
result = expensive_computation(a)
st.write("結果：", result)

