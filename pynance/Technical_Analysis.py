# Import Dependencies
import numpy as np
import pandas as pd


def simple_moving_average(historical_price, window_slow=200, window_fast=50):
    slow_sma = historical_price['Close'].rolling(window_slow).mean()
    fast_sma = historical_price['Close'].rolling(window_fast).mean()

    return slow_sma, fast_sma


def bollinger(historical_price, window=14, m=2):
    historical_price.loc[:, 'Typical Price'] = (historical_price["High"] + historical_price["Low"] + historical_price[
        "Adj Close"]) / 3

    bollinger_sma = historical_price.loc[:, 'Typical Price'].rolling(window=window).mean()
    bollinger_std = historical_price.loc[:, 'Typical Price'].rolling(window=window).std()

    bollinger_upper = bollinger_sma + bollinger_std * m
    bollinger_lower = bollinger_sma - bollinger_std * m

    return bollinger_upper, bollinger_lower, bollinger_sma


def relative_strength_index(historical_price):
    historical_return = historical_price["Adj Close"].pct_change(-1).dropna()
    avg_gain = pd.Series(np.where(historical_return > 0, historical_return, 0)).rolling(14).mean()
    avg_loss = pd.Series(np.where(historical_return < 0, abs(historical_return), 0)).rolling(14).mean()
    RSI = 100 - (100 / (1 + avg_gain / avg_loss))
    RSI = RSI.append(pd.Series(np.nan))
    RSI.index = historical_price.index

    return RSI


def on_balance_volume(historical_price, window=20):
    obv = []
    obv.append(0)

    for i in range(1, len(historical_price['Close'])):
        if historical_price["Close"].iloc[i] > historical_price["Close"].iloc[i - 1]:
            obv.append(obv[-1] + historical_price["Volume"].iloc[i])
        elif historical_price["Close"].iloc[i] < historical_price["Close"].iloc[i - 1]:
            obv.append(obv[-1] - historical_price["Volume"].iloc[i])
        else:
            obv.append(obv[-1])

    obv_ema = pd.Series(obv).ewm(span=window).mean()
    obv_ema.index = historical_price.index

    obv = pd.Series(obv)
    obv.index = historical_price.index

    return obv, obv_ema


def money_flow_index(historical_price, window=14):
    TypicalPrice = (historical_price['High'] + historical_price['Low'] + historical_price['Close']) / 3
    RawMoneyFlow = TypicalPrice * historical_price['Volume']
    MoneyFlowRatio = pd.Series(np.where(RawMoneyFlow.diff(1) > 0, RawMoneyFlow, 0)).rolling(14).sum() / pd.Series(
        np.where(RawMoneyFlow.diff(1) < 0, RawMoneyFlow, 0)).rolling(window).sum()
    MFI = 100 - 100 / (1 + MoneyFlowRatio)
    MFI.index = historical_price.index

    return MFI

def ichimoku_kinko_hyo(historical_price, tenkan_window=9, kijun_window=26, senkou_b_window=52):
    # Tenkan-Sen
    tenkan_sen = (historical_price['High'].rolling(tenkan_window).max() + historical_price['Low'].rolling(
        tenkan_window).min()) / 2

    # Kijun-Sen
    kijun_sen = (historical_price['High'].rolling(kijun_window).max() + historical_price['Low'].rolling(
        kijun_window).min()) / 2

    # Chikou Span
    chikou_span = historical_price['Close'].shift(-26)

    # Senkou A
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou B
    senkou_b = ((historical_price['High'].rolling(senkou_b_window).max() + historical_price['Low'].rolling(
        senkou_b_window).min()) / 2).shift(52)

    shade = np.where(senkou_a >= senkou_b, 1, 0)

    return senkou_a, senkou_b


def stochastic_oscillator(historical_price, window_low=14, window_high=14):
    C = historical_price['Close'].shift(1)
    L14 = historical_price['Low'].rolling(window_low).min()
    H14 = historical_price['High'].rolling(window_high).max()
    K = ((C - L14) / (H14 - L14)) * 100

    return K


