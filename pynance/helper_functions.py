# Import Dependencies
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Band, Span, NumeralTickFormatter
from datetime import date
import datetime
from hmmlearn import hmm
import numpy as np
from math import pi
from scipy.stats import norm
import streamlit as st
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import yfinance as yf
from Technical_Analysis import simple_moving_average, bollinger, relative_strength_index, on_balance_volume, money_flow_index, ichimoku_kinko_hyo, stochastic_oscillator


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
    warning = []
    for t in ticks:
        try:
            close_prices[t] = yf.download(t, start=one_year_ago, end=today)['Adj Close']

        except:
            warning.append(t)

    close_prices = pd.DataFrame(close_prices)

    return close_prices, warning


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


def regime_detection(ticker, start_date, end_date, SMA, bollinger_bands="No", ikh="No", sub_view="Volitility"):
    historical_price = yf.download(ticker, start=start_date, end=end_date)
    log_ret = np.log1p(historical_price['Adj Close'].pct_change(-1))

    model = hmm.GaussianHMM(n_components=2, covariance_type='diag')
    X = log_ret.dropna().to_numpy().reshape(-1, 1)
    try:
        model.fit(X)  # Viterbi Algo is used to find the max proba, mean and variance

        Z = model.predict(X)
        Z_Close = np.append(Z, False)

        Z2 = pd.DataFrame(Z, index=log_ret.dropna().index, columns=['state'])
        Z2_Close = pd.DataFrame(Z_Close, index=log_ret.index, columns=['state'])

        # dying the returns
        returns_high_volatility = np.empty(len(Z))
        returns_low_volatility = np.empty(len(Z))

        returns_high_volatility[:] = np.nan
        returns_low_volatility[:] = np.nan

        if len(log_ret.dropna()[Z == 0]) > 0 and len(log_ret.dropna()[Z == 1]) > 0:
            if max(log_ret.dropna()[Z == 1]) > max(log_ret.dropna()[Z == 0]):
                returns_high_volatility[Z == 1] = log_ret.dropna()[Z == 1]
                returns_low_volatility[Z == 0] = log_ret.dropna()[Z == 0]

            else:
                returns_high_volatility[Z == 0] = log_ret.dropna()[Z == 0]
                returns_low_volatility[Z == 1] = log_ret.dropna()[Z == 1]

        elif len(log_ret.dropna()[Z == 0]) > 0 and len(log_ret.dropna()[Z == 1]) == 0:
            returns_high_volatility[Z == 1] = np.array([])
            returns_low_volatility[Z == 0] = log_ret.dropna()[Z == 0]

        elif len(log_ret.dropna()[Z == 0]) == 0 and len(log_ret.dropna()[Z == 1]) > 0:
            returns_high_volatility[Z == 0] = np.array([])
            returns_low_volatility[Z == 1] = log_ret.dropna()[Z == 1]

        returns_high_volatility = np.concatenate(([np.nan], returns_high_volatility))
        returns_low_volatility = np.concatenate(([np.nan], returns_low_volatility))

        inc = historical_price["Adj Close"] > historical_price["Open"]
        dec = historical_price["Open"] > historical_price["Adj Close"]
        w = 12 * 60 * 60 * 1000  # half day in ms

        TOOLS = "pan, wheel_zoom, box_zoom, reset, save"

        # Historical Price
        p_historical = figure(x_axis_type="datetime", tools=TOOLS, width=1300, height=400)
        p_historical.xaxis.major_label_orientation = pi / 4
        p_historical.grid.grid_line_alpha = 0.3

        p_historical.segment(historical_price.index, historical_price["High"], historical_price.index,
                             historical_price["Low"], color="black")
        p_historical.vbar(historical_price.index[inc], w, historical_price["Open"][inc],
                          historical_price["Adj Close"][inc], width=w, fill_color="#99FFCC", line_color="black",
                          legend_label="Adjusted Close Price (Inc)")
        p_historical.vbar(historical_price.index[dec], w, historical_price["Open"][dec],
                          historical_price["Adj Close"][dec], width=w, fill_color="#F2583E", line_color="black",
                          legend_label="Adjusted Close Price (Dec)")

        # Simple Moving Average
        if SMA == "Yes":
            historical_price_sma = yf.download(ticker, start=start_date - datetime.timedelta(200), end=end_date)
            slow_sma, fast_sma = simple_moving_average(historical_price)
            green_upper, green_lower, red_upper, red_lower = convergence_divergence(fast_sma, slow_sma)

            green_upper_filtered = con_list(green_upper)
            green_lower_filtered = con_list(green_lower)
            red_upper_filtered = con_list(red_upper)
            red_lower_filtered = con_list(red_lower)

            p_historical.line(slow_sma.index, slow_sma, line_width=1, line_color="red")
            p_historical.line(fast_sma.index, fast_sma, line_width=1, line_color="green")

            for lower, upper in zip(green_lower_filtered, green_upper_filtered):
                green_source = ColumnDataSource({
                    'base': lower.index,
                    'lower': lower,
                    'upper': upper
                })

                green_band = Band(base='base', lower='lower', upper='upper', source=green_source, fill_alpha=0.5,
                                  fill_color="green")
                p_historical.add_layout(green_band)

            for lower, upper in zip(red_lower_filtered, red_upper_filtered):
                red_source = ColumnDataSource({
                    'base': lower.index,
                    'lower': lower,
                    'upper': upper
                })

                red_band = Band(base='base', lower='lower', upper='upper', source=red_source, fill_alpha=0.5,
                                fill_color="red")
                p_historical.add_layout(red_band)

            p_historical.yaxis[0].formatter = NumeralTickFormatter(format="0 a")
            p_historical.yaxis.axis_label = 'Moving Avg.'

        # Bollinger Bands
        if bollinger_bands == "Yes":
            bollinger_upper, bollinger_lower, bollinger_sma = bollinger(historical_price)
            source = ColumnDataSource({
                'base': bollinger_lower.index,
                'lower': bollinger_lower,
                'upper': bollinger_upper
            })

            band = Band(base='base', lower='lower', upper='upper', source=source, fill_color="#99CCFF", fill_alpha=0.4)
            p_historical.line(bollinger_sma.index, bollinger_sma, line_width=1, line_color="#99CCFF")
            p_historical.add_layout(band)

        if ikh == "Yes":
            senkou_a, senkou_b = ichimoku_kinko_hyo(historical_price)
            green_upper, green_lower, red_upper, red_lower = convergence_divergence(senkou_a, senkou_b)

            green_upper_filtered = con_list(green_upper)
            green_lower_filtered = con_list(green_lower)
            red_upper_filtered = con_list(red_upper)
            red_lower_filtered = con_list(red_lower)

            p_historical.line(senkou_a.index, senkou_b, line_width=1, line_color="green")
            p_historical.line(senkou_a.index, senkou_b, line_width=1, line_color="red")

            for lower, upper in zip(green_lower_filtered, green_upper_filtered):
                green_source = ColumnDataSource({
                    'base': lower.index,
                    'lower': lower,
                    'upper': upper
                })

                green_band = Band(base='base', lower='lower', upper='upper', source=green_source, fill_alpha=0.5,
                                  fill_color="green")
                p_historical.add_layout(green_band)

            for lower, upper in zip(red_lower_filtered, red_upper_filtered):
                red_source = ColumnDataSource({
                    'base': lower.index,
                    'lower': lower,
                    'upper': upper
                })

                red_band = Band(base='base', lower='lower', upper='upper', source=red_source, fill_alpha=0.5,
                                fill_color="red")
                p_historical.add_layout(red_band)

            p_historical.yaxis[0].formatter = NumeralTickFormatter(format="0 a")
            p_historical.yaxis.axis_label = 'Volume'

        # Sub View
        p_sub_view = figure(x_axis_type="datetime", x_range=p_historical.x_range, width=1300, height=200)
        p_sub_view.xaxis.major_label_orientation = pi / 4
        p_sub_view.grid.grid_line_alpha = 0.3
        # Volatility
        if sub_view == "Volitility":
            p_sub_view.vbar(x=historical_price.index, top=(np.exp(returns_high_volatility) - 1) * 100, width=w,
                            color="#FFDB46", line_color="black")
            p_sub_view.vbar(x=historical_price.index, top=(np.exp(returns_low_volatility) - 1) * 100, width=w,
                            line_color="black")
            p_sub_view.yaxis.axis_label = 'Return (%)'

        # Relative Strength Index
        if sub_view == "Relative Strength Index":
            rsi = relative_strength_index(historical_price)
            p_sub_view.line(rsi.index, rsi, line_width=1)
            upper_threshold = Span(location=70, dimension='width', line_color='#FF8000', line_width=1, line_alpha=0.5,
                                   line_dash='dashed')
            lower_threshold = Span(location=30, dimension='width', line_width=1, line_alpha=0.5, line_dash='dashed')
            p_sub_view.renderers.extend([upper_threshold, lower_threshold])
            p_sub_view.yaxis.axis_label = 'RSI (%)'

        # On Balance Volume
        elif sub_view == "On-Balance Volume":
            obv, obv_ema = on_balance_volume(historical_price)
            green_upper, green_lower, red_upper, red_lower = convergence_divergence(obv, obv_ema)
            green_upper_filtered = con_list(green_upper)
            green_lower_filtered = con_list(green_lower)
            red_upper_filtered = con_list(red_upper)
            red_lower_filtered = con_list(red_lower)

            p_sub_view.line(obv.index, obv, line_width=1, line_color="green")
            p_sub_view.line(obv_ema.index, obv_ema, line_width=1, line_color="red")

            for lower, upper in zip(green_lower_filtered, green_upper_filtered):
                green_source = ColumnDataSource({
                    'base': lower.index,
                    'lower': lower,
                    'upper': upper
                })

                green_band = Band(base='base', lower='lower', upper='upper', source=green_source, fill_alpha=0.5,
                                  fill_color="green")
                p_sub_view.add_layout(green_band)

            for lower, upper in zip(red_lower_filtered, red_upper_filtered):
                red_source = ColumnDataSource({
                    'base': lower.index,
                    'lower': lower,
                    'upper': upper
                })

                red_band = Band(base='base', lower='lower', upper='upper', source=red_source, fill_alpha=0.5,
                                fill_color="red")
                p_sub_view.add_layout(red_band)

            p_sub_view.yaxis[0].formatter = NumeralTickFormatter(format="0 a")
            p_sub_view.yaxis.axis_label = 'Volume'

        # Stochastic Oscillator
        elif sub_view == "Stochastic Oscillator":
            so = stochastic_oscillator(historical_price)
            p_sub_view.line(so.index, so, line_width=1)
            upper_threshold = Span(location=80, dimension='width', line_color='#FF8000', line_width=1, line_alpha=0.5,
                                   line_dash='dashed')
            lower_threshold = Span(location=20, dimension='width', line_width=1, line_alpha=0.5, line_dash='dashed')
            p_sub_view.renderers.extend([upper_threshold, lower_threshold])
            p_sub_view.yaxis.axis_label = 'SO (%)'

        # Money Flow Index
        elif sub_view == "Money Flow Index":
            mfi = money_flow_index(historical_price)
            p_sub_view.line(mfi.index, mfi, line_width=1)
            upper_threshold = Span(location=80, dimension='width', line_color='#FF8000', line_width=1, line_alpha=0.5,
                                   line_dash='dashed')
            lower_threshold = Span(location=20, dimension='width', line_width=1, line_alpha=0.5, line_dash='dashed')
            p_sub_view.renderers.extend([upper_threshold, lower_threshold])
            p_sub_view.yaxis.axis_label = 'MFI (%)'

        # show the results
        p_historical.legend.location = "top_left"
        p_historical.xaxis.visible = False
        p_historical.yaxis.axis_label = 'Price (USD)'

        return column(p_historical, p_sub_view), returns_high_volatility, returns_low_volatility

    except:
        return None, None, None


def var(ret, cleaned_weights_min_volatility, initial_investment, conf_level=.05):
    cov_matrix = ret.cov()
    avg_return = ret.mean()

    port_mean = avg_return.dot(cleaned_weights_min_volatility)
    port_stdev = np.sqrt(cleaned_weights_min_volatility.T.dot(cov_matrix).dot(cleaned_weights_min_volatility))

    mean_investment = (1 + port_mean) * initial_investment
    stdev_investment = initial_investment * port_stdev

    cutoff1 = norm.ppf(conf_level, mean_investment, stdev_investment)
    var_1d1 = initial_investment - cutoff1

    return var_1d1[0][0]


def consecutive_list(iterable):
    res = []
    for p in iterable:
        if p > 0:
            res.append(True)
        elif p < 0:
            res.append(False)

    volatility_clusters = {}
    count = 0
    st = 0
    for i, j in enumerate(res):
        if i + 1 < len(res):
            if res[i] is not res[i + 1]:
                volatility_clusters[count] = res[st:i + 1]
                count += 1
                st = i + 1

        else:
            volatility_clusters[count] = res[st + 1:]

    if len(volatility_clusters.keys()) > 0:
        n_grp = list(volatility_clusters.keys())[-1]
        list_cluster_len = [len(v) for v in volatility_clusters.values()]
        group_duration_median = np.median(list_cluster_len)
        group_duration_mean = sum(list_cluster_len) / len(volatility_clusters.values())
        net_return = np.expm1(np.nansum(iterable))

    else:
        return 0, 0, 0, 0

    return n_grp, group_duration_median, group_duration_mean, net_return


def con_list(list1):
    res = []
    count = 0
    start = 0
    for i in range(len(list1)):
        if i + 1 < len(list1):
            if np.isnan(list1[i + 1]) and ~np.isnan(list1[i]):
                res.append(list1[start:i + 1])
                count += 1
                start = i + 1

            elif np.isnan(list1[i]) and ~np.isnan(list1[i + 1]):
                start = i + 1

    res.append(list1[start + 1:])

    return res


def convergence_divergence(line1, line2):
    green_upper = []
    green_lower = []
    red_upper = []
    red_lower = []
    for i in range(len(line1)):
        if line1[i] >= line2[i]:
            green_upper.append(line1[i])
            green_lower.append(line2[i])
            red_upper.append(np.nan)
            red_lower.append(np.nan)

        else:
            red_upper.append(line1[i])
            red_lower.append(line2[i])
            green_upper.append(np.nan)
            green_lower.append(np.nan)

    green_upper = pd.Series(green_upper)
    green_lower = pd.Series(green_lower)
    red_upper = pd.Series(red_upper)
    red_lower = pd.Series(red_lower)

    green_upper.index = line1.index
    green_lower.index = line1.index
    red_upper.index = line1.index
    red_lower.index = line1.index

    return green_upper, green_lower, red_upper, red_lower