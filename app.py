import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import streamlit as st
import yfinance as yf

st.set_page_config(page_title='Portfolio Optimization & Risk Lab', layout='wide')

TRADING_DAYS = 252


def clean_tickers(raw: str):
    tickers = [t.strip().upper() for t in raw.replace('\n', ',').split(',') if t.strip()]
    # preserve order, remove duplicates
    return list(dict.fromkeys(tickers))


@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            prices = data['Close'].copy()
        else:
            prices = data.xs(data.columns.levels[0][0], axis=1, level=0).copy()
    else:
        prices = data.to_frame(name=tickers[0]) if len(tickers) == 1 else data.copy()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices = prices.dropna(how='all').ffill().dropna(axis=1, how='all')
    return prices


@st.cache_data(show_spinner=False)
def load_benchmark(benchmark, start, end):
    data = yf.download(benchmark, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            s = data['Close'].squeeze()
        else:
            s = data.iloc[:, 0]
    else:
        s = data['Close'] if 'Close' in data.columns else data.squeeze()
    return s.dropna()


def annualized_stats(returns: pd.DataFrame):
    mean_returns = returns.mean() * TRADING_DAYS
    cov_matrix = returns.cov() * TRADING_DAYS
    return mean_returns, cov_matrix


def portfolio_performance(weights, mean_returns, cov_matrix, rf):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else np.nan
    return port_return, port_vol, sharpe


def negative_sharpe(weights, mean_returns, cov_matrix, rf):
    return -portfolio_performance(weights, mean_returns, cov_matrix, rf)[2]


def portfolio_volatility(weights, mean_returns, cov_matrix, rf):
    return portfolio_performance(weights, mean_returns, cov_matrix, rf)[1]


def optimize_portfolio(mean_returns, cov_matrix, rf, objective='Max Sharpe', target_return=None, allow_short=False):
    n = len(mean_returns)
    bounds = tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    x0 = np.array([1.0 / n] * n)

    if objective == 'Max Sharpe':
        result = minimize(negative_sharpe, x0, args=(mean_returns, cov_matrix, rf), method='SLSQP', bounds=bounds, constraints=constraints)
    elif objective == 'Min Volatility':
        result = minimize(portfolio_volatility, x0, args=(mean_returns, cov_matrix, rf), method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        constraints.append({'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return})
        result = minimize(portfolio_volatility, x0, args=(mean_returns, cov_matrix, rf), method='SLSQP', bounds=bounds, constraints=constraints)

    return result


def simulate_frontier(mean_returns, cov_matrix, rf, n_portfolios=5000, allow_short=False, seed=42):
    rng = np.random.default_rng(seed)
    n = len(mean_returns)
    rows = []
    for _ in range(n_portfolios):
        if allow_short:
            w = rng.normal(size=n)
            if np.isclose(np.sum(np.abs(w)), 0):
                continue
            w = w / np.sum(np.abs(w))
            w = w / np.sum(w)
        else:
            w = rng.random(n)
            w = w / np.sum(w)
        pr, pv, ps = portfolio_performance(w, mean_returns, cov_matrix, rf)
        rows.append([pr, pv, ps, *w])
    cols = ['Return', 'Volatility', 'Sharpe'] + list(mean_returns.index)
    return pd.DataFrame(rows, columns=cols)


def portfolio_series(returns: pd.DataFrame, weights):
    return returns.mul(weights, axis=1).sum(axis=1)


def downside_metrics(portfolio_returns: pd.Series, confidence_level: float, mar_annual: float):
    mar_daily = mar_annual / TRADING_DAYS
    var_daily = np.quantile(portfolio_returns, 1 - confidence_level)
    cvar_daily = portfolio_returns[portfolio_returns <= var_daily].mean()
    downside = np.minimum(portfolio_returns - mar_daily, 0)
    lpsd_daily = np.sqrt(np.mean(downside ** 2))
    wealth_index = (1 + portfolio_returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = wealth_index / previous_peaks - 1
    return {
        'Daily VaR': var_daily,
        'Daily CVaR': cvar_daily,
        'Daily LPSD': lpsd_daily,
        'Max Drawdown': drawdowns.min(),
    }


def calculate_beta_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series, rf_annual: float):
    merged = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if merged.shape[0] < 30:
        return np.nan, np.nan, np.nan
    merged.columns = ['portfolio', 'benchmark']
    rf_daily = rf_annual / TRADING_DAYS
    ex_p = merged['portfolio'] - rf_daily
    ex_m = merged['benchmark'] - rf_daily
    beta = np.cov(ex_p, ex_m)[0, 1] / np.var(ex_m) if np.var(ex_m) > 0 else np.nan
    alpha_daily = ex_p.mean() - beta * ex_m.mean()
    alpha_annual = alpha_daily * TRADING_DAYS
    corr = merged['portfolio'].corr(merged['benchmark'])
    return beta, alpha_annual, corr


def format_pct(x):
    return f"{x:.2%}" if pd.notna(x) else 'N/A'


st.title('Portfolio Optimization & Risk Lab')
st.markdown(
    'Build a decision-support portfolio using **mean-variance optimization** and evaluate it with '
    '**VaR, CVaR, lower partial standard deviation, drawdown, and benchmark beta/alpha**.'
)

with st.sidebar:
    st.header('Inputs')
    tickers_raw = st.text_area('Asset tickers (comma-separated)', 'AAPL, MSFT, NVDA, JPM, XOM')
    benchmark = st.text_input('Benchmark ticker', 'SPY')
    start_date = st.date_input('Start date', pd.to_datetime('2021-01-01'))
    end_date = st.date_input('End date', pd.to_datetime('today'))
    rf_rate = st.number_input('Annual risk-free rate', min_value=0.0, max_value=0.20, value=0.04, step=0.005, format='%.3f')
    mar = st.number_input('Minimum acceptable annual return (for LPSD)', min_value=0.0, max_value=0.20, value=0.02, step=0.005, format='%.3f')
    confidence = st.slider('Confidence level for VaR/CVaR', min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    objective = st.selectbox('Optimization objective', ['Max Sharpe', 'Min Volatility', 'Target Return'])
    allow_short = st.checkbox('Allow short selling', value=False)
    n_sim = st.slider('Frontier simulation portfolios', 1000, 10000, 4000, 500)

    tickers = clean_tickers(tickers_raw)
    target_return = None

    if objective == 'Target Return':
        target_return = st.number_input('Target annual portfolio return', min_value=-0.20, max_value=0.60, value=0.15, step=0.01, format='%.2f')

if len(tickers) < 2:
    st.error('Enter at least two tickers.')
    st.stop()

prices = load_prices(tickers, start_date, end_date)
if prices.empty or prices.shape[1] < 2:
    st.error('Not enough price data to run the model. Try different tickers or dates.')
    st.stop()

returns = prices.pct_change().dropna()
valid_cols = returns.columns[returns.notna().sum() > 30]
returns = returns[valid_cols]
prices = prices[valid_cols]

if returns.shape[1] < 2:
    st.error('After cleaning missing data, fewer than two assets remain.')
    st.stop()

mean_returns, cov_matrix = annualized_stats(returns)
opt = optimize_portfolio(mean_returns, cov_matrix, rf_rate, objective=objective, target_return=target_return, allow_short=allow_short)

if not opt.success:
    st.error(f'Optimization failed: {opt.message}')
    st.stop()

weights = pd.Series(opt.x, index=mean_returns.index, name='Weight')
port_ret, port_vol, port_sharpe = portfolio_performance(weights.values, mean_returns.values, cov_matrix.values, rf_rate)
port_daily = portfolio_series(returns, weights)
downside = downside_metrics(port_daily, confidence, mar)
benchmark_px = load_benchmark(benchmark, start_date, end_date)
benchmark_ret = benchmark_px.pct_change().dropna() if not benchmark_px.empty else pd.Series(dtype=float)
beta, alpha, corr = calculate_beta_alpha(port_daily, benchmark_ret, rf_rate) if not benchmark_ret.empty else (np.nan, np.nan, np.nan)

cum_assets = (1 + returns).cumprod()
cum_port = (1 + port_daily).cumprod().rename('Optimized Portfolio')
if not benchmark_ret.empty:
    cum_bench = (1 + benchmark_ret).cumprod().rename(benchmark)
else:
    cum_bench = pd.Series(dtype=float)

frontier = simulate_frontier(mean_returns, cov_matrix, rf_rate, n_portfolios=n_sim, allow_short=allow_short)

col1, col2, col3, col4 = st.columns(4)
col1.metric('Expected Annual Return', format_pct(port_ret))
col2.metric('Annual Volatility', format_pct(port_vol))
col3.metric('Sharpe Ratio', f'{port_sharpe:.2f}')
col4.metric('Max Drawdown', format_pct(downside['Max Drawdown']))

col5, col6, col7, col8 = st.columns(4)
col5.metric(f'Daily VaR ({int(confidence*100)}%)', format_pct(downside['Daily VaR']))
col6.metric(f'Daily CVaR ({int(confidence*100)}%)', format_pct(downside['Daily CVaR']))
col7.metric('Daily LPSD', format_pct(downside['Daily LPSD']))
col8.metric('Beta vs Benchmark', f'{beta:.2f}' if pd.notna(beta) else 'N/A')

left, right = st.columns([1, 1])

with left:
    st.subheader('Optimal Weights')
    weights_df = weights.reset_index()
    weights_df.columns = ['Ticker', 'Weight']
    st.dataframe(weights_df.style.format({'Weight': '{:.2%}'}), use_container_width=True)
    fig_w = px.bar(weights_df, x='Ticker', y='Weight', title='Portfolio Allocation')
    st.plotly_chart(fig_w, use_container_width=True)

with right:
    st.subheader('Efficient Frontier (Simulated)')
    fig_f = px.scatter(
        frontier,
        x='Volatility',
        y='Return',
        color='Sharpe',
        title='Risk-Return Opportunity Set',
        hover_data=list(mean_returns.index),
    )
    fig_f.add_trace(go.Scatter(x=[port_vol], y=[port_ret], mode='markers', name='Optimal Portfolio', marker=dict(size=14, symbol='star')))
    st.plotly_chart(fig_f, use_container_width=True)

st.subheader('Cumulative Performance')
perf_df = pd.concat([cum_port, cum_bench], axis=1)
fig_perf = px.line(perf_df, x=perf_df.index, y=perf_df.columns, title='Optimized Portfolio vs Benchmark')
st.plotly_chart(fig_perf, use_container_width=True)

st.subheader('Asset Price History')
fig_prices = px.line(prices, x=prices.index, y=prices.columns, title='Adjusted Price Series')
st.plotly_chart(fig_prices, use_container_width=True)

st.subheader('Asset Summary Statistics')
summary = pd.DataFrame({
    'Annual Mean Return': mean_returns,
    'Annual Volatility': np.sqrt(np.diag(cov_matrix)),
    'Weight': weights,
})
summary['Contribution to Expected Return'] = summary['Annual Mean Return'] * summary['Weight']
st.dataframe(summary.style.format('{:.2%}'), use_container_width=True)

st.subheader('Economic Interpretation')
interpretation = pd.DataFrame({
    'Metric': [
        'Expected Annual Return', 'Annual Volatility', 'Sharpe Ratio',
        f'Daily VaR ({int(confidence*100)}%)', f'Daily CVaR ({int(confidence*100)}%)',
        'Daily LPSD', 'Max Drawdown', 'Beta', 'Annual Alpha'
    ],
    'Interpretation': [
        'Model-implied average annual payoff based on historical sample means.',
        'Total risk from the covariance matrix; higher means wider fluctuations in wealth.',
        'Risk-adjusted excess return per unit of volatility.',
        'Threshold daily loss exceeded only in the worst tail outcomes at the chosen confidence level.',
        'Average daily loss in the tail once VaR is breached; captures tail severity better than VaR alone.',
        'Downside risk relative to the minimum acceptable return, not total volatility.',
        'Largest peak-to-trough loss experienced in the sample period.',
        f'Sensitivity of the optimized portfolio to {benchmark} market movements.',
        'Excess annualized performance not explained by benchmark exposure.'
    ],
    'Value': [
        format_pct(port_ret), format_pct(port_vol), f'{port_sharpe:.2f}', format_pct(downside['Daily VaR']),
        format_pct(downside['Daily CVaR']), format_pct(downside['Daily LPSD']), format_pct(downside['Max Drawdown']),
        f'{beta:.2f}' if pd.notna(beta) else 'N/A', format_pct(alpha)
    ]
})
st.dataframe(interpretation, use_container_width=True)

st.caption(
    'Educational use only. Outputs depend on historical data, chosen sample window, and the assumption that '
    'past return and covariance patterns are informative about the future.'
)
