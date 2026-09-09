"""Portfolio analysis: correlations, hit rates, betas, capture ratios, drawdowns.

Pure functions — every plot returns a matplotlib Figure (no plt.show()) so the
caller (Streamlit, notebook, script) decides how to render it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

# Yahoo Finance has been blocking the default yfinance client. A curl_cffi
# session that impersonates a real browser routes around it. We import lazily
# so the module still works without the dependency installed.
try:
    from curl_cffi import requests as _cffi_requests
    _BROWSER_SESSION = _cffi_requests.Session(impersonate='chrome')
except Exception:  # noqa: BLE001 — fall back to default if curl_cffi missing
    _BROWSER_SESSION = None


# Placeholder holdings used until a Bloomberg API connection is wired in.
PLACEHOLDER_HOLDINGS: list[str] = [
    'PLTR', 'META', 'AMZN', 'GOOG', 'TSLA',
    'V', 'ROKU', 'FICO', 'MA', 'WMT', 'NFLX',
]

FUND_OPTIONS: dict[str, str] = {
    'JEPI': 'JPMorgan Equity Premium Income ETF',
    'IGM': 'iShares Expanded Tech Sector ETF',
}

BENCHMARK_OPTIONS: dict[str, str] = {
    'SPY': 'S&P 500',
    'QQQ': 'Nasdaq 100',
    'IWM': 'Russell 2000',
    'ACWI': 'MSCI All Countries World',
}

# Flat annual risk-free rate assumption used by the risk-metrics table below.
# We assume this rather than downloading an actual rate series (e.g. T-bill
# yields) — simpler, and fine for a rough risk-adjusted comparison.
RISK_FREE_RATE: float = 0.04


def load_weekly_returns(
    fund: str,
    benchmark: str,
    start_date: str,
    end_date: str,
    holdings: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download daily closes, resample to weekly, compute pct returns.

    Returns (prices, returns). Both indexed by week-end date with one column
    per ticker. The full ticker universe is holdings + [fund, benchmark].
    """
    if holdings is None:
        holdings = PLACEHOLDER_HOLDINGS

    tickers = list(dict.fromkeys(holdings + [fund, benchmark]))  # de-dup, preserve order

    download_kwargs = dict(
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
        threads=False,  # serial requests are gentler on Yahoo's anti-bot
    )
    if _BROWSER_SESSION is not None:
        download_kwargs['session'] = _BROWSER_SESSION

    raw = yf.download(tickers, **download_kwargs)

    # yfinance returns a multi-indexed frame for >1 ticker, simple frame for 1
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close']
    else:
        prices = raw[['Close']].rename(columns={'Close': tickers[0]})

    # Drop tickers that came back empty (Yahoo silently returns NaN columns
    # for blocked / mistyped / delisted symbols)
    prices = prices.dropna(axis=1, how='all')

    # Per-ticker fallback for anything missing — Ticker.history() uses a
    # different code path and sometimes succeeds when batch download didn't
    missing = [t for t in tickers if t not in prices.columns]
    for ticker in missing:
        try:
            t_obj = (
                yf.Ticker(ticker, session=_BROWSER_SESSION)
                if _BROWSER_SESSION is not None
                else yf.Ticker(ticker)
            )
            hist = t_obj.history(start=start_date, end=end_date, auto_adjust=False)
            if not hist.empty and 'Close' in hist.columns:
                # history() returns a tz-aware index; align to naive for resample
                close = hist['Close'].copy()
                close.index = close.index.tz_localize(None) if close.index.tz is not None else close.index
                prices[ticker] = close
        except Exception:  # noqa: BLE001 — skip ticker, surface in caller as missing
            continue

    prices = prices.resample('W').last().dropna(how='all')
    returns = prices.pct_change().dropna(how='all')
    return prices, returns


def correlation_matrix_fig(returns: pd.DataFrame, holdings: list[str]) -> Figure:
    """Heatmap of pearson correlations across the holdings, with top/bottom
    pairs called out in a side legend."""
    available = [t for t in holdings if t in returns.columns]
    corr = returns.loc[:, available].corr(method='pearson', min_periods=10)

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap=sns.diverging_palette(133, 10, as_cmap=True),
        vmin=-1,
        vmax=1,
        cbar=False,
        ax=ax,
    )
    ax.set_title('Correlation Matrix')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(False)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=True, labelleft=True)

    # Top/bottom 5 pairs (excluding self-correlations)
    pairs = corr.where(~mask).stack().dropna().sort_values(kind='quicksort')
    pairs = pairs[pairs != 1].dropna()
    top_5 = pairs.iloc[-5:][::-1]
    bottom_5 = pairs.iloc[:5]

    top_text = '\n'.join(f'{a} & {b}: {v:.2f}' for (a, b), v in top_5.items())
    bottom_text = '\n'.join(f'{a} & {b}: {v:.2f}' for (a, b), v in bottom_5.items())

    fig.text(0.7, 0.7, f'Highest:\n{top_text}', ha='left', fontsize=11,
             bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5, 'edgecolor': 'none'})
    fig.text(0.7, 0.45, f'Lowest:\n{bottom_text}', ha='left', fontsize=11,
             bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5, 'edgecolor': 'none'})

    fig.tight_layout()
    return fig


def compute_metrics(
    returns: pd.DataFrame,
    fund: str,
    benchmark: str,
    holdings: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Compute hit rate, beta, capture ratios, and drawdowns for every
    ticker (fund + holdings + benchmark) vs the benchmark.

    Returns (results_df, drawdown_series_by_ticker).
    """
    if benchmark not in returns.columns:
        raise ValueError(
            f"Benchmark '{benchmark}' is not in the downloaded data. "
            f'Available tickers: {list(returns.columns)}'
        )
    if fund not in returns.columns:
        raise ValueError(
            f"Fund '{fund}' is not in the downloaded data. "
            f'Available tickers: {list(returns.columns)}'
        )

    tickers = list(dict.fromkeys(holdings + [fund, benchmark]))

    results: dict[str, dict[str, float]] = {
        'Hit Rate': {}, 'Up Market Hit Rate': {}, 'Down Market Hit Rate': {},
        'Beta': {}, 'Up Market Beta': {}, 'Down Market Beta': {},
        'Up Capture Ratio': {}, 'Down Capture Ratio': {},
        'Max Drawdown': {}, 'Recuperation Ratio': {},
    }
    drawdown_series: dict[str, pd.Series] = {}

    bench_returns = returns[benchmark]
    up_market = returns[returns[benchmark] > 0]
    down_market = returns[returns[benchmark] <= 0]

    for stock in tickers:
        if stock not in returns.columns:
            continue

        stock_returns = returns[stock]

        if stock != benchmark:
            results['Hit Rate'][stock] = (stock_returns > bench_returns).mean()
            results['Up Market Hit Rate'][stock] = (
                (up_market[stock] > up_market[benchmark]).mean() if len(up_market) else 0
            )
            results['Down Market Hit Rate'][stock] = (
                (down_market[stock] > down_market[benchmark]).mean() if len(down_market) else 0
            )

            cov = returns[[stock, benchmark]].cov()
            results['Beta'][stock] = cov.loc[stock, benchmark] / bench_returns.var()

            results['Up Market Beta'][stock] = (
                up_market[[stock, benchmark]].cov().loc[stock, benchmark] / up_market[benchmark].var()
                if len(up_market) else 0
            )
            results['Down Market Beta'][stock] = (
                down_market[[stock, benchmark]].cov().loc[stock, benchmark] / down_market[benchmark].var()
                if len(down_market) else 0
            )

            results['Up Capture Ratio'][stock] = (
                up_market[stock].mean() / up_market[benchmark].mean()
                if up_market[benchmark].mean() != 0 else 0
            )
            results['Down Capture Ratio'][stock] = (
                down_market[stock].mean() / down_market[benchmark].mean()
                if down_market[benchmark].mean() != 0 else 0
            )

        cum_returns = (1 + stock_returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns / rolling_max) - 1

        drawdown_series[stock] = drawdown
        results['Max Drawdown'][stock] = drawdown.min()
        results['Recuperation Ratio'][stock] = cum_returns.max() / cum_returns.min()

    results_df = pd.DataFrame(results)
    return results_df, drawdown_series


# ---------------------------------------------------------------------------
# Risk / risk-adjusted return metrics — fund and benchmark
# ---------------------------------------------------------------------------

# Row-name -> display format. 'pct' for returns/vol-type figures, 'ratio' for
# unitless ratios and distribution stats — the two don't share one sensible
# numeric format, so formatting is looked up per metric name.
STANDALONE_METRIC_FORMATS: dict[str, str] = {
    'Annualized Return (CAGR)': 'pct',
    'Annualized Volatility': 'pct',
    'Sharpe Ratio': 'ratio',
    'Sortino Ratio': 'ratio',
    'Calmar Ratio': 'ratio',
    'Max Drawdown': 'pct',
    'Value at Risk (95%, weekly)': 'pct',
    'Skewness': 'ratio',
    'Kurtosis': 'ratio',
}

RELATIVE_METRIC_FORMATS: dict[str, str] = {
    'Beta': 'ratio',
    'Tracking Error': 'pct',
    'Information Ratio': 'ratio',
    "Jensen's Alpha": 'pct',
}

METRIC_DESCRIPTIONS: dict[str, str] = {
    'Annualized Return (CAGR)':
        'Compounded annual growth rate implied by weekly returns over the selected period.',
    'Annualized Volatility':
        'Standard deviation of weekly returns, annualized (x sqrt(52)) — how much returns swing '
        'around their average.',
    'Sharpe Ratio':
        f'Annualized return in excess of the risk-free rate (assumed {RISK_FREE_RATE:.0%}), per '
        'unit of volatility. Higher means better risk-adjusted performance.',
    'Sortino Ratio':
        'Like the Sharpe ratio, but only penalizes downside volatility (negative weeks), '
        'ignoring upside swings.',
    'Calmar Ratio':
        'Annualized return divided by the worst peak-to-trough drawdown — reward relative to '
        'the deepest pain endured.',
    'Max Drawdown':
        'Largest peak-to-trough decline in cumulative value over the period.',
    'Value at Risk (95%, weekly)':
        'The weekly loss that returns are not expected to exceed 95% of the time, based on the '
        'historical distribution of weekly returns. With roughly 1-2 years of weekly data this '
        'is a thin sample for tail estimates — treat it as indicative, not precise.',
    'Skewness':
        'Asymmetry of the weekly return distribution. Negative skew means occasional large '
        'losses; positive skew means occasional large gains.',
    'Kurtosis':
        'Excess tail-fatness of the return distribution vs. a normal distribution — higher '
        'means more extreme weeks than a normal curve would predict.',
    'Beta':
        "Sensitivity of the fund's weekly returns to the benchmark's. Beta above 1 amplifies "
        'benchmark moves, below 1 dampens them.',
    'Tracking Error':
        "Annualized standard deviation of the fund's return minus the benchmark's return — how "
        'much the fund deviates from the benchmark week to week.',
    'Information Ratio':
        'Annualized excess return over the benchmark, divided by tracking error — whether the '
        'fund is being rewarded for the risk it takes relative to the benchmark.',
    "Jensen's Alpha":
        f"The fund's annualized return minus what CAPM would predict from its beta and the "
        f'benchmark\'s return, using a {RISK_FREE_RATE:.0%} risk-free rate. Positive alpha '
        'suggests outperformance beyond what beta explains.',
}


def compute_risk_metrics(
    returns: pd.DataFrame,
    fund: str,
    benchmark: str,
    risk_free_rate: float = RISK_FREE_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Risk and risk-adjusted return metrics for the fund and benchmark.

    Returns (standalone_df, relative_df):
      - standalone_df: one column per ticker (fund, benchmark), each computed
        independently — annualized return, volatility, Sharpe, Sortino,
        Calmar, max drawdown, historical VaR, skewness, kurtosis.
      - relative_df: metrics that only exist as "fund vs. benchmark" (beta,
        tracking error, information ratio, Jensen's alpha) — a single column.

    All ratios assume a flat annual risk-free rate (`risk_free_rate`) rather
    than a downloaded rate series. Annualized return is the compounded
    (geometric) return of weekly returns scaled to a 52-week year; volatility
    figures scale weekly std dev by sqrt(52). Sharpe/Sortino/alpha all use
    this same CAGR figure, so the ratios stay consistent with the annualized
    return shown in the same table.
    """

    def _annualized_return(r: pd.Series) -> float:
        r = r.dropna()
        n = len(r)
        if n == 0:
            return np.nan
        return (1 + r).prod() ** (52 / n) - 1

    def _annualized_vol(r: pd.Series) -> float:
        r = r.dropna()
        return r.std(ddof=1) * np.sqrt(52) if len(r) > 1 else np.nan

    def _downside_vol(r: pd.Series) -> float:
        downside = r.dropna()
        downside = downside[downside < 0]
        return downside.std(ddof=1) * np.sqrt(52) if len(downside) > 1 else np.nan

    def _max_drawdown(r: pd.Series) -> float:
        r = r.dropna()
        if r.empty:
            return np.nan
        cum = (1 + r).cumprod()
        return (cum / cum.cummax() - 1).min()

    standalone: dict[str, dict[str, float]] = {}
    for ticker in (fund, benchmark):
        if ticker not in returns.columns:
            continue
        r = returns[ticker]
        ann_return = _annualized_return(r)
        ann_vol = _annualized_vol(r)
        downside_vol = _downside_vol(r)
        max_dd = _max_drawdown(r)
        standalone[ticker] = {
            'Annualized Return (CAGR)': ann_return,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio': (ann_return - risk_free_rate) / ann_vol if ann_vol else np.nan,
            'Sortino Ratio': (
                (ann_return - risk_free_rate) / downside_vol if downside_vol else np.nan
            ),
            'Calmar Ratio': ann_return / abs(max_dd) if max_dd else np.nan,
            'Max Drawdown': max_dd,
            'Value at Risk (95%, weekly)': r.quantile(0.05),
            'Skewness': r.skew(),
            'Kurtosis': r.kurt(),
        }
    standalone_df = pd.DataFrame(standalone)

    relative_df = pd.DataFrame()
    if fund in returns.columns and benchmark in returns.columns:
        fund_r = returns[fund]
        bench_r = returns[benchmark]

        cov = returns[[fund, benchmark]].cov()
        beta = cov.loc[fund, benchmark] / bench_r.var()
        tracking_error = (fund_r - bench_r).std(ddof=1) * np.sqrt(52)

        fund_ann = _annualized_return(fund_r)
        bench_ann = _annualized_return(bench_r)
        information_ratio = (
            (fund_ann - bench_ann) / tracking_error if tracking_error else np.nan
        )
        alpha = fund_ann - (risk_free_rate + beta * (bench_ann - risk_free_rate))

        relative_df = pd.DataFrame({
            f'{fund} vs {benchmark}': {
                'Beta': beta,
                'Tracking Error': tracking_error,
                'Information Ratio': information_ratio,
                "Jensen's Alpha": alpha,
            }
        })

    return standalone_df, relative_df


def format_risk_metrics_for_display(df: pd.DataFrame, formats: dict[str, str]) -> pd.DataFrame:
    """Render a raw risk-metrics DataFrame (from compute_risk_metrics) as
    display-ready strings — percentages for return/vol-type rows, 2dp
    ratios otherwise. Returns/vol and ratio metrics don't share one sensible
    numeric format, so this is done per-row rather than per-column."""
    display = df.copy().astype(object)
    for metric, kind in formats.items():
        if metric not in display.index:
            continue
        fmt = (lambda v: f'{v:.2%}') if kind == 'pct' else (lambda v: f'{v:.2f}')
        display.loc[metric] = df.loc[metric].map(lambda v: fmt(v) if pd.notna(v) else '—')
    return display


def hit_rate_scatter_fig(results_df: pd.DataFrame, fund: str) -> Figure:
    """Scatter of up- vs down-market hit rates, with the fund highlighted."""
    fig, ax = plt.subplots(figsize=(10, 8))

    df = results_df.dropna(subset=['Up Market Hit Rate', 'Down Market Hit Rate'])
    colors = ['red' if t == fund else 'orange' for t in df.index]

    ax.scatter(df['Up Market Hit Rate'], df['Down Market Hit Rate'], s=50, color=colors)
    for ticker, row in df.iterrows():
        ax.annotate(ticker, (row['Up Market Hit Rate'], row['Down Market Hit Rate']),
                    xytext=(5, 5), textcoords='offset points')

    ax.set_title('Up Market vs Down Market Hit Rates', fontsize=16)
    ax.set_xlabel('Up Market Hit Rate %', fontsize=12)
    ax.set_ylabel('Down Market Hit Rate %', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.05, 0.95, 'Low beta', ha='left', va='center', color='orange', fontsize=13)
    ax.text(0.95, 0.95, 'All-Weather Winners', ha='right', va='center', color='green', fontsize=13)
    ax.text(0.05, 0.05, 'Worst', ha='left', va='center', color='red', fontsize=13)
    ax.text(0.95, 0.05, 'High beta', ha='right', va='center', color='orange', fontsize=13)

    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    return fig


def rolling_pairwise_corr_fig(
    returns: pd.DataFrame,
    holdings: list[str],
    benchmark: str,
    prices: pd.DataFrame,
    window: int = 26,
) -> Figure:
    """Mean upper-triangle pairwise correlation of the holdings over a rolling
    window, plotted alongside the benchmark's price.

    On weekly data, 26 weeks (~6 months) is a sensible default. 13 weeks is
    noisy, 52 weeks smooths out short-term spikes. The benchmark price is
    plotted on a secondary axis so the user can see how holding co-movement
    evolves through up- and down-cycles.
    """
    available = [h for h in holdings if h in returns.columns]
    if len(available) < 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Need at least 2 holdings to compute pairwise correlation.',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    sub = returns[available]
    if len(sub) <= window:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5,
                f'Not enough data: need > {window} weekly observations, got {len(sub)}.\n'
                f'Either widen the date range or shrink the window.',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Rolling pairwise correlation matrix per date, then per-date upper-triangle mean
    rolling = sub.rolling(window).corr()
    n = len(available)
    iu = np.triu_indices(n, k=1)
    dates = sub.index[window - 1:]

    mean_corr_values = []
    for d in dates:
        block = rolling.loc[d].values
        mean_corr_values.append(np.nanmean(block[iu]))
    mean_corr = pd.Series(mean_corr_values, index=dates,
                          name=f'Avg pairwise corr ({window}w)').dropna()

    fig, ax_corr = plt.subplots(figsize=(11, 6))
    ax_corr.plot(mean_corr.index, mean_corr.values, color='steelblue', linewidth=1.8,
                 label=f'Avg pairwise corr ({window}w)')
    ax_corr.set_ylabel('Average Pairwise Correlation', color='steelblue')
    ax_corr.tick_params(axis='y', labelcolor='steelblue')
    ax_corr.set_ylim(-0.2, 1.0)
    ax_corr.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax_corr.grid(True, axis='y', linestyle=':', alpha=0.4)

    # Benchmark price on secondary axis, aligned to the same window
    if benchmark in prices.columns:
        bench_price = prices[benchmark].reindex(mean_corr.index).dropna()
        ax_bench = ax_corr.twinx()
        ax_bench.plot(bench_price.index, bench_price.values, color='gray', alpha=0.7,
                      linewidth=1.4, label=f'{benchmark} price')
        ax_bench.set_ylabel(f'{benchmark} Price', color='gray')
        ax_bench.tick_params(axis='y', labelcolor='gray')
        for spine in ('top', 'right'):
            ax_bench.spines[spine].set_visible(False)

        # Combined legend
        lines1, labels1 = ax_corr.get_legend_handles_labels()
        lines2, labels2 = ax_bench.get_legend_handles_labels()
        ax_corr.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False)
    else:
        ax_corr.legend(loc='upper left', frameon=False)

    ax_corr.set_title(f'Rolling Pairwise Correlation of Holdings vs. {benchmark}')
    for spine in ('top', 'right'):
        ax_corr.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def drawdown_fig(drawdown_series: dict[str, pd.Series], fund: str, benchmark: str) -> Figure:
    """Drawdown time series for fund + benchmark."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for ticker in (fund, benchmark):
        if ticker in drawdown_series:
            ax.plot(drawdown_series[ticker].index, drawdown_series[ticker], label=ticker)

    ax.set_title(f'{fund}, {benchmark} - Drawdown')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(loc='best', frameon=False)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{100 * x:.0f}%'))
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
