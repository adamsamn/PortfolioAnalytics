"""Streamlit UI for the portfolio analysis toolkit.

Run with: streamlit run app.py
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from analysis import (
    BENCHMARK_OPTIONS,
    FUND_OPTIONS,
    METRIC_DESCRIPTIONS,
    PLACEHOLDER_HOLDINGS,
    RELATIVE_METRIC_FORMATS,
    RISK_FREE_RATE,
    STANDALONE_METRIC_FORMATS,
    compute_metrics,
    compute_risk_metrics,
    correlation_matrix_fig,
    drawdown_fig,
    format_risk_metrics_for_display,
    hit_rate_scatter_fig,
    load_weekly_returns,
    rolling_pairwise_corr_fig,
)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title='Portfolio Analysis',
    page_icon=':bar_chart:',
    layout='wide',
)

st.title('Portfolio Analysis')
st.caption('Correlation, hit rates, beta, capture ratios, and drawdowns vs. a benchmark.')


# ---------------------------------------------------------------------------
# Cached data loader — keyed on inputs so identical Runs are instant
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_load(fund: str, benchmark: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_weekly_returns(fund, benchmark, start_date, end_date)


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header('Inputs')

    with st.form('inputs', clear_on_submit=False):
        fund_label = st.selectbox(
            'Fund',
            options=list(FUND_OPTIONS.keys()),
            format_func=lambda t: f'{t} — {FUND_OPTIONS[t]}',
        )

        benchmark = st.selectbox(
            'Benchmark',
            options=list(BENCHMARK_OPTIONS.keys()),
            format_func=lambda t: f'{t} — {BENCHMARK_OPTIONS[t]}',
        )

        today = date.today()
        default_start = today - timedelta(days=2 * 365)
        default_end = today

        start_date = st.date_input(
            'Start date',
            value=default_start,
            max_value=today,
        )
        end_date = st.date_input(
            'End date',
            value=default_end,
            max_value=today,
        )

        st.caption('Start and end must be at least 1 year apart.')

        run = st.form_submit_button('Run', type='primary', use_container_width=True)


# ---------------------------------------------------------------------------
# Disclaimer banner — always visible
# ---------------------------------------------------------------------------

st.info(
    'Holdings shown below are placeholder data. Bloomberg API integration is pending — '
    'actual fund constituents will replace these once connected.'
)

with st.expander('Placeholder holdings used in this analysis', expanded=True):
    st.write(', '.join(PLACEHOLDER_HOLDINGS))


# ---------------------------------------------------------------------------
# Run handler — fetch + compute, then stash in session_state so widgets
# inside tabs (like the rolling-window slider) can re-render without
# triggering another download.
# ---------------------------------------------------------------------------

if run:
    if end_date <= start_date:
        st.error('End date must be after start date.')
        st.stop()

    if (end_date - start_date).days < 365:
        st.error('Start and end dates must be at least one year apart.')
        st.stop()

    with st.spinner(f'Fetching prices for {fund_label} and {benchmark}…'):
        try:
            prices, returns = _cached_load(
                fund_label,
                benchmark,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
            )
        except Exception as exc:  # noqa: BLE001 — surface any fetch error to the UI
            st.error(f'Failed to download price data: {exc}')
            st.stop()

    if returns.empty:
        st.error('No return data available for the selected range. Try a wider date window.')
        st.stop()

    missing = [t for t in PLACEHOLDER_HOLDINGS + [fund_label, benchmark] if t not in returns.columns]
    if missing:
        st.warning(f'No data returned for: {", ".join(missing)}. They will be skipped.')

    critical_missing = [t for t in (fund_label, benchmark) if t not in returns.columns]
    if critical_missing:
        which = (
            'fund and benchmark' if len(critical_missing) == 2
            else ('fund' if fund_label in critical_missing else 'benchmark')
        )
        st.error(
            f'Could not download data for the selected {which} '
            f'({", ".join(critical_missing)}). This is usually Yahoo Finance blocking '
            'the request.\n\nTry: `pip install --upgrade yfinance curl_cffi` and restart '
            'the app. If it still fails, wait a minute (Yahoo may be rate-limiting) and '
            'hit Run again.'
        )
        st.stop()

    results_df, drawdown_series = compute_metrics(
        returns, fund_label, benchmark, PLACEHOLDER_HOLDINGS
    )

    st.session_state['analysis'] = {
        'fund': fund_label,
        'benchmark': benchmark,
        'prices': prices,
        'returns': returns,
        'results_df': results_df,
        'drawdown_series': drawdown_series,
    }


# ---------------------------------------------------------------------------
# Render — read from session_state so tab widgets don't re-trigger fetches
# ---------------------------------------------------------------------------

if 'analysis' not in st.session_state:
    st.markdown('Pick a fund, benchmark, and date range in the sidebar, then hit **Run**.')
    st.stop()

data = st.session_state['analysis']
fund_label = data['fund']
benchmark = data['benchmark']
prices = data['prices']
returns = data['returns']
results_df = data['results_df']
drawdown_series = data['drawdown_series']

st.subheader(f'{fund_label} vs. {benchmark}')
st.caption(
    f'Weekly returns from {returns.index.min().date()} to {returns.index.max().date()} '
    f'({len(returns)} observations).'
)


# ---------------------------------------------------------------------------
# Risk metrics — fund and benchmark, standalone + relative-to-benchmark
# ---------------------------------------------------------------------------

st.markdown('### Risk Metrics')
st.caption(f'Assumes a flat risk-free rate of {RISK_FREE_RATE:.1%} (not fetched live).')

standalone_metrics, relative_metrics = compute_risk_metrics(returns, fund_label, benchmark)

col_standalone, col_relative = st.columns([2, 1])
with col_standalone:
    st.markdown('**Standalone Metrics**')
    st.dataframe(
        format_risk_metrics_for_display(standalone_metrics, STANDALONE_METRIC_FORMATS),
        use_container_width=True,
    )
with col_relative:
    st.markdown('**Relative to Benchmark**')
    st.dataframe(
        format_risk_metrics_for_display(relative_metrics, RELATIVE_METRIC_FORMATS),
        use_container_width=True,
    )

with st.expander('Metric definitions'):
    for metric, description in METRIC_DESCRIPTIONS.items():
        st.markdown(f'**{metric}** — {description}')


tab_corr, tab_hit, tab_dd = st.tabs(['Correlation', 'Hit Rates', 'Drawdown'])

with tab_corr:
    st.markdown('### Correlation Matrix — Holdings')
    st.pyplot(
        correlation_matrix_fig(returns, PLACEHOLDER_HOLDINGS),
        use_container_width=True,
    )

    st.markdown('### Rolling Pairwise Correlation')
    # Cap window at what's actually feasible given the loaded data
    max_window = max(13, min(78, len(returns) - 1))
    default_window = min(26, max_window)
    window = default_window
    st.pyplot(
        rolling_pairwise_corr_fig(returns, PLACEHOLDER_HOLDINGS, benchmark, prices, window),
        use_container_width=True,
    )

with tab_hit:
    st.markdown('### Up vs. Down Market Hit Rates')
    st.pyplot(
        hit_rate_scatter_fig(results_df, fund_label),
        use_container_width=True,
    )

    st.markdown('### Metrics Table')
    percent_cols = ['Hit Rate', 'Up Market Hit Rate', 'Down Market Hit Rate', 'Max Drawdown']
    ratio_cols = [
        'Beta', 'Up Market Beta', 'Down Market Beta',
        'Up Capture Ratio', 'Down Capture Ratio', 'Recuperation Ratio',
    ]
    styled = (
        results_df.style
        .format({c: '{:.1%}' for c in percent_cols if c in results_df.columns})
        .format({c: '{:.2f}' for c in ratio_cols if c in results_df.columns})
    )
    st.dataframe(styled, use_container_width=True)

with tab_dd:
    st.markdown('### Drawdown — Fund vs. Benchmark')
    st.pyplot(
        drawdown_fig(drawdown_series, fund_label, benchmark),
        use_container_width=True,
    )
