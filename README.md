# Portfolio Analysis

Analysis for a fund and its holdings compared to a benchmark. Includes
correlation, hit rates, beta, capture ratios, and drawdown analysis, presented
through a Streamlit user interface.

Historical price data is retrieved via Yahoo Finance for demonstration
purposes only. The project was originally designed to work with Bloomberg's
BQNT — the placeholder holdings list will be replaced with live Bloomberg
data once that connection is in place.

## What the app does

Pick a fund and a benchmark from dropdowns, choose a date range (at least one
year), hit Run, and the app downloads weekly closes and renders three tabs of
analysis:

- **Correlation** — heatmap of pairwise correlations across the fund's
  holdings, plus a rolling pairwise correlation chart overlaid on the
  benchmark's cumulative return. The window is adjustable.
- **Hit Rates** — scatter of up- vs down-market hit rates with the fund
  highlighted, followed by a metrics table covering hit rate, beta, capture
  ratios, max drawdown, and recuperation ratio for every ticker.
- **Drawdown** — drawdown time series for the selected fund and benchmark.

A persistent disclaimer banner makes it explicit that the holdings list is
placeholder data, not the actual fund constituents.

## Fund and benchmark options

| Funds | Benchmarks |
| --- | --- |
| JEPI — JPMorgan Equity Premium Income ETF | SPY — S&P 500 |
| IGM — iShares Expanded Tech Sector ETF | QQQ — Nasdaq 100 |
| | IWM — Russell 2000 |
| | ACWI — MSCI All Countries World |

The placeholder holdings (PLTR, META, AMZN, GOOG, TSLA, V, ROKU, FICO, MA,
WMT, NFLX) are tech-heavy on purpose — they let JEPI showcase its defensive
character against a more volatile basket.

## Project structure

```
PortfolioAnalysis/
├── app.py                    # Streamlit UI (entry point)
├── analysis.py               # Pure analysis functions (data load, metrics, plots)
├── requirements.txt          # Python dependencies
├── Corr_drawd_Hitrates.ipynb # Original notebook the project was refactored from
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

The `requirements.txt` file pins the libraries the app depends on
(`streamlit`, `yfinance`, `curl_cffi`, `pandas`, `numpy`, `seaborn`,
`matplotlib`). `curl_cffi` is required because Yahoo Finance blocks the
default yfinance HTTP client; the app routes requests through a Chrome
impersonation session to bypass that.

## Running the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) — open
that in a browser.

## Roadmap

- Replace placeholder holdings with live constituents from Bloomberg's BQNT.
- Asset-class return time series.
- Optional retry/backoff for rate-limited Yahoo Finance requests.

## Notes

- Returns are computed on weekly close data (Friday).
- The minimum date range is one year, enforced after Run is pressed.
- Yahoo Finance occasionally throttles requests; if a Run fails, wait a
  minute and try again, or upgrade `yfinance` and `curl_cffi`.
