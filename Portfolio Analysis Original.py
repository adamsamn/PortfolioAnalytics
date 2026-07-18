import streamlit as st
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from datetime import date, timedelta

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Fund Analytics Dashboard", layout="wide")

# Placeholder holdings
selected_fund_holdings = ['PLTR', 'META', 'AMZN', 'GOOG', 'TSLA', 'V', 'ROKU', 'FICO', 'MA', 'WMT', 'NFLX']

# Options for dropdowns
fund_options = {
    "JPMorgan Equity Premium Income (JEPI)": "JEPI",
    "iShares Expanded Tech Sector ETF (IGM)": "IGM"
}

benchmark_options = ['SPY', 'QQQ', 'IWM', 'VTI', 'ACWI']

# --- 2. SIDEBAR UI ---
st.sidebar.header("Analysis Parameters")

selected_fund_name = st.sidebar.selectbox("Select Fund", options=list(fund_options.keys()))
selected_fund_ticker = fund_options[selected_fund_name]

selected_benchmark = st.sidebar.selectbox("Select Benchmark", options=benchmark_options)

# Date Selection
default_start = date.today() - timedelta(days=2*365)
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=date.today())

run_analysis = st.sidebar.button("Run Analysis")

# --- 3. DATA FETCHING (CACHED) ---
@st.cache_data
def get_data(tickers, start, end):
    # Set threads=False to prevent the signal/thread error
    df = yf.download(tickers, start=start, end=end, threads=False)['Close']
    
    # Safety check: if only one ticker is downloaded, yfinance returns a Series. 
    # We convert it back to a DataFrame for consistency.
    if isinstance(df, pd.Series):
        df = df.to_frame()
        
    return df.resample('W').last().pct_change().dropna()

# --- 4. MAIN LOGIC ---
if run_analysis:
    # Validate date range (at least 365 days)
    if (end_date - start_date).days < 365:
        st.error("Error: Please select a date range at least one year apart for valid statistical analysis.")
    else:
        with st.spinner("Analyzing market data..."):
            # Prepare Tickers
            all_tickers = selected_fund_holdings + [selected_fund_ticker] + [selected_benchmark]
            df_returns = get_data(all_tickers, start_date, end_date)
            
            # --- CALCULATIONS ---
            results = {
                'Hit Rate': {}, 'Up Market Hit Rate': {}, 'Down Market Hit Rate': {},
                'Beta': {}, 'Up Market Beta': {}, 'Down Market Beta': {},
                'Up Capture Ratio': {}, 'Down Capture Ratio': {},
                'Max Drawdown': {}, 'Recuperation Ratio': {}
            }
            drawdown_series = {}

            for stock in all_tickers:
                stock_returns = df_returns[stock]
                bench_returns = df_returns[selected_benchmark]
                
                if stock != selected_benchmark:
                    results['Hit Rate'][stock] = (stock_returns > bench_returns).mean()
                    up_mkt = df_returns[df_returns[selected_benchmark] > 0]
                    dn_mkt = df_returns[df_returns[selected_benchmark] <= 0]
                    
                    results['Up Market Hit Rate'][stock] = (up_mkt[stock] > up_mkt[selected_benchmark]).mean() if not up_mkt.empty else 0
                    results['Down Market Hit Rate'][stock] = (dn_mkt[stock] > dn_mkt[selected_benchmark]).mean() if not dn_mkt.empty else 0
                    
                    cov = df_returns[[stock, selected_benchmark]].cov().loc[stock, selected_benchmark]
                    results['Beta'][stock] = cov / df_returns[selected_benchmark].var()
                    
                    # Capture Ratios
                    results['Up Capture Ratio'][stock] = (up_mkt[stock].mean() / up_mkt[selected_benchmark].mean()) if not up_mkt.empty else 0
                    results['Down Capture Ratio'][stock] = (dn_mkt[stock].mean() / dn_mkt[selected_benchmark].mean()) if not dn_mkt.empty else 0

                # Drawdowns
                cum_rets = (1 + stock_returns).cumprod()
                drawdown = (cum_rets / cum_rets.cummax()) - 1
                drawdown_series[stock] = drawdown
                results['Max Drawdown'][stock] = drawdown.min()
                results['Recuperation Ratio'][stock] = cum_rets.max() / cum_rets.min()

            results_df = pd.DataFrame(results)

            # --- TABS FOR VISUALIZATION ---
            tab1, tab2, tab3 = st.tabs(["Correlation Matrix", "Performance Analysis", "Drawdowns"])

            with tab1:
                st.subheader("Asset Correlations")
                corr_matrix = df_returns.corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap=sns.diverging_palette(133, 10, as_cmap=True), ax=ax_corr)
                st.pyplot(fig_corr)

            with tab2:
                st.subheader("Up vs Down Market Hit Rates")
                fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
                colors = ['red' if s == selected_fund_ticker else 'orange' for s in all_tickers]
                ax_scatter.scatter(results_df['Up Market Hit Rate'], results_df['Down Market Hit Rate'], color=colors)
                
                for i, txt in enumerate(all_tickers):
                    ax_scatter.annotate(txt, (results_df['Up Market Hit Rate'].iloc[i], results_df['Down Market Hit Rate'].iloc[i]))
                
                ax_scatter.set_xlim(0, 1); ax_scatter.set_ylim(0, 1)
                ax_scatter.axhline(0.5, color='grey', ls='--'); ax_scatter.axvline(0.5, color='grey', ls='--')
                st.pyplot(fig_scatter)
                
                st.write("### Detailed Metrics")
                st.dataframe(results_df.style.format("{:.2f}"))

            with tab3:
                st.subheader("Historical Drawdown")
                fig_dd, ax_dd = plt.subplots(figsize=(10, 5))
                for t in [selected_fund_ticker, selected_benchmark]:
                    ax_dd.plot(drawdown_series[t], label=t)
                
                ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*100:.0f}%'))
                ax_dd.legend()
                st.pyplot(fig_dd)
else:
    st.info("Select your parameters in the sidebar and click 'Run Analysis' to begin.")