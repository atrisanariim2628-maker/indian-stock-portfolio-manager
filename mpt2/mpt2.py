import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# --- Page Configuration ---
st.set_page_config(page_title="Indian Stock Portfolio Optimizer", page_icon="📈", layout="wide")

st.title("🇮🇳 Indian Stock Portfolio Optimizer (MPT)")
st.write("Optimize your NSE stock portfolio using Modern Portfolio Theory to maximize your risk-adjusted returns (Sharpe Ratio).")

# --- Sidebar Inputs ---
st.sidebar.header("Portfolio Settings")

# Default Indian tickers (with .NS suffix)
default_tickers = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ITC.NS, TATAMOTORS.NS"
tickers_input = st.sidebar.text_area("Enter NSE Tickers (comma-separated):", default_tickers)

# Investment capital in INR
investment_amount = st.sidebar.number_input("Total Investment Amount (₹)", min_value=5000, value=100000, step=5000)

# Date range
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2026-01-01"))

# Run Button
run_button = st.sidebar.button("Run Optimization")

# --- Main App Logic ---
if run_button:
    # Clean ticker inputs
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if len(tickers) < 2:
        st.error("Please enter at least 2 valid stock tickers for portfolio optimization.")
    else:
        with st.spinner("Fetching historical data from Yahoo Finance and optimizing..."):
            try:
                # Fetch historical adjusted close prices
                data = yf.download(tickers, start=start_date, end=end_date)['Close']
                
                # Drop any columns with missing data
                data = data.dropna(axis=1, how='any')
                
                if data.empty or len(data.columns) < 2:
                    st.error("Could not fetch enough valid data for the selected tickers. Please check the ticker symbols.")
                else:
                    # Calculate expected returns and covariance matrix
                    mu = expected_returns.mean_historical_return(data)
                    S = risk_models.sample_cov(data)
                    
                    # Optimize for Maximum Sharpe Ratio
                    ef = EfficientFrontier(mu, S)
                    raw_weights = ef.max_sharpe()
                    cleaned_weights = ef.clean_weights()
                    
                    # Display Results in Layout Columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Optimal Portfolio Weights")
                        weights_df = pd.DataFrame(list(cleaned_weights.items()), columns=['Stock', 'Weight'])
                        weights_df['Weight (%)'] = (weights_df['Weight'] * 100).round(2)
                        # Filter out zero weights
                        weights_df = weights_df[weights_df['Weight'] > 0]
                        st.dataframe(weights_df[['Stock', 'Weight (%)']], hide_index=True)
                        
                        # Pie chart visualization
                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.pie(weights_df['Weight'], labels=weights_df['Stock'], autopct='%1.1f%%', startangle=90)
                        ax.axis('equal')
                        st.pyplot(fig)

                    with col2:
                        st.subheader("📈 Projected Performance")
                        perf = ef.portfolio_performance(verbose=False)
                        
                        st.metric(label="Expected Annual Return", value=f"{perf[0]*100:.2f}%")
                        st.metric(label="Expected Annual Volatility (Risk)", value=f"{perf[1]*100:.2f}%")
                        st.metric(label="Sharpe Ratio", value=f"{perf[2]:.2f}")
                        
                        st.subheader("🛒 Recommended Share Allocation")
                        latest_prices = get_latest_prices(data)
                        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=investment_amount)
                        allocation, leftover = da.lp_portfolio()
                        
                        alloc_df = pd.DataFrame(list(allocation.items()), columns=['Stock', 'Shares to Buy'])
                        st.dataframe(alloc_df, hide_index=True)
                        st.info(f"💰 Leftover Uninvested Cash: **₹{leftover:,.2f}**")
                        
            except Exception as e:
                st.error(f"An error occurred during optimization: {e}")
else:
    st.info("👈 Enter your tickers and parameters in the sidebar, then click **Run Optimization** to begin.")