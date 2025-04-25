import streamlit as st
import pandas as pd
import datetime

# Page config
st.set_page_config(page_title="BYOB EMA Dashboard", layout="wide")

# Title
st.title("🚀 BYOB EMA Dashboard")
st.markdown("Welcome to your 0DTE strategy dashboard!")

# --- 📥 Load CSV
@st.cache_data  # Cache to speed up repeated runs
def load_data():
    return pd.read_csv("EMA.csv", parse_dates=["OpenDate"])  # 🔥 Update filename if needed

ema_df = load_data()

# --- 📅 Get dynamic min/max dates from the dataset
min_date = ema_df['OpenDate'].min().date()
max_date = ema_df['OpenDate'].max().date()

# Default start date: 1 year back from max_date
one_year_ago = max_date - datetime.timedelta(days=365)
default_start_date = max(one_year_ago, min_date)  # Prevent going earlier than dataset


# -----------------------------------------------------
# --- ✨ User Input Layout with three columns
# -----------------------------------------------------
col1, col2, col3 = st.columns(3)

# --- 📅 Column 1: Dates
with col1:
    start_date = st.date_input("Start Date", value=default_start_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    
# --- 💵 Column 2: Risk + Entries
with col2:
    equity_start = st.number_input("Starting Equity ($)", value=400_000, step=10_000)
    risk = st.number_input("Risk per Day (%)", value=4.0, step=0.1)
    num_times = st.slider("Number of Entries", min_value=2, max_value=20, value=10)

# --- 🔎 Column 3: Lookbacks
with col3:
    man_near = st.number_input("Near Lookback (months)", value=2, step=1, min_value=1)
    man_mid = st.number_input("Mid Lookback (months)", value=5, step=1, min_value=1)
    man_far = st.number_input("Far Lookback (months)", value=9, step=1, min_value=1)

# --- 🧮 Filter and Calculate (based on user input)
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered_df = ema_df[(ema_df['OpenDate'] >= start_date) & (ema_df['OpenDate'] <= end_date)]

average_credit = filtered_df['Premium'].mean()

# Handle edge cases
if pd.isnull(average_credit) or average_credit <= 0:
    average_credit = 1.0  # Prevent divide by zero

contracts = int(equity_start * (risk / 100) / num_times / (average_credit * 100))

# -----------------------------------------------------
# --- 📋 Summary Expander BELOW user input columns
# -----------------------------------------------------
with st.expander("📋 Summary of Selections + Derived Metrics", expanded=True):
    subcol1, subcol2 = st.columns(2)

    with subcol1:
        st.markdown(f"- **Start Date**: `{start_date.date()}`")
        st.markdown(f"- **End Date**: `{end_date.date()}`")
        st.markdown(f"- **Near Lookback**: `{man_near}` months")
        st.markdown(f"- **Mid Lookback**: `{man_mid}` months")
        st.markdown(f"- **Far Lookback**: `{man_far}` months")

    with subcol2:
        st.markdown(f"- **Starting Equity**: `${equity_start:,.0f}`")
        st.markdown(f"- **Risk per Day**: `{risk:.2f}%`")
        st.markdown(f"- **Number of Entries**: `{num_times}`")
        st.markdown(f"- **Average Credit (per contract)**: `${average_credit:.2f}`")
        st.markdown(f"- **Starting Contracts per Trade**: `{contracts}`")



# -----------------------------------------------------
# --- 🛠️ Helper Functions
# -----------------------------------------------------

def calculate_performance_metrics(equity_curve, equity_column='Equity'):
    """
    Calculate CAGR, MAR ratio, and Sortino ratio from an equity curve.

    Parameters:
        equity_curve (DataFrame): Aggregated equity curve with cumulative equity.
        equity_column (str): Column name to use for cumulative equity.

    Returns:
        Tuple: (CAGR, MAR, Sortino Ratio)
    """
    if equity_curve.empty:
        st.warning("⚠️ Warning: Empty equity curve received. Returning (0, 0, 0).")
        return 0, 0, 0

    # Ensure sorted by date
    equity_curve = equity_curve.sort_values('Date')

    # Calculate the duration
    start_date = equity_curve['Date'].iloc[0]
    end_date = equity_curve['Date'].iloc[-1]
    years = max((end_date - start_date).days / 365.0, 1e-6)

    # Calculate CAGR
    initial_equity = equity_curve[equity_column].iloc[0]
    final_equity = equity_curve[equity_column].iloc[-1]
    if initial_equity <= 0:
        st.warning("⚠️ Warning: Initial equity is zero or negative. Returning (0, 0, 0).")
        return 0, 0, 0
    cagr = (final_equity / initial_equity) ** (1 / years) - 1

    # Calculate MAR ratio
    rolling_max = equity_curve[equity_column].cummax()
    drawdown = (equity_curve[equity_column] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    mar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else float('inf')

    # Calculate Sortino Ratio
    equity_curve['DailyReturn'] = equity_curve[equity_column].pct_change()
    downside_returns = equity_curve['DailyReturn'][equity_curve['DailyReturn'] < 0]
    downside_std = downside_returns.std()
    avg_return = equity_curve['DailyReturn'].mean()
    annualized_return = avg_return * 252
    sortino_ratio = annualized_return / downside_std if downside_std != 0 else float('inf')

    return cagr, mar_ratio, sortino_ratio



# -----------------------------------------------------
# --- 📊 Visualization Tabs
# -----------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Equity Curve", "📅 Monthly Performance", "🎯 Entries Optimization"])

# --- 📈 Tab 1: Equity Curve + Drawdown
with tab1:
    st.subheader("Equity Curve and Drawdown")
    st.info("Chart coming soon...")  # (Placeholder for now)

# --- 📅 Tab 2: Monthly Performance
with tab2:
    st.subheader("Monthly Performance")
    st.info("Chart coming soon...")  # (Placeholder for now)

# --- 🎯 Tab 3: Entries Optimization
with tab3:
    st.subheader("Entries Optimization")
    st.info("Chart coming soon...")  # (Placeholder for now)
