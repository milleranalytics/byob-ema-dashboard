# region --- 📦 Imports & Page Config
# -----------------------------------------------------
import streamlit as st
import pandas as pd
import datetime
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
from itertools import product
from dateutil.relativedelta import relativedelta
import time
import json
from pathlib import Path
import io
from math import floor

# Declare defautls
DEFAULTS_PATH = Path("defaults.json")

def load_defaults():
    if DEFAULTS_PATH.exists():
        with open(DEFAULTS_PATH, "r") as f:
            return json.load(f)
    return {}

def save_defaults(defaults):
    with open(DEFAULTS_PATH, "w") as f:
        json.dump(defaults, f, indent=2)

# Page config
st.set_page_config(page_title="BYOB EMA Dashboard", layout="wide")

# endregion

# User Inputs
defaults = load_defaults()

# Define variables from defaults so they're in scope
man_near = defaults.get("man_near", 2)
man_mid = defaults.get("man_mid", 5)
man_long = defaults.get("man_long", 10)
num_times = defaults.get("num_times", 12)
# default_min_times, default_target_times = defaults.get("num_times", (12, 12))
risk = defaults.get("risk", 4.0)
equity_start = defaults.get("equity_start", 400_000)
credit_target = defaults.get("credit_target", 2.5)
num_times_range = range(3, 25) # Range for num times optimizer
trend_ranking_days = defaults.get("trend_ranking_days", 120)
trend_smoothing_days = defaults.get("trend_smoothing_days", 20)
trend_smoothing_type = defaults.get("trend_smoothing_type", "SMA")

# Title
st.title("BYOB EMA Dashboard")
st.markdown(f"**${credit_target:.2f} Target Credit, 1.5X Stops**")


# region --- 📥 Load CSV & Prepare Data
# -----------------------------------------------------
@st.cache_data  # Cache to speed up repeated runs
def load_data():
    df = pd.read_csv("EMA.csv")

    # --- Format OpenDate ---
    df['OpenDate'] = pd.to_datetime(df['OpenDate'], errors='coerce')

    # --- Format OpenTime (from HH:MM:SS) ---
    df['OpenTime'] = pd.to_timedelta(df['OpenTime'], errors='coerce').dt.total_seconds()

    # --- Create OpenTimeFormatted (HH:MM string) ---
    df['OpenTimeFormatted'] = df['OpenTime'].apply(
        lambda x: '{:02d}:{:02d}'.format(int(x // 3600), int((x % 3600) // 60)) if pd.notnull(x) else '00:00'
    )

    # --- Additional Columns ---
    df['DayOfWeek'] = df['OpenDate'].dt.day_name()
    df['PremiumCapture'] = df['ProfitLossAfterSlippage'] * 100 - df['CommissionFees']
    df['PCR'] = df['PremiumCapture'] / (df['Premium'] * 100)

    return df

ema_df = load_data()

# --- 📅 Get dynamic min/max dates from the dataset
min_date = ema_df['OpenDate'].min().date()
max_date = ema_df['OpenDate'].max().date()

# Default start date: 1 year back from max_date
one_year_ago = max_date - datetime.timedelta(days=365) + datetime.timedelta(days=1)
default_start_date = max(one_year_ago, min_date)  # Prevent going earlier than dataset

# endregion


# region --- ✨ User Input Layout
# -----------------------------------------------------
col1, col2, col3 = st.columns(3)

# --- 📅 Column 1: Dates
with col1:
    start_date = st.date_input(
        "Start Date",
        value=default_start_date,
        min_value=min_date,
        max_value=max_date,
        help="First day of the backtest period. Only trades after this date are included."
    )
    end_date = st.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        help="Last day of the backtest period. Only trades up to and including this date are included."
    )
    selection_method = st.selectbox(
    "Entry Time Selection Method",
    options=["Average PCR", "Time Trends"],
    index=1,
    help="Choose how entry times are selected each period. 'Average PCR' uses a 3-tiered lookback. 'Time Trends' uses cumulative performance with trend filtering."
)

# --- 💵 Column 2: Risk + Entries
with col2:
    equity_start = st.number_input(
        "Starting Equity ($)",
        value=equity_start,
        step=10_000,
        help="Your starting account size..."
    )   
    risk = st.number_input(
        "Risk per Day (%)",
        value=risk,
        step=0.1,
        help="The maximum % of your equity you are willing to risk on a single day. Example: 4% of $400,000 = $16,000 max daily risk.  'Risk' is equal to target credit received for the day assuming -100% PCR is about as bad as it gets, hence 'Risk'."
    )
    num_times = st.slider(
        "Number of Entries",
        min_value=2,
        max_value=24,
        value=num_times,
        help="Number of entry times selected each day."
    )

# --- 🔎 Column 3: Lookbacks
with col3:
    man_near = st.number_input(
        "Near Lookback (months)",
        value=man_near,
        step=1,
        min_value=1,
        help="Short-term lookback period in months used to find best entry times. Emphasizes recent market behavior."
    )
    man_mid = st.number_input(
        "Mid Lookback (months)",
        value=man_mid,
        step=1,
        min_value=1,
        help="Medium-term lookback period in months to smooth entry time selection."
    )
    man_long = st.number_input(
        "Long Lookback (months)",
        value=man_long,
        step=1,
        min_value=1,
        help="Long-term lookback period in months to stabilize entry time selection against outliers."
    )


# endregion


# region --- 🧮 Calculate Initial Variables and Summary expander
# -----------------------------------------------------
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered_df = ema_df[(ema_df['OpenDate'] >= start_date) & (ema_df['OpenDate'] <= end_date)]

average_credit = filtered_df['Premium'].mean()

# Handle edge cases
if pd.isnull(average_credit) or average_credit <= 0:
    average_credit = 1.0  # Prevent divide by zero

contracts = floor(equity_start * (risk / 100) / num_times / (average_credit * 100))



# --- 📋 Summary Expander below user input columns
# -----------------------------------------------------
with st.expander("Metrics & Defaults", expanded=False):

        # 💾 Save button goes here
    if st.button("💾 Save These Settings as Default"):
        new_defaults = {
            "man_near": man_near,
            "man_mid": man_mid,
            "man_long": man_long,
            "num_times": num_times,
            "risk": risk,
            "equity_start": equity_start,
            "credit_target": credit_target
        }
        save_defaults(new_defaults)
        st.success("✅ Defaults saved! Restart the app to load them.")

    subcol1, subcol2, subcol3 = st.columns(3)

    with subcol1:
        st.markdown(f"- **Start Date**: `{start_date.date()}`")
        st.markdown(f"- **End Date**: `{end_date.date()}`")
        st.markdown(f"- **Near Lookback**: `{man_near}` months")
        st.markdown(f"- **Mid Lookback**: `{man_mid}` months")
        st.markdown(f"- **Far Lookback**: `{man_long}` months")

    with subcol2:
        st.markdown(f"- **Starting Equity**: `${equity_start:,.0f}`")
        st.markdown(f"- **Risk per Day**: `{risk:.2f}%`")
        st.markdown(f"- **Number of Entries**: `{num_times}`")
        st.markdown(f"- **Average Credit (per contract)**: `${average_credit:.2f}`")
        st.markdown(f"- **Starting Contracts per Trade**: `{contracts}`")

    with subcol3:
        st.markdown(f"- **Entry Selection Method:** `{selection_method}`")
        st.markdown(f"- **Trend Ranking Days**: `{trend_ranking_days}`")
        st.markdown(f"- **Trend Smoothing Days**: `{trend_smoothing_days}`")
        st.markdown(f"- **Smoothing Type**: `{trend_smoothing_type}`")



# endregion


# region --- 🛠️ Helper Functions
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
    downside_returns = np.minimum(equity_curve['DailyReturn'], 0)
    downside_std = np.sqrt(np.mean(downside_returns**2))

    avg_return = equity_curve['DailyReturn'].mean()
    annualized_return = avg_return * 252

    annualized_downside_std = downside_std * np.sqrt(252)

    sortino_ratio = annualized_return / annualized_downside_std if annualized_downside_std != 0 else float('inf')

    return cagr, mar_ratio, sortino_ratio


def calculate_pcr(df):
    """
    Calculate total Premium Capture Rate (PCR) for a given set of trades.

    Parameters:
        df (DataFrame): A DataFrame containing 'PremiumCapture' and 'Premium' columns.

    Returns:
        float: The Premium Capture Rate as a percentage.
    """
    if df.empty:
        st.warning("⚠️ Warning: Empty dataframe passed to calculate_pcr. Returning 0.0.")
        return 0.0

    total_premium_captured = df['PremiumCapture'].sum()
    total_premium_sold = (df['Premium'] * 100).sum()

    if total_premium_sold == 0:
        st.warning("⚠️ Warning: Total premium sold is zero. Returning 0.0.")
        return 0.0  # Avoid divide-by-zero

    return (total_premium_captured / total_premium_sold) * 100


def calculate_drawdown(equity_curve, equity_column='Equity'):
    """
    Calculate the drawdown and max drawdown from an equity curve.

    Parameters:
        equity_curve (DataFrame): DataFrame containing the equity values over time.
        equity_column (str): Column name for cumulative equity.

    Returns:
        DataFrame: Updated DataFrame with 'Drawdown' column.
        float: Maximum drawdown value (in percentage).
    """
    if equity_curve.empty:
        st.warning("⚠️ Warning: Empty equity curve passed to calculate_drawdown. Returning empty DataFrame and 0.")
        return equity_curve.copy(), 0.0

    if equity_column not in equity_curve.columns:
        st.error(f"❌ Error: Column '{equity_column}' not found in equity_curve DataFrame.")
        return equity_curve.copy(), 0.0

    equity_curve = equity_curve.copy()

    equity_curve['Peak'] = equity_curve[equity_column].cummax()
    equity_curve['Drawdown'] = (equity_curve[equity_column] - equity_curve['Peak']) / equity_curve['Peak'] * 100  # Drawdown in %

    max_drawdown = equity_curve['Drawdown'].min()

    return equity_curve, max_drawdown


def find_best_times(df, top_n):
    """
    Identify the best trading times based on average PCR, ensuring times are sorted chronologically.

    Parameters:
        df (DataFrame): The input DataFrame containing PCR and OpenTimeFormatted.
        top_n (int): Number of top times to select based on PCR.

    Returns:
        List[str]: List of top 'OpenTimeFormatted' times, sorted from earliest to latest.
    """
    if df.empty:
        st.warning("⚠️ Warning: Empty dataframe passed to find_best_times. Returning empty list.")
        return []

    if 'PCR' not in df.columns or 'OpenTimeFormatted' not in df.columns:
        st.error("❌ Error: Required columns 'PCR' or 'OpenTimeFormatted' not found in DataFrame.")
        return []

    df = df.copy()
    df = df.dropna(subset=['PCR'])

    if df.empty:
        st.warning("⚠️ Warning: All PCR values were NaN. Returning empty list.")
        return []

    # Calculate the average PCR for each OpenTimeFormatted
    average_pcr_by_time = df.groupby('OpenTimeFormatted', as_index=False)['PCR'].mean()

    # Limit top_n to available unique times
    top_n = min(top_n, len(average_pcr_by_time))

    # Convert OpenTimeFormatted to actual time for proper sorting
    try:
        average_pcr_by_time['TimeConverted'] = pd.to_datetime(average_pcr_by_time['OpenTimeFormatted'], format='%H:%M').dt.time
    except Exception as e:
        st.error(f"❌ Error parsing OpenTimeFormatted times: {e}")
        return []

    # Select top N times based on PCR, then sort by real time
    best_times_sorted = (
        average_pcr_by_time.nlargest(top_n, 'PCR')
        .sort_values(by='TimeConverted')
        .reset_index(drop=True)['OpenTimeFormatted']
        .tolist()
    )

    return best_times_sorted


def mark_best_times(df, best_times):
    """
    Mark rows as 'BestTime' based on the top selected times.

    Parameters:
        df (DataFrame): The input DataFrame containing 'OpenTimeFormatted'.
        best_times (List[str]): List of top times to mark as 'BestTime'.

    Returns:
        DataFrame: Modified DataFrame with a 'BestTime' column indicating selected rows.
    """
    if df.empty:
        st.warning("⚠️ Warning: Empty dataframe passed to mark_best_times. Returning empty DataFrame.")
        return df.copy()

    if 'OpenTimeFormatted' not in df.columns:
        st.error("❌ Error: 'OpenTimeFormatted' column not found in DataFrame.")
        return df.copy()

    if not best_times:
        st.warning("⚠️ Warning: No best_times provided to mark. Returning unmodified DataFrame.")
        df = df.copy()
        df['BestTime'] = 0
        return df

    df = df.copy()
    df['BestTime'] = df['OpenTimeFormatted'].isin(best_times).astype(int)  # 1 for best times, 0 otherwise

    return df


def select_times_via_time_trends(ema_df, end_date, num_times, ranking_window, smoothing_window, smoothing_type):
    recent_df = ema_df[ema_df['OpenDate'] <= end_date].copy()

    recent_dates = recent_df['OpenDate'].drop_duplicates().sort_values().tail(ranking_window)
    recent_df = recent_df[recent_df['OpenDate'].isin(recent_dates)]

    pnl_by_time = (
        recent_df.groupby(['OpenDate', 'OpenTimeFormatted'])['PremiumCapture']
        .sum()
        .reset_index()
        .pivot(index='OpenDate', columns='OpenTimeFormatted', values='PremiumCapture')
        .fillna(0)
    )

    cumulative = pnl_by_time.cumsum()

    if smoothing_type.upper() == 'EMA':
        trend = cumulative.ewm(span=smoothing_window, min_periods=1).mean()
    else:
        trend = cumulative.rolling(window=smoothing_window, min_periods=1).mean()

    final_cum = cumulative.iloc[-1]
    final_trend = trend.iloc[-1]

    # Select only those outperforming their trend
    selected = final_cum[final_cum > final_trend]

    if selected.empty:
        return []

    top_times = selected.sort_values(ascending=False).head(num_times).index.tolist()
    return sorted(top_times)


# WILL REMOVE THIS WHEN THE DYNAMIC ONE IS IN FULL EFFECT
def calculate_equity_curve_with_manual_lookbacks(
    ema_df, start_date, end_date, equityStart, risk, num_times, man_near, man_mid, man_long, average_credit
):
    import pandas as pd
    import streamlit as st

    if ema_df.empty:
        st.warning("⚠️ Warning: Empty EMA DataFrame passed to equity curve calculation.")
        return pd.DataFrame(), pd.DataFrame()

    ema_df = ema_df.copy()
    ema_df['OpenDate'] = pd.to_datetime(ema_df['OpenDate'])
    ema_df = ema_df.sort_values('OpenDate')

    current_equity = equityStart
    results = []

    # Filter to the final test window
    filtered_periods = ema_df[(ema_df['OpenDate'] >= pd.to_datetime(start_date)) & 
                              (ema_df['OpenDate'] <= pd.to_datetime(end_date))].copy()
    filtered_periods['PeriodStart'] = filtered_periods['OpenDate'].dt.to_period('M').dt.start_time
    unique_periods = filtered_periods['PeriodStart'].unique()

    for current_period in unique_periods:
        near_start = current_period - pd.DateOffset(months=man_near)
        mid_start = current_period - pd.DateOffset(months=man_mid)
        long_start = current_period - pd.DateOffset(months=man_long)
        lookback_end = current_period - pd.Timedelta(days=1)

        near_data = ema_df[(ema_df['OpenDate'] >= near_start) & (ema_df['OpenDate'] <= lookback_end)]
        mid_data = ema_df[(ema_df['OpenDate'] >= mid_start) & (ema_df['OpenDate'] <= lookback_end)]
        long_data = ema_df[(ema_df['OpenDate'] >= long_start) & (ema_df['OpenDate'] <= lookback_end)]

        if near_data.empty and mid_data.empty and long_data.empty:
            st.warning(f"⚠️ Skipping {current_period.strftime('%Y-%m')} — no available lookback data.")
            continue

        lookback_dfs = []
        if not near_data.empty:
            lookback_dfs.append(near_data.groupby('OpenTimeFormatted')['PCR'].mean().rename('Near_PCR'))
        if not mid_data.empty:
            lookback_dfs.append(mid_data.groupby('OpenTimeFormatted')['PCR'].mean().rename('Mid_PCR'))
        if not long_data.empty:
            lookback_dfs.append(long_data.groupby('OpenTimeFormatted')['PCR'].mean().rename('Long_PCR'))

        avg_pcr_df = pd.concat(lookback_dfs, axis=1).mean(axis=1).reset_index()
        avg_pcr_df.columns = ['OpenTimeFormatted', 'PCR']

        top_times_sorted = avg_pcr_df.nlargest(num_times, 'PCR') \
                                     .sort_values('OpenTimeFormatted')['OpenTimeFormatted'] \
                                     .tolist()

        current_period_data = ema_df[(ema_df['OpenDate'] >= current_period) & 
                                     (ema_df['OpenDate'] < current_period + pd.DateOffset(months=1))]

        current_period_data = current_period_data[current_period_data['OpenTimeFormatted'].isin(top_times_sorted)]
        current_period_data = current_period_data[current_period_data['OpenDate'] >= pd.to_datetime(start_date)]

        if current_period_data.empty:
            continue

        for trade_date, day_trades in current_period_data.groupby('OpenDate'):
            equity_at_day_start = equityStart if trade_date == pd.to_datetime(start_date) else current_equity
            day_trades = day_trades.sort_values('OpenTimeFormatted')
            contracts = floor(equity_at_day_start * (risk / 100) / num_times / (average_credit * 100)) if average_credit else 0

            for _, trade in day_trades.iterrows():
                profit_loss = trade['PremiumCapture'] * contracts
                current_equity += profit_loss
                results.append({
                    'Date': trade['OpenDate'],
                    'OpenTime': trade['OpenTimeFormatted'],
                    'Contracts': contracts,
                    'PremiumCapture': trade['PremiumCapture'],
                    'Premium': trade['Premium'],
                    'ProfitLoss': profit_loss,
                    'Equity': current_equity,
                    'NearLookback': man_near,
                    'NearLookbackStart': near_start,
                    'NearLookbackEnd': lookback_end,
                    'MidLookback': man_mid,
                    'MidLookbackStart': mid_start,
                    'MidLookbackEnd': lookback_end,
                    'LongLookback': man_long,
                    'LongLookbackStart': long_start,
                    'LongLookbackEnd': lookback_end,
                })


    results_df = pd.DataFrame(results)
    filtered_results = results_df[(results_df['Date'] >= pd.to_datetime(start_date)) & 
                                  (results_df['Date'] <= pd.to_datetime(end_date))]
    daily_equity = filtered_results.groupby('Date', as_index=False)['Equity'].last().sort_values(by='Date')

    return daily_equity, filtered_results


def calculate_equity_curve_with_dynamic_method(
    ema_df,
    start_date,
    end_date,
    equityStart,
    risk,
    num_times,
    man_near,
    man_mid,
    man_long,
    average_credit,
    selection_method,
    trend_ranking_days,
    trend_smoothing_days,
    trend_smoothing_type
):
    if ema_df.empty:
        st.warning("⚠️ Warning: Empty EMA DataFrame passed to equity curve calculation.")
        return pd.DataFrame(), pd.DataFrame()

    ema_df = ema_df.copy()
    ema_df['OpenDate'] = pd.to_datetime(ema_df['OpenDate'])
    ema_df = ema_df.sort_values('OpenDate')
    current_equity = equityStart
    results = []

    # Define loop frequency
    if selection_method == "Time Trends":
        first_monday = start_date - datetime.timedelta(days=start_date.weekday())
        loop_dates = pd.date_range(start=first_monday, end=end_date, freq='W-MON')
    else:
        ema_df['PeriodStart'] = ema_df['OpenDate'].dt.to_period('M').dt.start_time
        loop_dates = ema_df[
            (ema_df['OpenDate'] >= pd.to_datetime(start_date)) &
            (ema_df['OpenDate'] <= pd.to_datetime(end_date))
        ]['PeriodStart'].unique()

    for current_period in loop_dates:
        if selection_method == "Average PCR":
            near_start = current_period - pd.DateOffset(months=man_near)
            mid_start = current_period - pd.DateOffset(months=man_mid)
            long_start = current_period - pd.DateOffset(months=man_long)
            lookback_end = current_period - pd.Timedelta(days=1)

            near_data = ema_df[(ema_df['OpenDate'] >= near_start) & (ema_df['OpenDate'] <= lookback_end)]
            mid_data = ema_df[(ema_df['OpenDate'] >= mid_start) & (ema_df['OpenDate'] <= lookback_end)]
            long_data = ema_df[(ema_df['OpenDate'] >= long_start) & (ema_df['OpenDate'] <= lookback_end)]

            lookback_dfs = []
            if not near_data.empty:
                lookback_dfs.append(near_data.groupby('OpenTimeFormatted')['PCR'].mean().rename('Near_PCR'))
            if not mid_data.empty:
                lookback_dfs.append(mid_data.groupby('OpenTimeFormatted')['PCR'].mean().rename('Mid_PCR'))
            if not long_data.empty:
                lookback_dfs.append(long_data.groupby('OpenTimeFormatted')['PCR'].mean().rename('Long_PCR'))

            if not lookback_dfs:
                continue

            avg_pcr_df = pd.concat(lookback_dfs, axis=1).mean(axis=1).reset_index()
            avg_pcr_df.columns = ['OpenTimeFormatted', 'PCR']
            top_times_sorted = avg_pcr_df.nlargest(num_times, 'PCR') \
                                         .sort_values('OpenTimeFormatted')['OpenTimeFormatted'] \
                                         .tolist()

        elif selection_method == "Time Trends":
            top_times_sorted = select_times_via_time_trends(
                ema_df=ema_df,
                end_date=current_period,
                num_times=num_times,
                ranking_window=trend_ranking_days,
                smoothing_window=trend_smoothing_days,
                smoothing_type=trend_smoothing_type
            )
        else:
            continue

        if selection_method == "Time Trends":
            period_end = min(current_period + pd.Timedelta(days=7), end_date + pd.Timedelta(days=1))

            current_period_data = ema_df[
                (ema_df['OpenDate'] >= current_period) &
                (ema_df['OpenDate'] < period_end)
            ]
        else:
            current_period_data = ema_df[
                (ema_df['OpenDate'] >= current_period) &
                (ema_df['OpenDate'] < current_period + pd.DateOffset(months=1))
            ]

        current_period_data = current_period_data[current_period_data['OpenTimeFormatted'].isin(top_times_sorted)]
        current_period_data = current_period_data[current_period_data['OpenDate'] >= pd.to_datetime(start_date)]

        if current_period_data.empty:
            continue

        for trade_date, day_trades in current_period_data.groupby('OpenDate'):
            equity_at_day_start = equityStart if trade_date == pd.to_datetime(start_date) else current_equity
            contracts = floor(equity_at_day_start * (risk / 100) / num_times / (average_credit * 100)) if average_credit else 0
            day_trades = day_trades.sort_values('OpenTimeFormatted')

            for _, trade in day_trades.iterrows():
                profit_loss = trade['PremiumCapture'] * contracts
                current_equity += profit_loss

                row = {
                    'Date': trade['OpenDate'],
                    'OpenTime': trade['OpenTimeFormatted'],
                    'Contracts': contracts,
                    'PremiumCapture': trade['PremiumCapture'],
                    'Premium': trade['Premium'],
                    'ProfitLoss': profit_loss,
                    'Equity': current_equity,
                    'SelectionMethod': selection_method
                }

                if selection_method == "Average PCR":
                    row.update({
                        'NearLookback': man_near,
                        'NearLookbackStart': near_start,
                        'NearLookbackEnd': lookback_end,
                        'MidLookback': man_mid,
                        'MidLookbackStart': mid_start,
                        'MidLookbackEnd': lookback_end,
                        'LongLookback': man_long,
                        'LongLookbackStart': long_start,
                        'LongLookbackEnd': lookback_end
                    })

                results.append(row)

    results_df = pd.DataFrame(results)
    daily_equity = results_df.groupby('Date', as_index=False)['Equity'].last().sort_values(by='Date')

    return daily_equity, results_df


# --- 📈 Standard Figure Template to ensure charts are the same

def create_standard_fig(rows=1, cols=1, row_heights=None, vertical_spacing=0.1, height=600, theme="auto"):
    """
    Create a standard plotly figure with preset dark/light theme, margins, and sizing.
    """

    # Detect if in dark mode
    theme_bg = st.get_option("theme.backgroundColor")
    is_dark = True  # Default to dark
    if theme == "light":
        is_dark = False
    elif theme == "dark":
        is_dark = True
    elif theme_bg is not None and theme_bg.lower() == "#ffffff":
        is_dark = False

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        row_heights=row_heights
    )

    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(color="white" if is_dark else "black"),
        paper_bgcolor="rgba(0,0,0,0)" if is_dark else "white",
        plot_bgcolor="rgba(0,0,0,0)" if is_dark else "white",
        showlegend=False
    )

    return fig, is_dark  # ✅ return both the figure and the dark mode flag


# --- 📋 Standard Table Template

def create_standard_table(df, negative_cols=None, decimals=1, theme="auto", height=None, column_widths=None):
    """
    Create a standard plotly Table styled to match Streamlit dark/light mode.

    Parameters:
    - df (DataFrame): Pandas DataFrame to display
    - negative_cols (List[str], optional): Columns where negative numbers should be red
    - decimals (int): Number of decimals to show
    - theme (str): "auto", "dark", or "light"
    - height (int, optional): Manually set table height if needed
    - column_widths (List[int], optional): Relative widths for each column
    """

    import plotly.graph_objects as go

    # --- Theme detection
    theme_bg = st.get_option("theme.backgroundColor")
    is_dark = True  # default
    if theme == "light":
        is_dark = False
    elif theme == "dark":
        is_dark = True
    elif theme_bg is not None and theme_bg.lower() == "#ffffff":
        is_dark = False

    # --- Colors
    header_fill = '#262730' if is_dark else '#f7f7f7'
    header_font = 'white' if is_dark else 'black'
    cell_fill = '#0e1117' if is_dark else 'white'
    default_font = 'white' if is_dark else 'black'
    neg_font = 'rgba(234,92,81,1)'

    # --- Build values and font colors
    values = []
    font_colors = []
    fill_colors = []

    # Detect header name for the index column
    if df.index.name:
        index_col = df.index.name
    else:
        index_col = "Year"

    values.append(df.index.tolist())
    font_colors.append([default_font] * len(df))
    fill_colors.append([
        header_fill if idx == 'AVG' else cell_fill
        for idx in df.index
    ])

    for col in df.columns:
        col_vals = df[col].tolist()
        formatted = []
        colors = []
        backgrounds = []

        for idx, val in enumerate(col_vals):
            row_label = df.index[idx]

            # Format values
            if pd.isna(val):
                formatted.append("-")
            else:
                if isinstance(val, (int, float)):
                    formatted.append(f"{val:.{decimals}f}%")
                else:
                    formatted.append(str(val))

            # Font color
            if negative_cols and col in negative_cols and isinstance(val, (int, float)) and val < 0:
                colors.append(neg_font)
            else:
                colors.append(default_font)

            # Background color
            if row_label == 'AVG':
                backgrounds.append(header_fill)
            else:
                backgrounds.append(cell_fill)

        values.append(formatted)
        font_colors.append(colors)
        fill_colors.append(backgrounds)

    # --- Create Table with optional column widths
    fig = go.Figure(data=[go.Table(
        columnwidth=column_widths,  # ✅ Added here
        header=dict(
            values=[index_col] + list(df.columns),
            fill_color=header_fill,
            font_color=header_font,
            font_size=14,
            align='center'
        ),
        cells=dict(
            values=values,
            fill_color=fill_colors,
            font_color=font_colors,
            align='center',
            font_size=14,
            height=30
        )
    )])

    # Auto height if not specified
    if not height:
        height = 400 + 30 * len(df)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=height
    )

    return fig

# endregion


# region --- 📊 Visualization Tabs Structure
# -----------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Equity Curve",
    "🔎 Entries Optimization",
    "🔎 Risk Optimization",
    "🎯 Entry Time PCR",
    "📈 Entry Time Trends",
    "🔎 Lookback Optimization",
    "🔎 Trend Optimization",
    "📖 Instructions"
])
# endregion


# region --- 📈 Tab 1: Equity Curve + Drawdown
# -----------------------------------------------------

# --- Monthly Performance Table
def display_monthly_performance_table(equity_df, is_dark=True, equity_col='Equity', equity_start=equity_start):
    import calendar

    df = equity_df.copy()
    df['Month'] = df['Date'].dt.to_period('M')
    df['Year'] = df['Date'].dt.year
    df['MonthName'] = df['Date'].dt.month.apply(lambda x: calendar.month_abbr[x])

    # Monthly returns
    monthly_returns = []
    unique_months = df['Month'].unique()

    for month in unique_months:
        month_data = df[df['Month'] == month]
        if month_data.empty:
            continue

        month_end_equity = month_data[equity_col].iloc[-1]

        # Get the last equity from before this month starts
        prev_data = df[df['Date'] < month_data['Date'].iloc[0]]
        if not prev_data.empty:
            month_start_equity = prev_data[equity_col].iloc[-1]
        else:
            month_start_equity = equity_start

        month_return = (month_end_equity / month_start_equity - 1) * 100

        monthly_returns.append({
            'Month': month,
            'Return': month_return,
            'Year': month.year,
            'MonthName': calendar.month_abbr[month.month]
        })

    monthly_returns = pd.DataFrame(monthly_returns)

    # Max Drawdown per year
    drawdowns = []
    for year, group in df.groupby(df['Date'].dt.year):
        peak = group[equity_col].cummax()
        dd = (group[equity_col] - peak) / peak
        drawdowns.append({'YEAR': year, 'MaxDD': dd.min() * 100})
    dd_df = pd.DataFrame(drawdowns).set_index('YEAR')

    # Pivot table
    month_order = [calendar.month_abbr[i] for i in range(1, 13)]
    pivot = monthly_returns.pivot(index='Year', columns='MonthName', values='Return')
    pivot = pivot.reindex(columns=month_order)

    # --- ✅ Begin/End Balances with correct start logic
    equity_df = equity_df.set_index("Date")
    years = equity_df.index.year.unique()
    start_balances = {}
    end_balances = {}

    for year in years:
        # Find last date in prior year
        prior_year = year - 1
        prior_data = equity_df[equity_df.index.year == prior_year]
        if not prior_data.empty:
            begin = prior_data.iloc[-1][equity_col]
        else:
            begin = equity_start

        # Find last date in current year
        year_data = equity_df[equity_df.index.year == year]
        if not year_data.empty:
            end = year_data.iloc[-1][equity_col]
        else:
            end = None  # You could choose to skip this year if no data

        start_balances[year] = begin
        end_balances[year] = end

    start_bal = pd.Series(start_balances)
    end_bal = pd.Series(end_balances)
    pl_dollars = (end_bal - start_bal).round(2)
    pl_percent = ((end_bal / start_bal - 1) * 100).round(1)

    # Add new columns
    pivot.insert(0, "Begin Bal", start_bal.round(2))
    pivot["End Bal"] = end_bal.round(2)
    pivot["P/L ($)"] = pl_dollars
    pivot["P/L (%)"] = pl_percent
    pivot["MaxDD"] = dd_df['MaxDD'].round(1)

    # Add AVG row
    avg_row = pivot[month_order + ["P/L (%)"]].mean(numeric_only=True)
    avg_row["Begin Bal"] = None
    avg_row["End Bal"] = None
    avg_row["P/L ($)"] = None
    avg_row["MaxDD"] = None
    pivot.loc["AVG"] = avg_row
    pivot.index.name = None
    pivot.columns.name = None

    # Format for display
    display_df = pivot.copy()
    for col in ["Begin Bal", "End Bal", "P/L ($)"]:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "")
    display_df["P/L (%)"] = display_df["P/L (%)"].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
    display_df["MaxDD"] = display_df["MaxDD"].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")

    # --- 📊 Final Table Render
    # Widen specific columns and include the index
    wide_cols = ["Begin Bal", "End Bal", "P/L ($)"]
    index_width = 60  # width for Year/index column

    column_widths = [index_width]  # start with index column

    # Loop through actual dataframe columns
    for col in display_df.columns:
        if col in wide_cols:
            column_widths.append(75)  # wide columns
        else:
            column_widths.append(60)  # default width

    # Call the updated table function
    fig = create_standard_table(
        df=display_df,
        negative_cols=month_order,
        decimals=1,
        theme="dark" if is_dark else "light",
        height=250,
        column_widths=column_widths  # ✅ this now works
    )

    st.plotly_chart(fig, use_container_width=True)


with tab1:
    # Dynamic lookback string
    if selection_method == "Average PCR":
        lookback_str = f"Average PCR Lookbacks: Near {man_near}M / Mid {man_mid}M / Long {man_long}M"
    else:
        lookback_str = f"Trend Ranking: Last {trend_ranking_days} days / {trend_smoothing_days}-day {trend_smoothing_type.upper()}"
    st.subheader(f"Equity Curve and Drawdown ({start_date.date()} to {end_date.date()})")
    st.markdown(
        f"##### Target Credit: ${credit_target:.2f} | Entries: {num_times} | Risk: {risk:.1f}% | {lookback_str}"
    )

    # Reserve container for all UI output
    equity_container = st.empty()

    # Spinner gives grayed-out feedback while calculations are running
    with equity_container.container():
        with st.spinner("🔄 Calculating equity curve and performance metrics..."):
            equity_curve, full_trades = calculate_equity_curve_with_dynamic_method(
                ema_df=ema_df,
                start_date=start_date,
                end_date=end_date,
                equityStart=equity_start,
                risk=risk,
                num_times=num_times,
                man_near=man_near,
                man_mid=man_mid,
                man_long=man_long,
                average_credit=average_credit,
                selection_method=selection_method,  # <-- from the dropdown
                trend_ranking_days=trend_ranking_days,
                trend_smoothing_days=trend_smoothing_days,
                trend_smoothing_type=trend_smoothing_type
            )

            if equity_curve.empty:
                st.warning("⚠️ No trades generated for the selected period and parameters.")
            else:
                equity_curve_with_dd, max_drawdown = calculate_drawdown(equity_curve)
                pcr = calculate_pcr(full_trades)
                cagr, mar_ratio, sortino = calculate_performance_metrics(equity_curve_with_dd)

                # --- ✅ Win Rate Calculation
                total_trades = len(full_trades)
                wins = len(full_trades[full_trades["ProfitLoss"] > 0])
                win_rate = wins / total_trades if total_trades > 0 else 0

                # --- 📈 Top Metrics Display (6 columns now)
                col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
                col1.metric("CAGR", f"{cagr:.2%}")
                col2.metric("MAR Ratio", f"{mar_ratio:.2f}")
                col3.metric("Sortino Ratio", f"{sortino:.2f}")
                col4.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                col5.metric("PCR", f"{pcr:.2f}%")
                col6.metric("Win Rate", f"{win_rate:.2%}")

                # --- 📈 Chart
                fig, is_dark = create_standard_fig(rows=2, cols=1, row_heights=[0.6, 0.4], height=800)
                streamlit_red = 'rgba(234,92,81,1)'
                streamlit_red_fill = 'rgba(234,92,81,0.15)'

                fig.add_trace(go.Scatter(
                    x=equity_curve_with_dd['Date'],
                    y=equity_curve_with_dd['Equity'],
                    mode='lines',
                    name='Equity Curve',
                    line=dict(width=2),
                    hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>'
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=equity_curve_with_dd['Date'],
                    y=equity_curve_with_dd['Drawdown'],
                    mode='lines',
                    name='Drawdown (%)',
                    line=dict(width=2, color=streamlit_red),
                    fill='tozeroy',
                    fillcolor=streamlit_red_fill,
                    hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>'
                ), row=2, col=1)

                fig.update_yaxes(title_text="Equity ($)", type="log", row=1, col=1)
                fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

                st.plotly_chart(fig, use_container_width=True)

                # --- 📅 Monthly Performance Table
                st.subheader("Monthly Performance Table")
                display_monthly_performance_table(equity_curve_with_dd, is_dark=True)

                st.subheader("Detailed Trade Insights")

                expander_col, button_col = st.columns([2, 1])

                # --- 📅 Monthly Summary
                with expander_col:
                    with st.expander("📅 Trading Summary"):
                        if selection_method == "Average PCR":
                            period_group = full_trades['Date'].dt.to_period('M')
                            label_format = "%Y-%m"
                        else:
                            # Group by Monday of each week (start of trading week)
                            period_group = full_trades['Date'] - pd.to_timedelta(full_trades['Date'].dt.weekday, unit='D')
                            period_group = period_group.dt.to_period('W')
                            label_format = "Week of %Y-%m-%d"

                        for period, trades in full_trades.groupby(period_group):
                            period_start_date = period.start_time if hasattr(period, 'start_time') else period

                            if selection_method == "Average PCR":
                                st.markdown(f"##### {period.strftime('%Y-%m')}")
                            else:
                                st.markdown(f"##### Week of {period_start_date.strftime('%Y-%m-%d')}")

                            selected_times = sorted(trades['OpenTime'].unique())
                            selected_times_str = ', '.join(selected_times) if selected_times else "No trades"
                            end_equity = trades['Equity'].iloc[-1]

                            if selection_method == "Average PCR":
                                # Show lookback details only for PCR-based selection
                                far_start = trades['LongLookbackStart'].min()
                                far_end = trades['LongLookbackEnd'].max()
                                mid_start = trades['MidLookbackStart'].min()
                                mid_end = trades['MidLookbackEnd'].max()
                                near_start = trades['NearLookbackStart'].min()
                                near_end = trades['NearLookbackEnd'].max()

                                st.markdown(f"📅 **Far Lookback:** {man_long} months — {far_start.strftime('%m/%d/%Y')} to {far_end.strftime('%m/%d/%Y')}")
                                st.markdown(f"📅 **Mid Lookback:** {man_mid} months — {mid_start.strftime('%m/%d/%Y')} to {mid_end.strftime('%m/%d/%Y')}")
                                st.markdown(f"📅 **Near Lookback:** {man_near} months — {near_start.strftime('%m/%d/%Y')} to {near_end.strftime('%m/%d/%Y')}")
                            else:
                                # Get all unique trading days, sorted
                                trading_days = sorted(ema_df['OpenDate'].unique())

                                # First trade date for this week
                                first_trade_date = trades['Date'].min()

                                # Anchor to Monday of the week (start of that trading period)
                                monday_of_week = first_trade_date - datetime.timedelta(days=first_trade_date.weekday())

                                # Get all trading days strictly before this Monday
                                prior_trading_days = [d for d in trading_days if d < monday_of_week]

                                # Take the most recent N days before this Monday
                                recent_trading_days = prior_trading_days[-trend_ranking_days:]

                                # Determine actual start and end used in time trend logic
                                if recent_trading_days:
                                    trend_start = recent_trading_days[0]
                                    trend_end = recent_trading_days[-1]
                                else:
                                    trend_start = trend_end = None  # fallback if not enough data

                                if trend_start and trend_end:
                                    st.markdown(
                                        f"🔁 **Trend Ranking:** Last {trend_ranking_days} days "
                                        f"({trend_start.strftime('%Y-%m-%d')} to {trend_end.strftime('%Y-%m-%d')}) | "
                                        f"{trend_smoothing_days}-day {trend_smoothing_type.upper()}"
                                    )
                                else:
                                    st.markdown("⚠️ Not enough historical trading days for trend analysis.")
                            
                            st.markdown(f"⏰ **Selected Times:** {selected_times_str}")
                            st.markdown(f"💵 **End Equity:** ${end_equity:,.2f}")
                            
                with button_col:
                    if full_trades is not None and not full_trades.empty:
                    # ⏱️ Create sorted and trimmed DataFrame for export
                        export_cols = [
                            'Date', 'OpenTime', 'Contracts', 'PremiumCapture', 'ProfitLoss', 'Equity',
                            'NearLookback', 'NearLookbackStart', 'NearLookbackEnd',
                            'MidLookback', 'MidLookbackStart', 'MidLookbackEnd',
                            'LongLookback', 'LongLookbackStart', 'LongLookbackEnd'
                        ]
                        
                        # Only include columns that actually exist (some may be missing in edge cases)
                        export_cols = [col for col in export_cols if col in full_trades.columns]
                        
                        full_trades_sorted = full_trades[export_cols].sort_values(by=["Date", "OpenTime"])

                        # Round dollar columns to 2 decimal places
                        for col in ['PremiumCapture', 'ProfitLoss', 'Equity']:
                            if col in full_trades_sorted.columns:
                                full_trades_sorted[col] = full_trades_sorted[col].round(2)

                        # Format all date columns to YYYY-MM-DD strings
                        date_cols = [
                            'Date',
                            'NearLookbackStart', 'NearLookbackEnd',
                            'MidLookbackStart', 'MidLookbackEnd',
                            'LongLookbackStart', 'LongLookbackEnd'
                        ]

                        for col in date_cols:
                            if col in full_trades_sorted.columns:
                                full_trades_sorted[col] = pd.to_datetime(full_trades_sorted[col]).dt.strftime('%Y-%m-%d')

                        # Set up in-memory Excel export
                        excel_buffer = io.BytesIO()

                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            full_trades_sorted.to_excel(writer, index=False, sheet_name='Trades')

                            workbook = writer.book
                            worksheet = writer.sheets['Trades']

                            # Define formats
                            header_format = workbook.add_format({'bold': True, 'bg_color': '#333333', 'font_color': 'white'})
                            dollar_format = workbook.add_format({'num_format': '$#,##0.00'})

                            # Apply header formatting
                            for col_num, col_name in enumerate(full_trades_sorted.columns):
                                worksheet.write(0, col_num, col_name, header_format)

                            # Auto width and conditional formatting
                            for i, col in enumerate(full_trades_sorted.columns):
                                max_len = max(full_trades_sorted[col].astype(str).map(len).max(), len(col))
                                format_to_use = dollar_format if col in ['PremiumCapture', 'ProfitLoss', 'Equity'] else None
                                worksheet.set_column(i, i, max_len + 2, format_to_use)

                        # Streamlit download button
                        st.download_button(
                            label="📊 Download Excel Trade Log (.xlsx)",
                            data=excel_buffer.getvalue(),
                            file_name="equity_curve_trades.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

# endregion


# region --- 🎯 Tab 2: Entries Optimization
# -----------------------------------------------------
def optimize_num_entries(
    ema_df, num_times_range, equityStart, risk, start_date, end_date,
    man_near, man_mid, man_long,
    average_credit, selection_method,
    trend_ranking_days, trend_smoothing_days, trend_smoothing_type,
    performance_func
):
    """
    Optimize number of entries for manual lookbacks, returning CAGR, MAR, Sortino, and Max Drawdown.
    """
    results = []

    for num_times in num_times_range:
        daily_equity_manual, _ = calculate_equity_curve_with_dynamic_method(
            ema_df=ema_df,
            start_date=start_date,
            end_date=end_date,
            equityStart=equityStart,
            risk=risk,
            num_times=num_times,
            man_near=man_near,
            man_mid=man_mid,
            man_long=man_long,
            average_credit=average_credit,
            selection_method=selection_method,  # uses your dropdown
            trend_ranking_days=trend_ranking_days,
            trend_smoothing_days=trend_smoothing_days,
            trend_smoothing_type=trend_smoothing_type
        )

        if daily_equity_manual.empty:
            continue

        cagr, mar, sortino = performance_func(daily_equity_manual)
        _, max_drawdown = calculate_drawdown(daily_equity_manual)

        results.append({
            'NumEntries': num_times,
            'CAGR': cagr,
            'MAR': mar,
            'Sortino': sortino,
            'MaxDrawdown': max_drawdown
        })

    if not results:
        st.error("❌ No valid results found for the provided num_times range.")
        return pd.DataFrame()

    return pd.DataFrame(results)


# --- 📈 Cached Optimization Sweep
@st.cache_data(show_spinner=False)
def run_num_entries_sweep(
    ema_df, start_date, end_date, equity_start, risk,
    man_near, man_mid, man_long,
    average_credit, selection_method,
    trend_ranking_days, trend_smoothing_days, trend_smoothing_type
):
    return optimize_num_entries(
        ema_df=ema_df,
        num_times_range=num_times_range,
        equityStart=equity_start,
        risk=risk,
        start_date=start_date,
        end_date=end_date,
        man_near=man_near,
        man_mid=man_mid,
        man_long=man_long,
        average_credit=average_credit,
        selection_method=selection_method,
        trend_ranking_days=trend_ranking_days,
        trend_smoothing_days=trend_smoothing_days,
        trend_smoothing_type=trend_smoothing_type,
        performance_func=calculate_performance_metrics
    )


# --- 🎯 Tab 2: Entries Optimization
with tab2:
    if selection_method == "Average PCR":
        lookback_str = f"Average PCR Lookbacks: Near {man_near}M / Mid {man_mid}M / Long {man_long}M"
    else:
        lookback_str = f"Trend Ranking: Last {trend_ranking_days} days / {trend_smoothing_days}-day {trend_smoothing_type.upper()}"
    st.subheader(f"Entries Optimization Analysis ({start_date.date()} to {end_date.date()})")
    st.markdown(
        f"##### Target Credit: ${credit_target:.2f} | Risk: {risk:.1f}% | {lookback_str}"
    )

    # Reserve UI space early
    optimization_container = st.empty()

    # Define columns up front so layout doesn't flicker
    col1, col2 = st.columns(2)

    if ema_df.empty:
        with optimization_container.container():
            st.warning("⚠️ No data available.")
    else:
        # Run calculation inside the container and spinner
        with optimization_container.container():
            with st.spinner("🔄 Optimizing number of entries..."):
                optimization_results = run_num_entries_sweep(
                    ema_df=ema_df,
                    start_date=start_date,
                    end_date=end_date,
                    equity_start=equity_start,
                    risk=risk,
                    man_near=man_near,
                    man_mid=man_mid,
                    man_long=man_long,
                    average_credit=average_credit,
                    selection_method=selection_method,
                    trend_ranking_days=trend_ranking_days,
                    trend_smoothing_days=trend_smoothing_days,
                    trend_smoothing_type=trend_smoothing_type
                )

                if optimization_results.empty:
                    st.warning("⚠️ No valid optimization results.")
                else:
                    metrics_to_plot = [
                        ("CAGR", "CAGR", None),
                        ("MAR Ratio", "MAR", "#2ca02c"),
                        ("Sortino Ratio", "Sortino", "#9467bd"),
                        ("Max Drawdown (%)", "MaxDrawdown", "rgba(234,92,81,1)")
                    ]

                    for idx, (metric_title, metric_column, color) in enumerate(metrics_to_plot):
                        fig, is_dark = create_standard_fig(height=400)

                        fig.add_trace(go.Scatter(
                            x=optimization_results['NumEntries'],
                            y=optimization_results[metric_column],
                            mode='lines+markers',
                            marker=dict(size=6),
                            line=dict(width=2) if color is None else dict(width=2, color=color),
                            name=metric_title,
                        ))

                        fig.add_vline(
                            x=num_times,
                            line_dash="dash",
                            line_color="rgba(234,234,234,0.4)",
                            line_width=1,
                            opacity=0.8
                        )

                        if metric_column == "CAGR":
                            yaxis_tickformat = ".0%"
                            hovertemplate = '%{y:.2%}<extra></extra>'
                        else:
                            yaxis_tickformat = None
                            hovertemplate = '%{y:.2f}<extra></extra>'

                        fig.data[0].hovertemplate = hovertemplate

                        fig.update_layout(
                            title=metric_title,
                            xaxis_title="Number of Entries",
                            yaxis_title=metric_title,
                            xaxis=dict(tickmode='linear', dtick=1),
                            yaxis_tickformat=yaxis_tickformat,
                        )

                        # Alternate columns
                        if idx % 2 == 0:
                            with col1:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            with col2:
                                st.plotly_chart(fig, use_container_width=True)


# endregion


# region --- 📈 Tab 3: Risk Optimization
# -----------------------------------------------------

# --- 📈 Helper: Optimize Risk (with caching)
@st.cache_data(show_spinner=False)
def optimize_risk(
    ema_df, num_times, risk_range, step_size,
    start_date, end_date, equityStart,
    man_near, man_mid, man_long,
    average_credit, selection_method,
    trend_ranking_days, trend_smoothing_days, trend_smoothing_type
):
    risks = np.arange(risk_range[0], risk_range[1] + step_size, step_size)
    results = []

    for risk in risks:
        daily_equity, _ = calculate_equity_curve_with_dynamic_method(
            ema_df=ema_df,
            start_date=start_date,
            end_date=end_date,
            equityStart=equityStart,
            risk=risk,
            num_times=num_times,
            man_near=man_near,
            man_mid=man_mid,
            man_long=man_long,
            average_credit=average_credit,
            selection_method=selection_method,
            trend_ranking_days=trend_ranking_days,
            trend_smoothing_days=trend_smoothing_days,
            trend_smoothing_type=trend_smoothing_type
        )

        if daily_equity.empty:
            continue

        cagr, mar, sortino = calculate_performance_metrics(daily_equity)
        _, max_drawdown = calculate_drawdown(daily_equity)

        results.append({
            'Risk': risk,
            'CAGR': cagr,
            'MAR': mar,
            'Sortino': sortino,
            'MaxDrawdown': max_drawdown
        })

    if not results:
        return pd.DataFrame(columns=['Risk', 'CAGR', 'MAR', 'Sortino', 'MaxDrawdown'])

    return pd.DataFrame(results).sort_values(by='Risk')

# --- 📈 Tab 3: Risk Optimization
with tab3:
    if selection_method == "Average PCR":
        lookback_str = f"Average PCR Lookbacks: Near {man_near}M / Mid {man_mid}M / Long {man_long}M"
    else:
        lookback_str = f"Trend Ranking: Last {trend_ranking_days} days / {trend_smoothing_days}-day {trend_smoothing_type.upper()}"

    st.subheader(f"Risk Optimization Analysis ({start_date.date()} to {end_date.date()})")
    st.markdown(
        f"##### Target Credit: ${credit_target:.2f} | Entries: {num_times} | {lookback_str}"
    )

    risk_container = st.empty()
    col1, col2 = st.columns(2)  # Reserve layout ahead of time

    if ema_df.empty:
        with risk_container.container():
            st.warning("⚠️ No data available.")
    else:
        with risk_container.container():
            with st.spinner("🔄 Optimizing risk levels..."):
                risk_range = (0.8, 6.0)
                step_size = 0.2

                optimization_results_risk = optimize_risk(
                    ema_df=ema_df,
                    num_times=num_times,
                    risk_range=risk_range,
                    step_size=step_size,
                    start_date=start_date,
                    end_date=end_date,
                    equityStart=equity_start,
                    man_near=man_near,
                    man_mid=man_mid,
                    man_long=man_long,
                    average_credit=average_credit,
                    selection_method=selection_method,
                    trend_ranking_days=trend_ranking_days,
                    trend_smoothing_days=trend_smoothing_days,
                    trend_smoothing_type=trend_smoothing_type
                )

                if optimization_results_risk.empty:
                    st.warning("⚠️ No valid optimization results.")
                else:
                    metrics_to_plot = [
                        ("CAGR", "CAGR", None),
                        ("MAR Ratio", "MAR", "#2ca02c"),
                        ("Sortino Ratio", "Sortino", "#9467bd"),
                        ("Max Drawdown (%)", "MaxDrawdown", "rgba(234,92,81,1)")
                    ]

                    for idx, (metric_title, metric_column, color) in enumerate(metrics_to_plot):
                        fig, is_dark = create_standard_fig(height=400)

                        fig.add_trace(go.Scatter(
                            x=optimization_results_risk['Risk'],
                            y=optimization_results_risk[metric_column],
                            mode='lines+markers',
                            marker=dict(size=6),
                            line=dict(width=2) if color is None else dict(width=2, color=color),
                            name=metric_title,
                        ))

                        fig.add_vline(
                            x=risk,
                            line_dash="dash",
                            line_color="rgba(234,234,234,0.4)",
                            line_width=1,
                            opacity=0.8
                        )

                        if metric_column == "CAGR":
                            yaxis_tickformat = ".0%"
                            hovertemplate = '%{y:.2%}<extra></extra>'
                        else:
                            yaxis_tickformat = None
                            hovertemplate = '%{y:.2f}<extra></extra>'

                        fig.data[0].hovertemplate = hovertemplate

                        fig.update_layout(
                            title=metric_title,
                            xaxis_title="Risk (%)",
                            yaxis_title=metric_title,
                            xaxis=dict(tickmode='linear'),
                            yaxis_tickformat=yaxis_tickformat,
                        )

                        if idx % 2 == 0:
                            with col1:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            with col2:
                                st.plotly_chart(fig, use_container_width=True)


# endregion


# region --- 🎯 Tab 4: Entry Time PCR
# -----------------------------------------------------
def create_time_pcr_table(df, selected_times=None, timezone='US/Central', is_dark=True):

    # --- Dark/Light mode setup
    cmap = matplotlib.colormaps['RdYlGn']
    cell_fill = 'rgba(30,30,30,1)' if is_dark else 'rgba(245,245,245,1)'
    header_fill = 'rgba(50,50,50,1)' if is_dark else 'rgba(230,230,230,1)'
    header_font_color = 'white' if is_dark else 'black'

    # --- Correct Local Time
    local_offset = {
        'US/Eastern': 0,
        'US/Central': -1,
        'US/Mountain': -2,
        'US/Pacific': -3
    }.get(timezone, -1)

    df = df.copy()
    df['LocalTime'] = pd.to_datetime(df['OpenTime'], format='%H:%M', errors='coerce')
    df['LocalTime'] = df['LocalTime'] + pd.to_timedelta(local_offset, unit='h')
    df['LocalTime'] = df['LocalTime'].dt.strftime('%I:%M %p')


    # --- Selected Times Column
    if selected_times is not None:
        df['Selected'] = df['OpenTime'].isin(selected_times).map({True: '✅', False: ''})
    else:
        df['Selected'] = ''

    # --- Insert Gaps for spacing
    gap = [""] * len(df)
    df.insert(4, 'Gap1', gap)
    df.insert(7, 'Gap2', gap)
    df.insert(10, 'Gap3', gap)

    # --- Final Column Order
    columns_order = [
        'LocalTime', 'OpenTime',
        'Far_Premium', 'Far_PCR', 'Gap1',
        'Mid_Premium', 'Mid_PCR', 'Gap2',
        'Near_Premium', 'Near_PCR', 'Gap3',
        'Avg_Premium', 'Avg_PCR', 'Selected'
    ]
    df = df[columns_order]

    # --- Format Premium/PCR Columns
    for col in df.columns:
        if '_Premium' in col:
            df[col] = df[col].map('${:,.2f}'.format)
        elif '_PCR' in col:
            df[col] = df[col].astype(float).round(1).astype(str) + '%'

    # --- Add AVG Row
    avg_row = {}
    for col in df.columns:
        if '_Premium' in col or '_PCR' in col:
            clean_col = df[col].replace(r'[\$,%,]', '', regex=True).astype(float)
            avg_val = clean_col.mean()
            if '_Premium' in col:
                avg_row[col] = f"${avg_val:.2f}"
            elif '_PCR' in col:
                avg_row[col] = f"{avg_val:.1f}%"
        else:
            avg_row[col] = 'AVG' if col == 'LocalTime' else ''

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # --- Background shading
    cell_colors = []
    for col in df.columns:
        if '_Premium' in col or '_PCR' in col:
            clean_col = df[col].replace(r'[\$,%,]', '', regex=True).astype(float)
            if clean_col.max() == clean_col.min():
                norm_col = np.zeros_like(clean_col)
            else:
                norm = mcolors.Normalize(vmin=clean_col.min(), vmax=clean_col.max())
                norm_col = norm(clean_col)

            backgrounds = []
            for idx, val in enumerate(norm_col):
                if idx == len(df) - 1:  # AVG row
                    backgrounds.append(header_fill)
                else:
                    rgba = cmap(val)
                    backgrounds.append(f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.5)')
            cell_colors.append(backgrounds)
        else:
            backgrounds = [header_fill if idx == len(df) - 1 else cell_fill for idx in range(len(df))]
            cell_colors.append(backgrounds)

    # --- Build Table
    fig = go.Figure(data=[go.Table(
        columnwidth=[80, 80, 80, 80, 10, 80, 80, 10, 80, 80, 10, 80, 80, 50],
        header=dict(
            values=[col if not col.startswith('Gap') else '' for col in df.columns],
            fill_color=header_fill,
            align='center',
            font=dict(color=header_font_color, size=14)
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color=cell_colors,
            align='center',
            font=dict(color=header_font_color, size=14),
            height=30
        )
    )])

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)


with tab4:

    st.subheader(f"Entry Time PCR Analysis")

    if ema_df.empty:
        st.warning("⚠️ No data available.")
    else:
        min_date = ema_df['OpenDate'].min().date()
        max_date = ema_df['OpenDate'].max().date()

        col1, col2 = st.columns(2)

        with col1:
            next_trading_day = st.date_input(
                "Next Trading Day to Simulate",
                value=max_date + datetime.timedelta(days=1),
                min_value=min_date,
                max_value=max_date + datetime.timedelta(days=1)
            )

        with col2:
            timezone = st.selectbox(
                "Display Timezone for Local Time",
                options=["US/Eastern", "US/Central", "US/Mountain", "US/Pacific"],
                index=1  # Central default
            )

        # --- ✅ Define lookback calculator and dictionary BEFORE using it
        def calculate_lookback_range(base_date, months_back):
            base = pd.to_datetime(base_date)
            start = base - pd.DateOffset(months=months_back)
            end = base - pd.Timedelta(days=1)
            return start, end

        lookbacks = {
            'Far': man_long,
            'Mid': man_mid,
            'Near': man_near
        }

        # --- 📈 Build Lookback Tables
        lookback_tables = {}

        for label, months in lookbacks.items():
            lb_start, lb_end = calculate_lookback_range(next_trading_day, months)

            mask = (ema_df['OpenDate'] >= lb_start) & (ema_df['OpenDate'] <= lb_end)
            df = ema_df.loc[mask]

            if df.empty:
                st.warning(f"⚠️ No data found for {label} Lookback ({lb_start.date()} to {lb_end.date()})")
                continue

            pcr_table = df.groupby('OpenTimeFormatted').agg({
                'PremiumCapture': 'mean',
                'PCR': 'mean'
            }).reset_index()

            pcr_table.rename(columns={"OpenTimeFormatted": "OpenTime"}, inplace=True)
            lookback_tables[label] = pcr_table

        # --- 📈 Merge Lookbacks into Combined Table
        if lookback_tables:
            combined = lookback_tables['Far'].copy()
            combined = combined.rename(columns={"PremiumCapture": "Far_Premium", "PCR": "Far_PCR"})
            combined = combined.merge(
                lookback_tables['Mid'].rename(columns={"PremiumCapture": "Mid_Premium", "PCR": "Mid_PCR"}),
                on="OpenTime", how="outer")
            combined = combined.merge(
                lookback_tables['Near'].rename(columns={"PremiumCapture": "Near_Premium", "PCR": "Near_PCR"}),
                on="OpenTime", how="outer")

            # --- Calculate Average
            combined['Avg_Premium'] = combined[['Far_Premium', 'Mid_Premium', 'Near_Premium']].mean(axis=1)
            combined['Avg_PCR'] = combined[['Far_PCR', 'Mid_PCR', 'Near_PCR']].mean(axis=1)

            # --- Multiply PCR to %
            for col in ['Far_PCR', 'Mid_PCR', 'Near_PCR', 'Avg_PCR']:
                combined[col] = combined[col] * 100

            # --- Select Top N Times
            top_times = combined.sort_values(by='Avg_PCR', ascending=False).head(num_times)['OpenTime'].tolist()

            # --- Create Final Table
            create_time_pcr_table(combined, selected_times=top_times, timezone=timezone, is_dark=True)

        # --- 📅 Lookback Ranges and Selected Times side-by-side
        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("#### Lookback Date Ranges Used")
            for label, months in lookbacks.items():
                lb_start, lb_end = calculate_lookback_range(next_trading_day, months)
                st.markdown(
                    f"**{label} Lookback:** {months} months — {lb_start.strftime('%m/%d/%Y')} to {lb_end.strftime('%m/%d/%Y')}"
                )

        with right_col:
            st.markdown("#### Selected Times")

            top_times_sorted = sorted(top_times) if top_times else []

            if top_times_sorted:
                # Join all times into a single line, comma separated
                selected_times_str = ', '.join(top_times_sorted)
                st.markdown(f"**{selected_times_str}**")
            else:
                st.markdown("No times selected.")

# endregion


# region --- 📈 Tab 5: Entry Time Trends
# -----------------------------------------------------
# --- 📈 Full Slot Equity Curve Visualization Function
def plot_slot_equity_curves_plotly(
    daily_slot_pnl,
    ema_df,
    ranking_window=120,
    smoothing_window=20,
    smoothing_type='SMA',
    selected_times=None,
    columns=2,
    lookback_end=None,
    highlight_times=None
):
    """
    Plot cumulative PnL and rolling average for each entry time using Plotly.

    Parameters:
    - daily_slot_pnl: DataFrame of daily PnL by OpenDate and OpenTimeFormatted.
    - ema_df: Original full EMA dataset (for WinRate/PCR calculation).
    - ranking_window: Number of most recent trading days to analyze.
    - smoothing_window: Window size for moving average smoothing.
    - smoothing_type: 'SMA' or 'EMA' for smoothing.
    - selected_times: Optional list of times to filter.
    - columns: Number of chart columns in grid.
    - lookback_end: Optional cutoff date (defaults to no cutoff).
    """

    df = daily_slot_pnl.copy().sort_values('OpenDate')

    # --- Filter by lookback end if provided
    if lookback_end is not None:
        df = df[df['OpenDate'] <= lookback_end]

    # --- Focus on last N trading days
    recent_dates = (
        df['OpenDate']
        .drop_duplicates()
        .sort_values()
        .tail(ranking_window)
    )
    filtered = df[df['OpenDate'].isin(recent_dates)]

    # --- Pivot for cumulative PnL
    pivot = filtered.pivot(index='OpenDate', columns='OpenTimeFormatted', values='DailyPnL').fillna(0)

    if selected_times is not None:
        pivot = pivot[selected_times]

    cumulative_pnl = pivot.cumsum()

    if smoothing_type.upper() == "EMA":
        rolling_avg = cumulative_pnl.ewm(span=smoothing_window, min_periods=1).mean()
    else:
        rolling_avg = cumulative_pnl.rolling(window=smoothing_window, min_periods=1).mean()

    # --- Get latest values for cumulative and smoothed lines
    latest_cum = cumulative_pnl.iloc[-1]
    latest_smooth = rolling_avg.iloc[-1]

    # --- Summarize Win Rate and PCR
    summary_stats = ema_df[
        ema_df['OpenDate'].isin(recent_dates)
    ].groupby('OpenTimeFormatted').agg(
        TotalTrades=('PremiumCapture', 'count'),
        Wins=('PremiumCapture', lambda x: (x > 0).sum()),
        TotalPremiumCaptured=('PremiumCapture', 'sum'),
        TotalPremiumSold=('Premium', lambda x: (x * 100).sum())
    ).assign(
        WinRate=lambda df: (df['Wins'] / df['TotalTrades']) * 100,
        PCR=lambda df: (df['TotalPremiumCaptured'] / df['TotalPremiumSold']) * 100
    )

    # --- Grid Layout
    num_slots = len(pivot.columns)
    rows = (num_slots + columns - 1) // columns

    # Build chart titles separately first
    subplot_titles = []
    for slot in pivot.columns:
        check = " ✅" if highlight_times and slot in highlight_times else ""
        win = summary_stats.loc[slot, 'WinRate'] if slot in summary_stats.index else None
        pcr = summary_stats.loc[slot, 'PCR'] if slot in summary_stats.index else None
        cum = latest_cum[slot] * 5 if slot in latest_cum else None
        avg = latest_smooth[slot] * 5 if slot in latest_smooth else None

        title = f"{slot}{check}"

        if win is not None and pcr is not None:
            title += f" | Win: {win:.1f}% | PCR: {pcr:.1f}%"
        if cum is not None and avg is not None:
            title += f" | PnL: ${cum:,.0f} | {smoothing_type.upper()}: ${avg:,.0f}"

        subplot_titles.append(title)

    # Now create subplots using the generated titles
    fig = make_subplots(
        rows=rows, cols=columns,
        subplot_titles=subplot_titles
    )

    # --- Add Cumulative and Rolling lines per slot
    for idx, slot in enumerate(pivot.columns):
        r = (idx // columns) + 1
        c = (idx % columns) + 1

        fig.add_trace(
            go.Scatter(
                x=cumulative_pnl.index, y=cumulative_pnl[slot] * 5,
                mode='lines', name=f'{slot} Cumulative',
                line=dict(width=2, color="#6d6af3"),  # <<<<< Fixed color (adjust if you want)
                hovertemplate='%{x|%Y-%m-%d}<br>$%{y:.0f}<extra></extra>'
            ),
            row=r, col=c
        )

        # --- Rolling Average Line
        fig.add_trace(
            go.Scatter(
                x=rolling_avg.index, y=rolling_avg[slot] * 5,
                mode='lines', name=f'{slot} Rolling',
                line=dict(width=1.5, dash='dash', color='gray'),  # <<<<< Rolling line gray
                hovertemplate='$%{y:.0f}<extra></extra><br>%{x|%Y-%m-%d}'
            ),
            row=r, col=c
        )

    # --- Final Layout Settings
    fig.update_layout(
        height=400 * rows,
        showlegend=False,
        margin=dict(t=50, l=20, r=20, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


with tab5:
    st.subheader("Entry Time Trends")

    # --- 📅 Selection Inputs
    trend_col1, trend_col2 = st.columns(2)

    with trend_col1:
        max_available_date = ema_df['OpenDate'].max().date()
        future_limit = max_available_date + datetime.timedelta(days=7)

        trend_check_date = st.date_input(
            "Select Date to Inspect (Entry Times from that Week)",
            value=min(end_date.date(), future_limit),
            min_value=ema_df['OpenDate'].min().date(),
            max_value=future_limit
        )

    with trend_col2:
        timezone = st.selectbox(
            "Display Timezone",
            options=["US/Eastern", "US/Central", "US/Mountain", "US/Pacific"],
            index=1
        )

    # --- Compute Monday of selected week
    monday = pd.to_datetime(trend_check_date) - pd.Timedelta(days=trend_check_date.weekday())

    # --- Calculate trend date range (rolling N trading days before Monday)
    trading_days = sorted(ema_df['OpenDate'].unique())
    prior_trading_days = [d for d in trading_days if d < monday]
    recent_trading_days = prior_trading_days[-trend_ranking_days:]

    if recent_trading_days:
        trend_start = recent_trading_days[0]
        trend_end = recent_trading_days[-1]
    else:
        trend_start = trend_end = None

    # --- Select times using Time Trend method for this week
    selected_times_trend = select_times_via_time_trends(
        ema_df=ema_df,
        end_date=monday,
        num_times=num_times,
        ranking_window=trend_ranking_days,
        smoothing_window=trend_smoothing_days,
        smoothing_type=trend_smoothing_type
    )

    # --- Local Time conversion
    def convert_to_local(open_times, timezone_str):
        offset = {
            'US/Eastern': 0,
            'US/Central': -1,
            'US/Mountain': -2,
            'US/Pacific': -3
        }.get(timezone_str, -1)

        local_times = []
        for t in open_times:
            dt = pd.to_datetime(t, format='%H:%M', errors='coerce')
            dt_local = (dt + pd.to_timedelta(offset, unit='h')).strftime('%I:%M %p')
            local_times.append(dt_local)
        return local_times

    local_times = convert_to_local(selected_times_trend, timezone)

    # --- Display Trend Range and Times
    if trend_start and trend_end:
        st.markdown(
            f"🧮 **Trend Ranking Range:** {trend_start.strftime('%Y-%m-%d')} to {trend_end.strftime('%Y-%m-%d')} "
            f"({trend_ranking_days} trading days)"
        )

    top_row = ', '.join(selected_times_trend)
    bottom_row = ', '.join(local_times)
    st.markdown(f"⏰ **Selected Times:**\n\n**OpenTime:** {top_row}\n\n**LocalTime:** {bottom_row}")

    # --- Create Daily PnL per Slot
    ema_df['DailyPnL'] = ema_df['PremiumCapture']

    daily_slot_pnl = (
        ema_df
        .groupby(['OpenDate', 'OpenTimeFormatted'], as_index=False)
        .agg({'DailyPnL': 'sum'})
    )

    # --- Plot all entry slots, highlight selected
    plot_slot_equity_curves_plotly(
        daily_slot_pnl=daily_slot_pnl,
        ema_df=ema_df,
        ranking_window=trend_ranking_days,
        smoothing_window=trend_smoothing_days,
        smoothing_type=trend_smoothing_type,
        selected_times=None,
        columns=2,
        lookback_end=monday,  # ✅ Use the dropdown-driven Monday
        highlight_times=selected_times_trend
    )

# endregion
    

# region --- 🔎 Tab 6: Lookback Optimization
# -----------------------------------------------------
# --- 🔎 Tab 6: Lookback Optimization
# -----------------------------------------------------
# --- 📈 Cached heavy calculation
@st.cache_data(show_spinner=False)
def test_lookback_stability_with_overlap_cached(
    ema_df, entry_range, risk, equityStart, near_range, mid_range, long_range, rolling_windows
):
    from itertools import product
    import pandas as pd

    results = []

    # --- ✅ Pre-filter valid lookback combinations
    lookback_combinations = [
        (near, mid, long)
        for near, mid, long in product(near_range, mid_range, long_range)
        if near <= mid <= long
    ]

    for num_times in entry_range:
        for start_date, end_date in rolling_windows:
            for man_near, man_mid, man_long in lookback_combinations:

                daily_equity, _ = calculate_equity_curve_with_manual_lookbacks(
                    ema_df=ema_df,
                    start_date=start_date,
                    end_date=end_date,
                    equityStart=equityStart,
                    risk=risk,
                    num_times=num_times,
                    man_near=man_near,
                    man_mid=man_mid,
                    man_long=man_long,
                    average_credit=average_credit
                )

                if daily_equity.empty:
                    continue

                cagr, _, _ = calculate_performance_metrics(daily_equity)

                results.append({
                    'Near': man_near,
                    'Mid': man_mid,
                    'Long': man_long,
                    'NumEntries': num_times,
                    'CAGR': cagr,
                    'Start': start_date,
                    'End': end_date
                })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return pd.DataFrame(columns=['Near', 'Mid', 'Long', 'Stability Score'])

    # --- 📊 Compute Rankings
    results_df['Rank'] = results_df.groupby(['Start', 'End', 'NumEntries'])['CAGR'].rank(ascending=False, method="dense")

    # --- 📈 Calculate Stability Score
    stability_scores = (
        results_df.groupby(['Near', 'Mid', 'Long'])['Rank']
        .mean()
        .reset_index()
        .rename(columns={'Rank': 'Stability Score'})
    )

    # --- ✅ Round Stability Score for nicer display
    stability_scores['Stability Score'] = stability_scores['Stability Score'].round(2)

    return stability_scores.sort_values(by='Stability Score')


with tab6:
    st.subheader("Lookback Stability Optimization")

    if ema_df.empty:
        st.warning("⚠️ No data available.")
    else:
        st.markdown("##### Stability Test Settings")

        col1, col2 = st.columns(2)

        with col1:
            lastDay_default = ema_df['OpenDate'].max().date()
            lastDay = st.date_input(
                "Select Last Day for Analysis",
                value=lastDay_default,
                min_value=ema_df['OpenDate'].min().date(),
                max_value=lastDay_default
            )
            entry_min, entry_max = st.slider(
                "Select Entry Range (Number of Entries per Day)", 
                min_value=3, max_value=20, 
                value=(8, 13)
            )

        with col2:
            near_min, near_max = st.slider("Near Lookback Range (Months)", 1, 12, (2, 5))
            mid_min, mid_max = st.slider("Mid Lookback Range (Months)", 1, 12, (5, 9))
            long_min, long_max = st.slider("Long Lookback Range (Months)", 1, 12, (9, 12))

        # Button stays above progress/results
        run_button = st.button("🚀 Run Stability Test")

        # Reserve container for both progress and results
        results_container = st.empty()

        if run_button:
            start_time = time.time()

            with results_container.container():
                # Progress bar starts first, inside results container
                progress_bar = st.progress(0, text="🔍 Running stability optimization...")

                # Prep ranges and params
                entry_range = range(entry_min, entry_max + 1)
                near_range = range(near_min, near_max + 1)
                mid_range = range(mid_min, mid_max + 1)
                long_range = range(long_min, long_max + 1)

                lastDay_dt = pd.to_datetime(lastDay)
                rolling_windows = [
                    (lastDay_dt - relativedelta(months=18), lastDay_dt),
                    (lastDay_dt - relativedelta(months=12), lastDay_dt),
                    (lastDay_dt - relativedelta(months=6), lastDay_dt)
                ]

                lookback_combinations = [
                    (near, mid, long)
                    for near, mid, long in product(near_range, mid_range, long_range)
                    if near <= mid <= long
                ]
                total_tests = len(entry_range) * len(rolling_windows) * len(lookback_combinations)
                completed_tests = 0
                results = []

                # Main test loop
                for num_times in entry_range:
                    for start_date, end_date in rolling_windows:
                        for man_near, man_mid, man_long in lookback_combinations:
                            daily_equity, _ = calculate_equity_curve_with_manual_lookbacks(
                                ema_df=ema_df,
                                start_date=start_date,
                                end_date=end_date,
                                equityStart=equity_start,
                                risk=risk,
                                num_times=num_times,
                                man_near=man_near,
                                man_mid=man_mid,
                                man_long=man_long,
                                average_credit=average_credit
                            )

                            if not daily_equity.empty:
                                cagr, _, _ = calculate_performance_metrics(daily_equity)
                                results.append({
                                    'Near': man_near,
                                    'Mid': man_mid,
                                    'Long': man_long,
                                    'NumEntries': num_times,
                                    'CAGR': cagr,
                                    'Start': start_date,
                                    'End': end_date
                                })

                            completed_tests += 1
                            progress_bar.progress(
                                completed_tests / total_tests,
                                text=f"🔍 Running stability optimization... {completed_tests}/{total_tests}"
                            )

                results_df = pd.DataFrame(results)

                if not results_df.empty:
                    results_df['Rank'] = results_df.groupby(['Start', 'End', 'NumEntries'])['CAGR'].rank(
                        ascending=False, method="dense"
                    )

                    stability_df = (
                        results_df.groupby(['Near', 'Mid', 'Long'])['Rank']
                        .mean()
                        .reset_index()
                        .rename(columns={'Rank': 'Stability Score'})
                        .sort_values(by='Stability Score')
                        .round(2)
                    )
                else:
                    stability_df = pd.DataFrame()

                # Finalize
                end_time = time.time()
                minutes, seconds = divmod(int(end_time - start_time), 60)

                # Optional: hide progress bar after complete
                progress_bar.empty()

                if not stability_df.empty:
                    st.success(f"✅ Stability test completed! ({completed_tests:,} combinations tested in {minutes}m {seconds}s)")

                    fig = go.Figure(data=[go.Table(
                        header=dict(
                            values=list(stability_df.head(15).columns),
                            align='center',
                            font=dict(size=14, color='white'),
                            fill_color='rgba(50,50,50,1)',
                            height=30
                        ),
                        cells=dict(
                            values=[stability_df.head(15)[col] for col in stability_df.head(15).columns],
                            align='center',
                            font=dict(size=14),
                            fill_color='rgba(0,0,0,0)',
                            height=28
                        )
                    )])

                    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=520)

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No stable lookbacks found.")



#endregion


# region --- 🔁 Tab 7: Trend Stability Optimization
# -----------------------------------------------------
def run_trend_stability_test_generator(
    ema_df, entry_range, risk, equity_start,
    ranking_windows, smoothing_windows, smoothing_types, rolling_windows
):
    from itertools import product

    results = []
    combinations = list(product(ranking_windows, smoothing_windows, smoothing_types))

    total_tests = len(entry_range) * len(rolling_windows) * len(combinations)
    completed = 0

    for num_times in entry_range:
        for start_date, end_date in rolling_windows:
            for rank_win, smooth_win, smooth_type in combinations:
                daily_equity, _ = calculate_equity_curve_with_dynamic_method(
                    ema_df=ema_df,
                    start_date=start_date,
                    end_date=end_date,
                    equityStart=equity_start,
                    risk=risk,
                    num_times=num_times,
                    man_near=None, man_mid=None, man_long=None,
                    average_credit=average_credit,
                    selection_method="Time Trends",
                    trend_ranking_days=rank_win,
                    trend_smoothing_days=smooth_win,
                    trend_smoothing_type=smooth_type
                )

                if not daily_equity.empty:
                    cagr, _, _ = calculate_performance_metrics(daily_equity)
                    results.append({
                        'RankingWindow': rank_win,
                        'SmoothingWindow': smooth_win,
                        'SmoothingType': smooth_type,
                        'NumEntries': num_times,
                        'Start': start_date,
                        'End': end_date,
                        'CAGR': cagr
                    })

                completed += 1
                yield completed, total_tests, results

with tab7:
    st.subheader("Trend Stability Optimization")

    if ema_df.empty:
        st.warning("⚠️ No data available.")
    else:
        st.markdown("##### Stability Test Settings")

        col1, col2 = st.columns(2)
        with col1:
            lastDay_default = ema_df['OpenDate'].max().date()
            lastDay = st.date_input("Select Last Day for Analysis", value=lastDay_default)
            entry_min, entry_max = st.slider("Number of Entries per Day", 3, 20, (8, 13))
        with col2:
            rank_min, rank_max = st.slider("Ranking Window Range (Days)", 30, 200, (90, 150), step = 10)
            smooth_min, smooth_max = st.slider("Smoothing Window Range", 5, 60, (10, 30))
            smooth_types = st.multiselect("Smoothing Types", options=["SMA", "EMA"], default=["SMA", "EMA"])

        if st.button("🚀 Run Trend Stability Optimization"):
            progress_container = st.empty()
            result_container = st.empty()

            entry_range = range(entry_min, entry_max + 1)
            ranking_windows = range(rank_min, rank_max + 1, 10)
            smoothing_windows = range(smooth_min, smooth_max + 1, 5)

            lastDay_dt = pd.to_datetime(lastDay)
            rolling_windows = [
                (lastDay_dt - relativedelta(months=18), lastDay_dt),
                (lastDay_dt - relativedelta(months=12), lastDay_dt),
                (lastDay_dt - relativedelta(months=6), lastDay_dt)
            ]

            all_results = []
            with progress_container.container():
                progress = st.progress(0, text="🔍 Running trend stability optimization...")
                for completed, total, batch_results in run_trend_stability_test_generator(
                    ema_df, entry_range, risk, equity_start,
                    ranking_windows, smoothing_windows, smooth_types,
                    rolling_windows
                ):
                    all_results = batch_results
                    progress.progress(completed / total, text=f"🔍 {completed:,}/{total:,} combinations tested...")

            df = pd.DataFrame(all_results)

            if df.empty:
                st.warning("⚠️ No stable results found.")
            else:
                df['Rank'] = df.groupby(['Start', 'End', 'NumEntries'])['CAGR'].rank(ascending=False, method="dense")

                summary = (
                    df.groupby(['RankingWindow', 'SmoothingWindow', 'SmoothingType'])['Rank']
                    .mean()
                    .reset_index()
                    .rename(columns={'Rank': 'Stability Score'})
                    .sort_values('Stability Score')
                    .round(2)
                )

                st.success("✅ Stability optimization complete.")
                st.markdown("Top Results:")

                st.dataframe(summary.head(20), use_container_width=True)
# endregion

# region --- 📖 Tab 8: Documenation
# -----------------------------------------------------
# --- 📖 Tab 8: Documentation
# -----------------------------------------------------
with tab8:
    st.subheader("📖 Instructions and Strategy Assumptions")

    try:
        with open("README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()
        st.markdown(readme_content, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("❌ README.md file not found. Please add it to your project directory.")

#endregion

