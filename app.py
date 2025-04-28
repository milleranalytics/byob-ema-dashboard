# -----------------------------------------------------
# --- 📦 Imports
# -----------------------------------------------------
import streamlit as st
import pandas as pd
import datetime
import pytz
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import io
from itertools import product
from dateutil.relativedelta import relativedelta

# Page config
st.set_page_config(page_title="BYOB EMA Dashboard", layout="wide")

# Title
st.title("BYOB 5/40 EMA Backtest Dashboard")
st.markdown("**$2.60 Target Credit, 1.5X Stops**")


# -----------------------------------------------------
# --- 📥 Load CSV & Prepare Data
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
one_year_ago = max_date - datetime.timedelta(days=365)
default_start_date = max(one_year_ago, min_date)  # Prevent going earlier than dataset


# -----------------------------------------------------
# --- ✨ User Input Layout with three columns
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

# --- 💵 Column 2: Risk + Entries
with col2:
    equity_start = st.number_input(
        "Starting Equity ($)",
        value=400_000,
        step=10_000,
        help="Your starting account size in dollars. Used to size trades relative to risk. Higher levels mitigate impacts from jumping up number of contracts as you scale."
    )
    risk = st.number_input(
        "Risk per Day (%)",
        value=4.0,
        step=0.1,
        help="The maximum % of your equity you are willing to risk on a single day. Example: 4% of $400,000 = $16,000 max daily risk.  'Risk' is equal to target credit received for the day assuming -100% PCR is about as bad as it gets, hence 'Risk'."
    )
    num_times = st.slider(
        "Number of Entries",
        min_value=2,
        max_value=20,
        value=10,
        help="Number of entry times selected each day."
    )

# --- 🔎 Column 3: Lookbacks
with col3:
    man_near = st.number_input(
        "Near Lookback (months)",
        value=2,
        step=1,
        min_value=1,
        help="Short-term lookback period in months used to find best entry times. Emphasizes recent market behavior."
    )
    man_mid = st.number_input(
        "Mid Lookback (months)",
        value=5,
        step=1,
        min_value=1,
        help="Medium-term lookback period in months to smooth entry time selection."
    )
    man_long = st.number_input(
        "Long Lookback (months)",
        value=9,
        step=1,
        min_value=1,
        help="Long-term lookback period in months to stabilize entry time selection against outliers."
    )


# -----------------------------------------------------
# --- 🧮 Calculate Initial Variables (based on user input)
# -----------------------------------------------------
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
        st.markdown(f"- **Far Lookback**: `{man_long}` months")

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


def calculate_equity_curve_with_manual_lookbacks(
    ema_df, start_date, end_date, equityStart, risk, num_times, man_near, man_mid, man_long
):
    """
    Calculate the equity curve using fixed manual lookback periods for Near, Mid, and Long terms.
    Ensures historical data is retained for proper lookback calculations.
    """
    if ema_df.empty:
        st.warning("⚠️ Warning: Empty EMA DataFrame passed to equity curve calculation.")
        return pd.DataFrame(), pd.DataFrame()

    ema_df = ema_df.copy()
    ema_df['OpenDate'] = pd.to_datetime(ema_df['OpenDate'])
    ema_df = ema_df.sort_values('OpenDate')

    current_equity = equityStart
    results = []

    # Get periods within selected range
    filtered_periods = ema_df[(ema_df['OpenDate'] >= pd.to_datetime(start_date)) & 
                              (ema_df['OpenDate'] <= pd.to_datetime(end_date))].copy()

    filtered_periods['PeriodStart'] = filtered_periods['OpenDate'].dt.to_period('M').dt.start_time
    unique_periods = filtered_periods['PeriodStart'].unique()

    for current_period in unique_periods:
        near_start = current_period - pd.DateOffset(months=man_near)
        mid_start = current_period - pd.DateOffset(months=man_mid)
        long_start = current_period - pd.DateOffset(months=man_long)
        lookback_end = current_period - pd.Timedelta(days=1)

        # Lookback windows from full dataset
        near_data = ema_df[(ema_df['OpenDate'] >= near_start) & (ema_df['OpenDate'] <= lookback_end)]
        mid_data = ema_df[(ema_df['OpenDate'] >= mid_start) & (ema_df['OpenDate'] <= lookback_end)]
        long_data = ema_df[(ema_df['OpenDate'] >= long_start) & (ema_df['OpenDate'] <= lookback_end)]

        if near_data.empty and mid_data.empty and long_data.empty:
            st.warning(f"⚠️ Skipping {current_period.strftime('%Y-%m')} — no available lookback data.")
            continue

        # Build average PCR from available lookbacks
        lookback_dfs = []
        if not near_data.empty:
            lookback_dfs.append(near_data.groupby('OpenTimeFormatted')['PCR'].mean().rename('Near_PCR'))
        if not mid_data.empty:
            lookback_dfs.append(mid_data.groupby('OpenTimeFormatted')['PCR'].mean().rename('Mid_PCR'))
        if not long_data.empty:
            lookback_dfs.append(long_data.groupby('OpenTimeFormatted')['PCR'].mean().rename('Long_PCR'))

        avg_pcr_df = pd.concat(lookback_dfs, axis=1).mean(axis=1).reset_index()
        avg_pcr_df.columns = ['OpenTimeFormatted', 'PCR']

        # Select top N times
        top_times_sorted = avg_pcr_df.nlargest(num_times, 'PCR') \
                                     .sort_values('OpenTimeFormatted')['OpenTimeFormatted'] \
                                     .tolist()

        current_period_data = ema_df[(ema_df['OpenDate'] >= current_period) & 
                                     (ema_df['OpenDate'] < current_period + pd.DateOffset(months=1))]

        current_period_data = mark_best_times(current_period_data, top_times_sorted)
        current_period_data = current_period_data[current_period_data['BestTime'] == 1]

        if current_period_data.empty:
            continue

        # Daily trade grouping
        for trade_date, day_trades in current_period_data.groupby('OpenDate'):
            average_credit = day_trades['Premium'].mean()
            contracts = int(current_equity * (risk / 100) / num_times / (average_credit * 100)) if average_credit else 0

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

    # Filter to final selected date range
    filtered_results = results_df[(results_df['Date'] >= pd.to_datetime(start_date)) & 
                                  (results_df['Date'] <= pd.to_datetime(end_date))]

    # Aggregate final daily equity
    daily_equity = filtered_results.groupby('Date', as_index=False) \
                                   .agg({'Equity': 'last'}) \
                                   .sort_values(by='Date')

    return daily_equity, filtered_results


# -----------------------------------------------------
# --- 📈 Standard Figure Template to ensure charts are the same
# -----------------------------------------------------
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

# -----------------------------------------------------
# --- 📋 Standard Table Template
# -----------------------------------------------------
def create_standard_table(df, negative_cols=None, decimals=1, theme="auto", height=None):
    """
    Create a standard plotly Table styled to match Streamlit dark/light mode.
    
    Parameters:
    - df (DataFrame): Pandas DataFrame to display
    - negative_cols (List[str], optional): Columns where negative numbers should be red
    - decimals (int): Number of decimals to show
    - theme (str): "auto", "dark", or "light"
    - height (int, optional): Manually set table height if needed
    """

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
                backgrounds.append(header_fill)  # Subtle highlight for AVG row
            else:
                backgrounds.append(cell_fill)

        values.append(formatted)
        font_colors.append(colors)
        fill_colors.append(backgrounds)


    # --- Create Table
    fig = go.Figure(data=[go.Table(
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


# -----------------------------------------------------
# --- 📊 Visualization Tabs
# -----------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Equity Curve",
    "🔎 Entries Optimization",
    "🔎 Risk Optimization",
    "🎯 Entry Time PCR",
    "📈 Entry Time Trends",
    "🔎 Lookback Optimization",
    "📖 Instructions"
])


# -----------------------------------------------------
# --- 📈 Tab 1: Equity Curve + Drawdown
# -----------------------------------------------------

# --- Monthly Performance Table
def display_monthly_performance_table(equity_df, is_dark=True, equity_col='Equity'):
    df = equity_df.copy()
    df['Month'] = df['Date'].dt.to_period('M')
    df['Year'] = df['Date'].dt.year
    df['MonthName'] = df['Date'].dt.month.apply(lambda x: calendar.month_abbr[x])
    df['DailyReturn'] = df[equity_col].pct_change()

    # Monthly returns
    monthly_returns = (
        df.groupby('Month')[equity_col]
        .apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
        .reset_index(name='Return')
    )
    monthly_returns['Year'] = monthly_returns['Month'].dt.year
    monthly_returns['MonthName'] = monthly_returns['Month'].dt.month.apply(lambda x: calendar.month_abbr[x])

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

    # Add TOTAL and MaxDD
    start_of_year = df.groupby(df['Date'].dt.year).first()[equity_col]
    end_of_year = df.groupby(df['Date'].dt.year).last()[equity_col]
    total_returns = ((end_of_year / start_of_year - 1) * 100).round(1)
    pivot['TOTAL'] = total_returns
    pivot['MaxDD'] = dd_df['MaxDD'].round(1)

    # Add AVG row
    avg_row = pivot[month_order + ['TOTAL']].mean(numeric_only=True)
    avg_row['MaxDD'] = None
    pivot.loc['AVG'] = avg_row
    pivot.index.name = None
    pivot.columns.name = None

    # --- 🆕 Create Table using your helper --- 
    fig = create_standard_table(
        df=pivot,
        negative_cols=month_order,
        decimals=1,
        theme="dark" if is_dark else "light",
        height=250  # 🧠 Set this smaller instead of auto height!
    )

    st.plotly_chart(fig, use_container_width=True)

with tab1:
    st.subheader(f"Equity Curve and Drawdown ({start_date.date()} to {end_date.date()})")

    equity_curve, full_trades = calculate_equity_curve_with_manual_lookbacks(
        ema_df=ema_df,
        start_date=start_date,
        end_date=end_date,
        equityStart=equity_start,
        risk=risk,
        num_times=num_times,
        man_near=man_near,
        man_mid=man_mid,
        man_long=man_long
    )

    if equity_curve.empty:
        st.warning("⚠️ No trades generated for the selected period and parameters.")
    else:
        equity_curve_with_dd, max_drawdown = calculate_drawdown(equity_curve)
        pcr = calculate_pcr(full_trades)
        cagr, mar_ratio, sortino = calculate_performance_metrics(equity_curve_with_dd)

        # --- 📈 Top Metrics Display
        metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns([1,1,1,1,1])

        with metrics_col1:
            st.metric(label="CAGR", value=f"{cagr:.2%}")

        with metrics_col2:
            st.metric(label="MAR Ratio", value=f"{mar_ratio:.2f}")

        with metrics_col3:
            st.metric(label="Sortino Ratio", value=f"{sortino:.2f}")

        with metrics_col4:
            st.metric(label="Max Drawdown", value=f"{max_drawdown:.2f}%")

        with metrics_col5:
            st.metric(label="PCR", value=f"{pcr:.2f}%")

        # --- 📈 Create Standard Plot ---
        fig, is_dark = create_standard_fig(rows=2, cols=1, row_heights=[0.6, 0.4], height=800)

        # --- Colors
        streamlit_red = 'rgba(234,92,81,1)'
        streamlit_red_fill = 'rgba(234,92,81,0.15)'

        # --- Equity Curve
        fig.add_trace(go.Scatter(
            x=equity_curve_with_dd['Date'],
            y=equity_curve_with_dd['Equity'],
            mode='lines',
            name='Equity Curve',
            line=dict(width=2),
            hovertemplate='$%{y:,.0f}<extra></extra>'
        ), row=1, col=1)

        # --- Drawdown
        fig.add_trace(go.Scatter(
            x=equity_curve_with_dd['Date'],
            y=equity_curve_with_dd['Drawdown'],
            mode='lines',
            name='Drawdown (%)',
            line=dict(width=2, color=streamlit_red),
            fill='tozeroy',
            fillcolor=streamlit_red_fill,
            hovertemplate='%{y:.2f}%<extra></extra>'
        ), row=2, col=1)

        # --- Axis Titles
        fig.update_yaxes(title_text="Equity ($)", type="log", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        # --- Show
        st.plotly_chart(fig, use_container_width=True)

        # --- 📅 Monthly Performance Table ---
        st.subheader("Monthly Performance Table")

        if equity_curve.empty:
            st.warning("⚠️ No data available to generate monthly performance.")
        else:
            display_monthly_performance_table(equity_curve_with_dd, is_dark=True)

        # --- 📅 Expandable Monthly Summary ---
        with st.expander("📅 Monthly Trading Summary"):
            monthly_summary = full_trades.groupby(full_trades['Date'].dt.to_period('M'))

            for month_period, month_trades in monthly_summary:
                st.markdown(f"#### {month_period.strftime('%Y-%m')}")

                # --- Selected Times
                selected_times = sorted(month_trades['OpenTime'].unique())
                selected_times_str = ', '.join(selected_times) if selected_times else "No trades"

                # --- End Equity (last trade of month)
                end_equity = month_trades['Equity'].iloc[-1]

                # --- Lookback Ranges
                far_start = month_trades['LongLookbackStart'].min()
                far_end = month_trades['LongLookbackEnd'].max()
                mid_start = month_trades['MidLookbackStart'].min()
                mid_end = month_trades['MidLookbackEnd'].max()
                near_start = month_trades['NearLookbackStart'].min()
                near_end = month_trades['NearLookbackEnd'].max()

                # --- Show Details
                st.markdown(f"📅 **Far Lookback:** {man_long} months — {far_start.strftime('%m/%d/%Y')} to {far_end.strftime('%m/%d/%Y')}")
                st.markdown(f"📅 **Mid Lookback:** {man_mid} months — {mid_start.strftime('%m/%d/%Y')} to {mid_end.strftime('%m/%d/%Y')}")
                st.markdown(f"📅 **Near Lookback:** {man_near} months — {near_start.strftime('%m/%d/%Y')} to {near_end.strftime('%m/%d/%Y')}")
                st.markdown(f"⏰ **Selected Times:** {selected_times_str}")
                st.markdown(f"💵 **End Equity:** ${end_equity:,.2f}")



# -----------------------------------------------------
# --- 🎯 Tab 2: Entries Optimization
# -----------------------------------------------------
def optimize_num_entries_with_manual_lookbacks(
    ema_df, num_times_range, equityStart, risk, start_date, end_date, man_near, man_mid, man_long,
    performance_func
):
    """
    Optimize number of entries for manual lookbacks, returning CAGR, MAR, Sortino, and Max Drawdown.
    """

    results = []

    # Streamlit loading spinner
    with st.spinner("🔄 Optimizing number of entries..."):
        for num_times in num_times_range:
            daily_equity_manual, _ = calculate_equity_curve_with_manual_lookbacks(
                ema_df=ema_df,
                start_date=start_date,
                end_date=end_date,
                equityStart=equityStart,
                risk=risk,
                num_times=num_times,
                man_near=man_near,
                man_mid=man_mid,
                man_long=man_long
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
def run_num_entries_sweep(ema_df, start_date, end_date, equity_start, risk, man_near, man_mid, man_long):
    return optimize_num_entries_with_manual_lookbacks(
        ema_df=ema_df,
        num_times_range=range(3, 21),
        equityStart=equity_start,
        risk=risk,
        start_date=start_date,
        end_date=end_date,
        man_near=man_near,
        man_mid=man_mid,
        man_long=man_long,
        performance_func=calculate_performance_metrics
    )


# --- 🎯 Tab 2: Entries Optimization
with tab2:
    st.subheader(f"Entries Optimization Analysis ({start_date.date()} to {end_date.date()})")

    num_times_range = range(3, 21)

    if ema_df.empty:
        st.warning("⚠️ No data available.")
    else:
        # ✅ Cached sweep function
        optimization_results = run_num_entries_sweep(
            ema_df=ema_df,
            start_date=start_date,
            end_date=end_date,
            equity_start=equity_start,
            risk=risk,
            man_near=man_near,
            man_mid=man_mid,
            man_long=man_long
        )

        if optimization_results.empty:
            st.warning("⚠️ No valid optimization results.")
        else:
            # --- 📈 Split into two columns
            col1, col2 = st.columns(2)

            # --- Metrics to plot
            metrics_to_plot = [
                ("CAGR", "CAGR", None),
                ("MAR Ratio", "MAR", "#2ca02c"),
                ("Sortino Ratio", "Sortino", "#9467bd"),
                ("Max Drawdown (%)", "MaxDrawdown", "rgba(234,92,81,1)")
            ]

            for idx, (metric_title, metric_column, color) in enumerate(metrics_to_plot):
                # 🆕 Use standard figure template
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

                # 🧠 Only format CAGR y-axis as %  
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

                # --- 📈 Alternate between columns
                if idx % 2 == 0:
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    with col2:
                        st.plotly_chart(fig, use_container_width=True)



# -----------------------------------------------------
# --- 📈 Tab 3: Risk Optimization
# -----------------------------------------------------

# --- 📈 Helper: Optimize Risk (with caching)
@st.cache_data(show_spinner=True)
def optimize_risk_for_manual_lookbacks_cached(
    ema_df, num_times, risk_range, step_size, start_date, end_date, equityStart, man_near, man_mid, man_long
):
    risks = np.arange(risk_range[0], risk_range[1] + step_size, step_size)
    results = []

    for risk in risks:
        daily_equity, _ = calculate_equity_curve_with_manual_lookbacks(
            ema_df=ema_df,
            start_date=start_date,
            end_date=end_date,
            equityStart=equityStart,
            risk=risk,
            num_times=num_times,
            man_near=man_near,
            man_mid=man_mid,
            man_long=man_long
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
    st.subheader(f"Risk Optimization Analysis ({start_date.date()} to {end_date.date()})")

    if ema_df.empty:
        st.warning("⚠️ No data available.")
    else:
        risk_range = (0.8, 6.0)  # ✅ Default risk range (0.8% to 6%)
        step_size = 0.2

        optimization_results_risk = optimize_risk_for_manual_lookbacks_cached(
            ema_df=ema_df,
            num_times=num_times,
            risk_range=risk_range,
            step_size=step_size,
            start_date=start_date,
            end_date=end_date,
            equityStart=equity_start,
            man_near=man_near,
            man_mid=man_mid,
            man_long=man_long
        )

        if optimization_results_risk.empty:
            st.warning("⚠️ No valid optimization results.")
        else:
            # --- 📈 Create two columns for the plots
            col1, col2 = st.columns(2)

            # --- 📈 Metrics to Plot
            metrics_to_plot = [
                ("CAGR", "CAGR", None),
                ("MAR Ratio", "MAR", "#2ca02c"),
                ("Sortino Ratio", "Sortino", "#9467bd"),
                ("Max Drawdown (%)", "MaxDrawdown", "rgba(234,92,81,1)")
            ]

            for idx, (metric_title, metric_column, color) in enumerate(metrics_to_plot):
                # 🆕 Use standard figure template
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

                # 🧠 Only format CAGR y-axis as %  
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

                # --- 📈 Alternate between columns
                if idx % 2 == 0:
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    with col2:
                        st.plotly_chart(fig, use_container_width=True)



# -----------------------------------------------------
# --- 🎯 Tab 4: Entry Time PCR
# -----------------------------------------------------
def create_time_pcr_table(df, selected_times=None, timezone='US/Central', is_dark=True):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # --- Dark/Light mode setup
    cmap = cm.get_cmap('RdYlGn')
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

    df['LocalTime'] = pd.to_datetime(df['OpenTime'], format='%H:%M') + pd.to_timedelta(local_offset, unit='h')
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
            clean_col = df[col].replace('[\$,%,]', '', regex=True).astype(float)
            avg_val = clean_col.mean()
            if '_Premium' in col:
                avg_row[col] = f"${avg_val:.2f}"
            elif '_PCR' in col:
                avg_row[col] = f"{avg_val:.2f}%"
        else:
            avg_row[col] = 'AVG' if col == 'LocalTime' else ''

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # --- Background shading
    cell_colors = []
    for col in df.columns:
        if '_Premium' in col or '_PCR' in col:
            clean_col = df[col].replace('[\$,%,]', '', regex=True).astype(float)
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

    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=800)
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
                max_value=max_date + datetime.timedelta(days=5)
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



# -----------------------------------------------------
# --- 📈 Tab 5: Entry Time Trends
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
    lookback_end=None
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

    fig = make_subplots(
        rows=rows, cols=columns,
        subplot_titles=[
            f"{slot} | Win: {summary_stats.loc[slot, 'WinRate']:.1f}% | PCR: {summary_stats.loc[slot, 'PCR']:.1f}%"
            if slot in summary_stats.index else f"{slot}"
            for slot in pivot.columns
        ]
    )

    # --- Add Cumulative and Rolling lines per slot
    for idx, slot in enumerate(pivot.columns):
        r = (idx // columns) + 1
        c = (idx % columns) + 1

        fig.add_trace(
            go.Scatter(
                x=cumulative_pnl.index, y=cumulative_pnl[slot] * 100,
                mode='lines', name=f'{slot} Cumulative',
                line=dict(width=2, color="#6d6af3"),  # <<<<< Fixed color (adjust if you want)
                hovertemplate='%{y:.2f}%<extra></extra>'
            ),
            row=r, col=c
        )

        # --- Rolling Average Line
        fig.add_trace(
            go.Scatter(
                x=rolling_avg.index, y=rolling_avg[slot] * 100,
                mode='lines', name=f'{slot} Rolling',
                line=dict(width=1.5, dash='dash', color='gray'),  # <<<<< Rolling line gray
                hovertemplate='%{y:.2f}%<extra></extra>'
            ),
            row=r, col=c
        )

    # --- Final Layout Settings
    fig.update_layout(
        height=400 * rows,
        title_text="Slot Equity Trends by Time",
        showlegend=False,
        margin=dict(t=50, l=20, r=20, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


with tab5:
    st.subheader("Entry Time Trends")
    # --- 📈 User Controls for Slot Equity Curves
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        ranking_window = st.slider("Analysis Lookback Days", min_value=30, max_value=365, value=120)

    with col2:
        smoothing_window = st.slider("Smoothing Window (Days)", min_value=5, max_value=60, value=20)

    with col3:
        smoothing_type = st.selectbox("Smoothing Type", options=["SMA", "EMA"], index=0)

    # --- 📈 Create Daily PnL per Slot
    ema_df['DailyPnL'] = ema_df['PremiumCapture']  # Already per-contract

    daily_slot_pnl = (
        ema_df
        .groupby(['OpenDate', 'OpenTimeFormatted'], as_index=False)
        .agg({'DailyPnL': 'sum'})
    )

    plot_slot_equity_curves_plotly(
        daily_slot_pnl=daily_slot_pnl,
        ema_df=ema_df,
        ranking_window=ranking_window,
        smoothing_window=smoothing_window,
        smoothing_type=smoothing_type,
        selected_times=None,
        columns=2,
        lookback_end=pd.to_datetime(end_date)
    )
    

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
                    man_long=man_long
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

        # --- 📋 Create two columns
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
                value=(9, 11)
            )

        with col2:
            near_min, near_max = st.slider(
                "Near Lookback Range (Months)", 
                min_value=1, max_value=6, 
                value=(2, 5)
            )

            mid_min, mid_max = st.slider(
                "Mid Lookback Range (Months)", 
                min_value=3, max_value=10, 
                value=(5, 9)
            )

            long_min, long_max = st.slider(
                "Long Lookback Range (Months)", 
                min_value=6, max_value=12, 
                value=(9, 12)
            )

        st.markdown("")

        # --- Button to run
        if st.button("🚀 Run Stability Test"):
            with st.spinner("🔎 Running stability optimization... this might take a while!"):
                
                # --- Build ranges from user input
                entry_range = range(entry_min, entry_max + 1)
                near_range = range(near_min, near_max + 1)
                mid_range = range(mid_min, mid_max + 1)
                long_range = range(long_min, long_max + 1)

                # --- Prepare rolling windows
                lastDay_dt = pd.to_datetime(lastDay)

                rolling_windows = [
                    (lastDay_dt - relativedelta(years=2), lastDay_dt),
                    (lastDay_dt - relativedelta(months=18), lastDay_dt - relativedelta(months=6)),
                    (lastDay_dt - relativedelta(years=1), lastDay_dt),
                    (lastDay_dt - relativedelta(months=6), lastDay_dt)
                ]

                # --- Run optimization
                stability_df = test_lookback_stability_with_overlap_cached(
                    ema_df=ema_df,
                    entry_range=entry_range,
                    risk=risk,
                    equityStart=equity_start,
                    near_range=near_range,
                    mid_range=mid_range,
                    long_range=long_range,
                    rolling_windows=[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')) for start, end in rolling_windows]
                )

                if stability_df.empty:
                    st.warning("⚠️ No stable lookbacks found.")
                else:
                    num_tests = len(stability_df)
                    st.success(f"✅ Stability test completed! ({num_tests:,} combinations tested)")

                    # --- Create Plotly Table
                    fig = go.Figure(data=[go.Table(
                        header=dict(
                            values=list(stability_df.head(10).columns),
                            align='center',
                            font=dict(size=14, color='white'),
                            fill_color='rgba(50,50,50,1)',
                            height=30
                        ),
                        cells=dict(
                            values=[stability_df.head(10)[col] for col in stability_df.head(10).columns],
                            align='center',
                            font=dict(size=14),
                            fill_color='rgba(0,0,0,0)',
                            height=28
                        )
                    )])

                    fig.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=400
                    )

                    # --- Create three columns: empty, table, empty
                    col1, col2, col3 = st.columns([1, 2, 1])

                    with col2:
                        st.plotly_chart(fig, use_container_width=True)  # ✅ Middle column only


# -----------------------------------------------------
# --- 📖 Tab 7: Documentation
# -----------------------------------------------------
with tab7:
    st.subheader("📖 Instructions and Strategy Assumptions")

    try:
        with open("README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()
        st.markdown(readme_content, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("❌ README.md file not found. Please add it to your project directory.")

