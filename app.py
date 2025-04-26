# -----------------------------------------------------
# --- 📦 Imports
# -----------------------------------------------------
import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import datetime
import io

# Page config
st.set_page_config(page_title="BYOB EMA Dashboard", layout="wide")

# Title
st.title("BYOB EMA Dashboard")
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

# --- 🎨 Helper Function: Display Matplotlib High-DPI in Streamlit

def display_matplotlib_highdpi(fig, dpi=300):
    """
    Render a Matplotlib figure in Streamlit at high DPI with no auto-resizing.

    Parameters:
        fig (matplotlib.figure.Figure): The Matplotlib figure to display.
        dpi (int): Dots per inch for rendering resolution (default 300).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    
    st.image(buf, use_column_width=False)  # 🚨 critical: stop Streamlit from resizing automatically!


# -----------------------------------------------------
# --- 📊 Visualization Tabs
# -----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Equity Curve", "📅 Monthly Performance", "🎯 Entries Optimization", "📖 Instructions"])


# -----------------------------------------------------
# --- 📈 Tab 1: Equity Curve + Drawdown
# -----------------------------------------------------

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

        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

        with metric_col1:
            st.metric(label="CAGR", value=f"{cagr:.2%}")
        with metric_col2:
            st.metric(label="MAR Ratio", value=f"{mar_ratio:.2f}")
        with metric_col3:
            st.metric(label="Sortino Ratio", value=f"{sortino:.2f}")
        with metric_col4:
            st.metric(label="Max Drawdown", value=f"{max_drawdown:.2f}%")
        with metric_col5:
            st.metric(label="PCR", value=f"{pcr:.2f}%")

        # --- 📈 Build Subplots ---
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Equity Curve (Log Scale)", "Drawdown (%)")
        )

        # --- Equity Curve ---
        fig.add_trace(go.Scatter(
            x=equity_curve_with_dd['Date'],
            y=equity_curve_with_dd['Equity'],
            mode='lines',
            name='Equity Curve',
            line=dict(width=2),
            hovertemplate='$%{y:,.0f}<extra></extra>'
        ), row=1, col=1)

        # --- Drawdown (Area Fill) ---
        fig.add_trace(go.Scatter(
            x=equity_curve_with_dd['Date'],
            y=equity_curve_with_dd['Drawdown'],
            mode='lines',
            name='Drawdown (%)',
            line=dict(width=2, color='rgba(234,92,81)'),
            fill='tozeroy',  # ✅ Fills area down to y=0
            fillcolor='rgba(234,92,81,0.12)',  # ✅ Red fill with 15% opacity
            hovertemplate='%{y:.2f}%<extra></extra>'
        ), row=2, col=1)

        # --- Update Layout ---
        fig.update_layout(
            height=800,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50),
        )

        fig.update_yaxes(type="log", title="Equity ($)", row=1, col=1)
        fig.update_yaxes(title="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title="Date", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # --- 📅 Expandable Monthly Summary ---
        with st.expander("📅 Monthly Trading Summary"):
            monthly_summary = full_trades.groupby(full_trades['Date'].dt.to_period('M')).first()[[
                'NearLookback', 'MidLookback', 'LongLookback', 'Equity'
            ]].reset_index()

            for _, row in monthly_summary.iterrows():
                month_period = row['Date']

                # Filter trades for this month
                month_trades = full_trades[full_trades['Date'].dt.to_period('M') == month_period]

                # Get unique trading times used
                selected_times = sorted(month_trades['OpenTime'].unique())

                # Format times for display
                selected_times_str = ', '.join(selected_times) if selected_times else "No trades"

                st.markdown(
                    f"**{month_period.strftime('%Y-%m')}**  \n"
                    f"🔍 Lookbacks: Near={row['NearLookback']}M, Mid={row['MidLookback']}M, Long={row['LongLookback']}M  \n"
                    f"💰 End Equity: `${row['Equity']:,.2f}`  \n"
                    f"⏰ Selected Times: {selected_times_str}"
                )
                # st.markdown("---")  # Optional separator line


# -----------------------------------------------------
# --- 📅 Tab 2: Monthly Performance
# -----------------------------------------------------

def display_monthly_performance_table(equity_df, equity_col='Equity'):
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

    # Use toggle from sidebar
    from streamlit import session_state
    is_dark = session_state.get("theme_override", True) if "theme_override" in session_state else True


    # Theme colors
    header_fill = '#262730' if is_dark else '#f7f7f7'
    header_font = 'white' if is_dark else 'black'
    cell_fill = '#0e1117' if is_dark else 'white'
    default_font = 'white' if is_dark else 'black'
    neg_font = 'rgba(234,92,81,1)'

    # Build values matrix and font color list
    values = [list(pivot.index)]
    font_colors = [ [default_font] * len(pivot.index) ]  # For index column

    for col in pivot.columns:
        col_vals = pivot[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "-").tolist()
        values.append(col_vals)

        # Handle font coloring
        col_colors = []
        for idx, val in pivot[col].items():
            is_negative = (
                col in month_order and
                isinstance(val, (int, float)) and
                pd.notnull(val) and val < 0
            )
            col_colors.append(neg_font if is_negative else default_font)
        font_colors.append(col_colors)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Year"] + list(pivot.columns),
            fill_color=header_fill,
            font_color=header_font,
            align='center'
        ),
        cells=dict(
            values=values,
            fill_color=cell_fill,
            font_color=font_colors,
            align='center',
            font_size=12,
            height=30
        )
    )])

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=400 + 30 * len(pivot)
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Monthly Performance Table")
    
    if equity_curve.empty:
        st.warning("⚠️ No data available to generate monthly performance.")
    else:
        display_monthly_performance_table(equity_curve_with_dd)

# -----------------------------------------------------
# --- 🎯 Tab 3: Entries Optimization
# -----------------------------------------------------
with tab3:
    st.subheader("Entries Optimization")
    st.info("Chart coming soon...")  # (Placeholder for now)


# -----------------------------------------------------
# --- 📖 Tab 4: Documentation
# -----------------------------------------------------
with tab4:
    st.subheader("📖 Instructions and Strategy Assumptions")

    try:
        with open("README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()

        st.markdown(readme_content, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("❌ README.md file not found. Please add it to your project directory.")
