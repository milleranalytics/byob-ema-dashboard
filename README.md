Welcome to the BYOB EMA Trading Dashboard — a private research tool designed to model and optimize a systematic daily options trading strategy.

---

### 🚀 How to Use This Dashboard

1. **Select Start and End Dates**
   - These dates define the backtest window.
   - Only trades within this range will be included.

2. **Configure Starting Equity and Risk per Day**
   - *Starting Equity*: The initial balance used in the simulation.  Using a higher number here will mute the impacts of scaling as jumping up contracts won't have as much of a difference.
   - *Risk %*: The percentage of starting equity risked daily, divided across the number of entries.  Risk is also equal to the total credit target per day under the assumption that a -100% PCR is about as bad as a day gets and is therefore your total risk for the day.

3. **Set Number of Entries per Day**
   - Choose how many entry times to enter each day.  The total number of contracts per tranche will be calculated based on your risk and the number of entries per day so that your total daily credit target remains consistent. 

4. **Choose Entry Time Selection Method**
   - Two models are supported:
      - *Time Trends (default)*: Uses a trend-following filter on individual entry times, ranking them by cumulative PnL and filtering below a moving average.  New times are selected each Monday. This allows the system to adapt to weekly shifting markets.
      - *Average PCR*: Ranks times based on average Premium Capture Rate across selected lookback windows (Near, Mid, Long) in months.
   - The Equity Curve, Entries Optimization and Risk Optimization Tabs reflect the selected entry method.

5. **Explore Tabs for Analysis**
   - **Tab 1: Equity Curve and Drawdown** — View an equity curve, the main performance metrics, as well as a table of monthly performance history for the variables that were chosen above.  This tab also has some confirmation checks so you can see inside the backtest data to confirm the entry times chosen in the backtest match with what you expect them to be.
   - **Tab 2: Entries Optimization** — Explore how performance metrics vary by number of daily entries.
   - **Tab 3: Risk Optimization** — Explore how performance metrics vary by daily risk percentage.
   - **Tab 4: Entry Time Trends** — Track rolling equity trends of individual entry times for discretionary overlay. Audit the chosen trading times for a given trading day when using the Time Trend entry method. Use this tab before trading each Monday to select your times for the week.
   - **Tab 5: Entry Time Avg PCR** — Audit best entry times for the next trading day based on current lookbacks for the Average PCR entry method. Run this on the 1st of each month and leave static for the entire month to match the way the equity curve is calculated. Going up and down number of entries mid-month is expected as account grows and shrinks.  You can manually choose the date to run this analysis as well, which allows you to double check the bascktest historical logic.
   - **Tab 6: Time Trend Optimization** — Re-optimize for the most stable lookback days and moving average smoothing periods.
   - **Tab 7: Lookback Stability Optimization** — Re-optimize for the most stable lookback periods each month to maintain robustness. While this level of optimization is great for a rolling-window approach in future trading, this part of the strategy does allow some lookahead bias to creep into the backtest results.  The hope is that by testing over a wide variety of lookbacks and number of entries, it won't be hyper-optimized for the best results but find something that has been stable and will continue to be stable.

---

### 📜 Strategy Philosophy

The BYOB EMA Dashboard is designed to help guide systematic monthly trading decisions without relying on lookahead bias or overfitting. Rather than optimizing for the highest possible backtest result, the workflow emphasizes *stability* across different entry counts, lookback periods, and time windows. Credit targets are reviewed monthly to stay responsive to changing market dynamics. Entry times are carefully balanced between diversification and maintaining an edge. Lookbacks and variables are selected through rigorous multi-window stability testing. The goal is to create a durable, walk-forward research framework that favors robust consistency over fragile perfection — building confidence that results will generalize into live trading conditions.

---

### 📊 Key Assumptions

- ✅ **No Lookahead Bias in Time Selection**  
  Each month's best times are selected using only historical data.

- ✅ **Realistic Position Sizing**  
  Contracts per tranche are calculated daily based on available risk and trade credit.

- ✅ **Commissions and Slippage Included**  
  Premium capture rates reflect all typical trading costs.

- ✅ **Daily Drawdown Measured**  
  Max drawdowns are calculated daily, not just at month-end.

---

### ⚙️ Important Operational Notes

- **Updating Data**  
  - Data comes from [Trade Automation Toolbox](https://tradeautomationtoolbox.com/byob-ticks/?save=GkxAZ8D).
  - At the start of each month, review results for all credit targets on all time slots on a 1-year and 3-month lookback window and pick the credit target that is holding up the best.
  - Overwrite the existing `EMA.csv` with the latest monthly data using the new credit target (if it changes) for all available time slots except 15:30 & 15:45 and a single contract from 2022 to current.
  - Re-run the app to reflect the updated data.

- **Lookback Stability Testing**  
  - At the beginning of each month, the re-optimize the **lookback periods** or **Time Trend Variables** used for entry timing.
  - **Rolling windows include:**
    - Last **18 months**
    - Last **12 months**
    - Last **6 months**
    - This ensures:
      - Recent market behavior is **always included** in the analysis to capture the changing market dynamics (best for walk-forward).
      - The system captures **both long-term and short-term** stability factors.
      - **Older, outdated market regimes** (>18 months ago) have **less influence**.
  - The lookback stability testing also is over a wide range of number of entries per day.  This ensures that we are stable over several entries and not optimizing on a single number of entries.
  - Why this matters:
    - Markets change — volatility cycles, ideal credit targets, and sentiment regimes evolve. 0DTE trading is relatively new and the players are likely adapting to this new instrument.
    - By **emphasizing recency** while still **respecting broader trends**, the strategy adapts in a rolling-window fashion *without overfitting*.
    - This approach maintains a careful balance between **robustness** and **nimbleness** in real-world trading.

---

### ⚠️ Risk Disclaimer

🚨 **This dashboard is for research and educational purposes only. Trading involves significant risk and may not be suitable for all investors. Past performance is not necessarily indicative of future results. Always do your own due diligence.**

