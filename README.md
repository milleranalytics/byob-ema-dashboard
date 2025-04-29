# 📈 BYOB EMA Strategy Dashboard

Welcome to the BYOB EMA Trading Dashboard — a private research tool designed to model and optimize a systematic daily options trading strategy.

---

## 🚀 How to Use This Dashboard

1. **Select Start and End Dates**
   - These dates define the backtest window.
   - Only trades within this range will be included.

2. **Configure Starting Equity and Risk per Day**
   - *Starting Equity*: The initial balance used in the simulation.  Using a higher number here will mute the impacts of scaling as jumping up contracts won't have as much of a difference.
   - *Risk %*: The percentage of starting equity risked daily, divided across the number of entries.  Risk is also equal to the total credit target per day under the assumption that a -100% PCR is about as bad as a day gets and is therefore your total risk for the day.

3. **Set Number of Entries per Day**
   - Choose how many entry times to target each day based.

4. **Define Lookback Periods**
   - *Near*, *Mid*, and *Long* lookback windows (in months) are used to rank the best entry times.
   - These should ideally be selected based on stability testing (see Optimization tabs).

5. **Explore Tabs for Analysis**
   - **Tab 1: Equity Curve and Drawdown** — View the main performance metrics.
   - **Tab 2: Entries Optimization** — Explore how performance varies by number of daily entries.
   - **Tab 3: Risk Optimization** — Explore how performance varies by daily risk percentage.
   - **Tab 4: Entry Time PCR Analysis** — Audit best entry times for the next trading day based on current lookbacks.
   - **Tab 5: Entry Time Trends** — Track rolling equity trends of individual entry times for discretionary overlay.
   - **Tab 6: Lookback Stability Optimization** — Re-optimize best lookback periods each month to maintain robustness.

---

## 📜 Strategy Philosophy

The BYOB EMA Dashboard is designed to help guide systematic monthly trading decisions without relying on lookahead bias or overfitting. Rather than optimizing for the highest possible backtest result, the workflow emphasizes *stability* across different entry counts, lookback periods, and time windows. Credit targets are reviewed monthly to stay responsive to changing market dynamics. Entry times are carefully balanced between diversification and maintaining an edge. Lookbacks are selected through rigorous multi-window stability testing. The goal is to create a durable, walk-forward research framework that favors robust consistency over fragile perfection — building confidence that results will generalize into live trading conditions.

---

## 📊 Key Assumptions

- ✅ **No Lookahead Bias**  
  Each month's best times are selected using only historical data.

- ✅ **Realistic Position Sizing**  
  Contracts are calculated daily based on available risk and trade credit.

- ✅ **Commissions and Slippage Included**  
  Premium capture rates reflect all typical trading costs.

- ✅ **Daily Drawdown Measured**  
  Max drawdowns are calculated daily, not just at month-end.

---

## ⚙️ Important Operational Notes

- **Updating Data**  
  - Data comes from [Trade Automation Toolbox](https://tradeautomationtoolbox.com/byob-ticks/?save=GkxAZ8D).
  - Review results for all credit targets on all time slots on a 1-year and 3-month lookback window and pick the credit target that is holding up the best.
  - Overwrite the existing `EMA.csv` with the latest monthly data using the new credit target (if it changes) for all available time slots.
  - Re-run the app to reflect the updated data.

- **Lookback Stability Testing**  
  - At the beginning of each month, the system re-optimizes the **lookback periods** used for entry timing.
  - **Rolling windows include:**
    - Last **18 months**
    - Last **12 months**
    - Last **6 months**
  -This ensures:
    - Recent market behavior is **always included** in the analysis.
    - The system captures **both long-term and short-term** stability factors.
    - **Older, outdated market regimes** (>18 months ago) have **less influence**.
  - Why this matters:
    - Markets change — volatility cycles, ideal credit targets, and sentiment regimes evolve. 0DTE trading is relatively new and the players are likely adapting to this new instrument.
    - By **emphasizing recency** while still **respecting broader trends**, the strategy adapts in a rolling-window fashion *without overfitting*.
    - This approach maintains a careful balance between **robustness** and **nimbleness** in real-world trading.

- **Hosting**  
  - This app is designed for private local use.
  - Public hosting is possible (e.g., Streamlit Cloud) but not recommended for sensitive research tools.

---

## ⚠️ Risk Disclaimer

🚨 **This dashboard is for research and educational purposes only. Trading involves significant risk and may not be suitable for all investors. Past performance is not necessarily indicative of future results. Always do your own due diligence.**

