# 📈 BYOB EMA Strategy Dashboard

Welcome to the BYOB EMA Trading Dashboard — a private research tool designed to model and optimize a systematic daily options trading strategy.

---

## 🚀 How to Use This Dashboard

1. **Select Start and End Dates**
   - These dates define the backtest window.
   - Only trades within this range will be included.

2. **Configure Starting Equity and Risk per Day**
   - *Starting Equity*: The initial balance used in the simulation.
   - *Risk %*: The percentage of starting equity risked daily, divided across the number of entries.

3. **Set Number of Entries per Day**
   - Choose how many entry times to target each day based on historical performance.

4. **Define Lookback Periods**
   - *Near*, *Mid*, and *Long* lookback windows (in months) are used to rank the best entry times.
   - These should ideally be selected based on stability testing (see Optimization tabs).

5. **Explore Tabs for Analysis**
   - **Tab 1: Equity Curve and Drawdown** — View the main performance metrics.
   - **Tab 2: Entries Optimization** — Explore how performance varies by number of daily entries.
   - **Tab 3: Risk Optimization** — Explore how performance varies by daily risk percentage.
   - **Tab 4: Entry Time PCR Analysis** — Audit best entry times for the next trading day based on current lookbacks.
   - **Tab 5: Entry Time Trends** — Track rolling equity trends of individual entry times for discretionary overlay.
   - **Tab 6: Lookback Stability Optimization** — (Optional) Re-optimize best lookback periods each month to maintain robustness.

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
  - Overwrite the existing `EMA.csv` with the latest monthly data.
  - Data comes from [Trade Automation Toolbox](https://tradeautomationtoolbox.com/byob-ticks/?save=GkxAZ8D)
  - Re-run the app to reflect the updated data.

- **Stability Testing**  
  - Only re-run the Lookback Stability tab at the start of a new month.
  - Plug the new stable lookbacks into your main inputs afterward.

- **Hosting**  
  - This app is designed for private local use.
  - Public hosting is possible (e.g., Streamlit Cloud) but not recommended for sensitive research tools.

---

## ⚠️ Risk Disclaimer

🚨 **This dashboard is for research and educational purposes only. Trading involves significant risk and may not be suitable for all investors. Past performance is not necessarily indicative of future results. Always do your own due diligence.**

