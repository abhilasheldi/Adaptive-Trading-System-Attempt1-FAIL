# **MASTER CONTEXT** 

You are helping build a systematic crypto trading research system. Before writing any code, internalize these constraints. They are not suggestions — violating them makes the output unusable regardless of how well it otherwise works. 

# **Design philosophy** 

The central metric is Expected Value per trade, not win rate or accuracy. Priority order: capital preservation > risk-adjusted returns > robustness across regimes > adaptability (detecting when the system stops working) > raw prediction accuracy. 

# **The single edge hypothesis being tested** 

Post-compression liquidity-sweep reversals in BTC and ETH on the 1H–4H timeframe, during risk-on macro regimes, with multi-timeframe structural alignment, carry positive expected value after costs, with minimum Sharpe 1.0 over a minimum 200-trade out-ofsample sample. Everything in this system exists to test or protect this hypothesis. Nothing else. 

# **Non-negotiable global constraints** 

1. **Temporal partitioning is physical, not conventional.** Training, validation, and test data live in separate, immutable partitions. Code in one partition must not access data from a later partition, even accidentally. Every feature must be calculable using only data available at that exact timestamp — no forwardlooking windows, ever. 

2. **No lookahead bias, anywhere.** Every rolling/percentile calculation uses only closed candles up to and including the current timestamp. This is the single most important rule in the whole project. 

3. **Raw and normalized data are stored separately.** Raw is never overwritten. Normalized data is always reproducible from raw. 

4. **Walk-forward validation only.** Train → validate → test, rolling forward, never overlapping. Hyperparameters and ensemble weights are tuned on the validation fold only, never the training fold, never the test fold. 

5. **Ensemble Meta-Model is linear only.** A weighted average with learned weights. No non-linear stacking, no secondary model at this layer. 

6. **Kelly sizing is hard-capped at 0.25 (quarter-Kelly),** and is overridden unconditionally by: max 2% single-trade risk, max 6% correlated-direction risk, max 10% aggregate open risk. These three numbers are enforced at the execution layer independent of any model output. 

7. **EV threshold uses the lower confidence-interval bound, not the point estimate.** Minimum: EV lower CI bound > +0.15% per unit risk. 

8. **Six trade-entry gates must all pass simultaneously:** EV lower CI bound > +0.15%; MTF alignment ≥ 3/4; macro regime is Risk-On or Transitional; volatility regime is not Spike; analog sample size ≥ 50 independent instances; risk score ≤ 65/100. 

9. **Minimum sample gate for the k-NN analog engine: N ≥ 50 independent instances** , with overlapping-window instances from the same directional move counted as one, not many. 

# **Two separate gate systems — do not conflate these** 

There are **two distinct threshold sets** in this spec, used at different points. Mixing them up is a real failure mode, so keep them explicitly separate in code and in any reporting output. 

**A. Layer 14 research-phase kill criteria** (used during Phase 1/2 backtesting to decide continue-vs-revise-the-hypothesis): 

- Out-of-sample Sharpe < 0.5 → stop, revise hypothesis 

- Out-of-sample profit factor < 1.2 → stop, revise hypothesis 

- Out-of-sample max drawdown > 25% → stop, revise risk engine 

- Risk-On regime performance worse than aggregate performance → macro conditioning isn't working, remove and retest simpler hypothesis 

- Ensemble weight instability across > 2 consecutive folds → unreliable, fall back to equal weighting and investigate 

**B. Tier 1 live-trading gate** (a separate, stricter six-metric checklist — all six must clear simultaneously on the out-of-sample test set before real capital is ever deployed; this is checked again at Phase 5, not just once in Phase 1): 

|**Metric**|**Defnition**|**Minimum Threshold**|
|---|---|---|
|Proft Factor|Gross proft / gross loss|> 1.5|
|Sharpe Ratio|Risk-adjusted return vs. risk-free rate|> 1.0|
|Sortino Ratio|Downside-deviation-adjusted return|> 1.5|
|Calmar Ratio|Annual return / max drawdown|> 0.5|
|Maximum Drawdown|Peak-to-trough capital loss|< 20%|
|Expected Value per Trade|Average EV after all costs|> 0, target > 0.3%|



**Definition** 

**Metric** 

**Minimum Threshold** 

**Tier 2 diagnostic metrics** (informational, not gates — track these but they don't block deployment on their own): 

|**Metric**|**Purpose**|
|---|---|
|Win Rate|Understanding the system, not optimising it|
|Average Hold Time|Matching empirical holds to the holding period<br>assumption|
|Trade Frequency|Ensuring suficient sample size for statistical validity|
|Signal Quality<br>(Precision/Recall)|Evaluating the Layer 7B direction classifer|
|Prediction Calibration Score|Are 70% forecasts right 70% of the time?|
|Regime-Conditional Sharpe|Where specifcally does edge exist, and where does it<br>degrade?|
|Ensemble Weight Stability|Are meta-model weights stable across walk-forward<br>folds?|



# **Macro regime states (Layer 3) — exact table** 

|**Regime**|**Conditions**|**Trading Implication**|
|---|---|---|
|Risk-On|DXY falling/fat, VIX < 20 and falling, BTC<br>dominance expanding, BTC/QQQ<br>correlation positive|<br>Full trading, standard EV<br>thresholds|
||DXY rising, VIX > 25 or spiking, cross-|No new long entries in risk|
|Risk-Of|asset correlations compressing toward<br>1.0|assets; existing positions<br>reviewed for exit|
|Transitional|Mixed signals, no clear directional<br>consensus|Halve position sizes, apply<br>more conservative EV<br>threshold|
|Macro-Event<br>Window|Scheduled high-impact event within 24<br>hours (FOMC, NFP, CPI, BOJ meeting,<br>GDP release)|No new entries; existing<br>positions get tighter stops|



Note: BTC/QQQ rolling 20-day correlation determines whether macro conditioning is even relevant to BTC at all — in low-correlation periods, reduce macro weight rather than applying it blindly. 

# **MTF alignment scoring (Layer 4) — exact table** 

|**Score**|**Meaning**|**Action**|
|---|---|---|
|4/4|All timeframes aligned|Full position size permitted|
|3/4|One minor divergence, clear overall bias|Standard position size permitted|
|2/4|Mixed structure, no clear directional<br>case|No entry|
|1/4 or 0/|4 Severe structural confict|No entry, existing positions<br>reviewed|



# **Volatility state classification (Layer 6) — exact table** 

|**State**|**ATR Percentile**|**Trading Implication**|
|---|---|---|
|Compressed|< 30th|Pre-breakout; reduce size until directional<br>confrmation|
|Expanding|30th–70th|Active trend initiation; standard position sizing|
|Extended|> 70th|Mature trend/blow-of; reduce size, tighten stops<br>aggressively|
|Spike|ATR doubles within 5<br>candles|Tail-risk event; no new entries across all assets|



# **Kill Switch trigger conditions (Layer 17) — exact table, all seven** 

|**Condition**|**Immediate Action**|
|---|---|
|Rolling 30-day drawdown > 10%|Halt all new entries; manual engineering review<br>before re-enable|
|Drawdown > 15% from equity high-|Full system pause; physical administrator reset|
|water mark|only, no automatic re-enable|
|Rolling prediction calibration error ><br>15 percentage points|Halt order routing; isolate model for recalibration|
|Feature drift confrmed (p-value <|Halt entries; inspect data feed and macro regime|
|0.01)|state|
|Exchange data feed outage > 15|Unconditional halt — no current price data is not|



# **Condition** 

# **Immediate Action** 

minutes a neutral state Volatility Spike (ATR doubles within 5 Block all new entries; flag open positions for candles) emergency review Three consecutive signal failures of Suspend that signal category; investigate before the same type re-enabling 

No automated re-enable path exists anywhere for any of these seven — every one requires explicit human sign-off. 

# **Tech stack** 

- Python 3.11+, uv for dependency management 

- pandas, pandas-ta (feature engineering — don't hand-roll indicators) 

- ccxt (free exchange OHLCV), yfinance + FRED (free macro data) 

- duckdb (data storage/queries) 

- scikit-learn (k-NN), xgboost or lightgbm (gradient boosting + quantile regression) 

- vectorbt (backtesting scaffold — don't build walk-forward/Monte Carlo from scratch) 

- mlflow (experiment tracking, model versioning) 

- optuna (hyperparameter tuning — validation fold only) 

- pytest, ruff 

# **Repo structure** 

trade-system/ 

├── data/{raw,normalized}/ 

├── src/{data_engine,features,predictive_core,backtesting, 

│        macro_mtf_liquidity_vol,risk_sizing_selection,monitoring_killswitch}/ 

├── tests/ 

├── mlruns/ 

└── notebooks/ 

# **Definition of done, for every chunk of code produced** 

- [ ] Passes pytest including any no-lookahead / temporal-boundary test for that module 

- [ ] Type hints on all function signatures 

- [ ] Docstring stating what data it expects and what it returns 

- [ ] No hardcoded file paths — reads from config 

- [ ] Raw data never mutated 

- [ ] If it touches position sizing or risk: the 2%/6%/10% caps are enforced in code 

- [ ] If it touches a gate or threshold: confirm which of the two gate systems (Layer 14 research criteria vs. Tier 1 live gate) it belongs to, and use the exact numbers from the tables above — don't approximate or paraphrase a threshold 

