# Copula-Based Pairs Trading with Options Hedging
# -------------------------------------------------
# This strategy extends the Copula Pairs Trading logic by adding an options hedging layer.
# 
# Core Strategy (Copula):
# - Uses Clayton Copula to identify mispricing between GDX and GLD.
# - Trades based on conditional probabilities (Mispricing Index).
#
# Hedging Layer (Options):
# - Monitors market volatility (using SPY rolling standard deviation).
# - High Volatility Regime: Buys Protective Puts on the long leg to limit downside.
# - Low/Normal Volatility Regime: Sells Covered Calls on the long leg to generate income.

from AlgorithmImports import *
import math
from collections import deque

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
except Exception:
    MarkovRegression = None

class HedgedCopulaPairsTrading(QCAlgorithm):

    @staticmethod
    def _to_uniform_ranks(values):
        """Convert a 1D sequence to pseudo-uniforms in (0, 1) using ordinal ranks.
        Implemented in pure Python to avoid binary deps (NumPy/SciPy) that can crash LEAN."""
        n = len(values)
        if n == 0:
            return []
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        for r, idx in enumerate(order, start=1):
            ranks[idx] = float(r)
        denom = float(n) + 1.0
        return [r / denom for r in ranks]

    @staticmethod
    def _kendall_tau_a(x, y):
        """Kendall's tau-a (ignores ties). Pure Python O(n^2), OK for ~252 points."""
        n = len(x)
        if n != len(y) or n < 2:
            return float("nan")

        def sign(v):
            if v > 0:
                return 1
            if v < 0:
                return -1
            return 0

        concordant = 0
        discordant = 0
        for i in range(n - 1):
            xi = x[i]
            yi = y[i]
            for j in range(i + 1, n):
                sx = sign(x[j] - xi)
                sy = sign(y[j] - yi)
                s = sx * sy
                if s > 0:
                    concordant += 1
                elif s < 0:
                    discordant += 1

        denom = n * (n - 1) / 2.0
        return (concordant - discordant) / denom

    def Initialize(self):
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2026, 9, 1)
        self.SetCash(100000)

        # --- Copula Strategy Parameters ---
        self.lookback = 252
        self.cap_threshold = 0.95
        self.floor_threshold = 0.05
        self.exit_threshold = 0.5
        self.exit_tolerance = 0.1
        
        # --- Assets ---
        self.pair = ["GDX", "GLD"] 
        self.symbols = []
        for ticker in self.pair:
            equity = self.AddEquity(ticker, Resolution.Daily)
            # Options require Raw normalization on the underlying
            equity.SetDataNormalizationMode(DataNormalizationMode.Raw)
            self.symbols.append(equity.Symbol)
        self.s1 = self.symbols[0] # GDX
        self.s2 = self.symbols[1] # GLD

        # --- Options Setup ---
        # We will NOT subscribe to full option chains (often unstable locally).
        # Instead, we select specific contracts via OptionChainProvider and then
        # subscribe with AddOptionContract only when needed.
        self.min_dte = 1
        self.max_dte = 40
        self.call_otm_mult = 1.02

        # --- HMM Regime Detection (Volatility) ---
        # Use SPY daily returns to fit a 2-regime Markov Switching model.
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.hmm_lookback_days = 252 * 3
        self.min_regime_days = 2
        self.min_confidence = 0.52
        self.current_regime = None
        self.predicted_regime = None
        self.regime_day_count = 0
        self.is_high_vol_regime = False
        self._last_hmm_date = None
        self._last_is_high_vol = None

        # Update HMM once per day before close
        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.BeforeMarketClose(self.spy, 5),
            self.UpdateHMMRegime
        )

        # --- Data Structures ---
        self.window_size = self.lookback + 1
        # Use pure-Python deques to avoid pythonnet generic crashes seen with RollingWindow in some LEAN setups
        self.s1_window = deque(maxlen=self.window_size)
        self.s2_window = deque(maxlen=self.window_size)

        # Track active hedge legs to avoid iterating Portfolio (can crash under pythonnet)
        self.active_put = None
        self.active_call = None
        self.active_underlying = None

        self.SetWarmUp(self.window_size)

    def OnData(self, data: Slice):
        # ------------------------------------------------
        # 1. Update Data & Copula Logic
        # ------------------------------------------------
        if data.ContainsKey(self.s1) and data[self.s1] is not None:
            self.s1_window.append(float(data[self.s1].Close))
        if data.ContainsKey(self.s2) and data[self.s2] is not None:
            self.s2_window.append(float(data[self.s2].Close))

        if self.IsWarmingUp or len(self.s1_window) < self.window_size or len(self.s2_window) < self.window_size:
            return

        # Extract prices
        # Use list comprehension with float cast for safety to avoid PAL_SEHException
        prices_x = list(self.s1_window)
        prices_y = list(self.s2_window)

        # Calculate Log Returns (paired + cleaned)
        ret_x = []
        ret_y = []
        for i in range(1, min(len(prices_x), len(prices_y))):
            px0, px1 = prices_x[i - 1], prices_x[i]
            py0, py1 = prices_y[i - 1], prices_y[i]
            if px0 <= 0 or px1 <= 0 or py0 <= 0 or py1 <= 0:
                continue
            rx = math.log(px1 / px0)
            ry = math.log(py1 / py0)
            if math.isfinite(rx) and math.isfinite(ry):
                ret_x.append(rx)
                ret_y.append(ry)

        if len(ret_x) < 100:
            return

        # Fit Marginals & Copula
        u_series = self._to_uniform_ranks(ret_x)
        v_series = self._to_uniform_ranks(ret_y)
        u_today, v_today = u_series[-1], v_series[-1]

        tau = self._kendall_tau_a(ret_x, ret_y)
        
        if tau <= 0 or tau >= 1 or math.isnan(tau):
            return

        theta = (2 * tau) / (1 - tau)

        # Calculate Mispricing Probability
        try:
            u_pow = pow(u_today, -theta)
            v_pow = pow(v_today, -theta)
            term_base = u_pow + v_pow - 1
            if term_base <= 0: return 
            part_a = pow(term_base, (-1/theta - 1))
            part_b = pow(v_today, (-theta - 1))
            mispricing_prob = part_a * part_b
            if math.isnan(mispricing_prob) or math.isinf(mispricing_prob):
                return
        except:
            return

        # ------------------------------------------------
        # 2. Execute Pairs Trading Signals
        # ------------------------------------------------
        
        # Entry Logic
        if not self.Portfolio.Invested:
            if mispricing_prob < self.floor_threshold:
                # Long S1, Short S2
                self.SetHoldings(self.s1, 0.5)
                self.SetHoldings(self.s2, -0.5)
            elif mispricing_prob > self.cap_threshold:
                # Short S1, Long S2
                self.SetHoldings(self.s1, -0.5)
                self.SetHoldings(self.s2, 0.5)

        # Exit Logic
        else:
            # Mean Reversion Exit
            if (0.5 - self.exit_tolerance) < mispricing_prob < (0.5 + self.exit_tolerance):
                self.Liquidate() # Liquidates stocks and options
                self.active_put = None
                self.active_call = None
            # Correlation Breakdown Exit
            elif tau < 0.2 and self.is_high_vol_regime:
                self.Liquidate()
                self.active_put = None
                self.active_call = None

        # 3. Options Hedging Logic
        # ------------------------------------------------
        # Only hedge if we have an equity position
        if self.Portfolio.Invested:
            self.ManageOptionsHedge(data)
            # pass

    def ManageOptionsHedge(self, data):
        # Use HMM regime signal
        is_high_vol = bool(self.is_high_vol_regime)

        # Lightweight visibility: print once per day when in high-vol
        if is_high_vol and self._last_hmm_date is not None and self.Time.date() == self._last_hmm_date:
            # Keep it low-noise: only print at the first OnData after the HMM update.
            if self._last_is_high_vol is not True:
                self.Debug(f"[HMM] High-vol active on {self.Time.date()} (regime==1). Options hedge should run when invested.")
                self._last_is_high_vol = True
        
        # Identify the Long position (we only hedge the long leg)
        long_symbol = None
        for symbol in self.symbols:
            if self.Portfolio[symbol].IsLong:
                long_symbol = symbol
                break
        
        if not long_symbol: return

        # If the long underlying changes (pair flips), close any previous hedges
        if self.active_underlying is not None and self.active_underlying != long_symbol:
            if self.active_put is not None and self.Portfolio[self.active_put].Invested:
                self.Liquidate(self.active_put)
                self.RemoveSecurity(self.active_put)
            if self.active_call is not None and self.Portfolio[self.active_call].Invested:
                self.Liquidate(self.active_call)
                self.RemoveSecurity(self.active_call)
            self.active_put = None
            self.active_call = None
        self.active_underlying = long_symbol
        
        # Calculate required contracts (1 contract = 100 shares)
        # Round down to avoid over-hedging
        shares_held = self.Portfolio[long_symbol].Quantity
        required_contracts = int(shares_held // 100)
        
        if required_contracts <= 0:
            # We have a long leg but less than 100 shares => cannot hedge with options
            if is_high_vol:
                self.Debug(f"[HEDGE] High-vol but {long_symbol.Value} shares={shares_held:.2f} (<100). No option contracts.")
            return

        underlying_price = self.Securities[long_symbol].Price

        # Pull available contracts from provider (no OptionChains dependency)
        try:
            contracts = self.OptionChainProvider.GetOptionContractList(long_symbol, self.Time)
        except:
            if is_high_vol:
                self.Debug(f"[HEDGE] High-vol but OptionChainProvider failed for {long_symbol.Value} @ {self.Time}.")
            return

        if contracts is None or len(contracts) == 0:
            if is_high_vol:
                self.Debug(f"[HEDGE] High-vol but no option contracts returned for {long_symbol.Value} @ {self.Time}.")
            return

        # Filter contracts by DTE window
        filtered = []
        # Use date-only difference to avoid time-of-day / float truncation issues
        algo_date = self.Time.date()
        min_seen = None
        max_seen = None
        for contract in contracts:
            try:
                expiry_dt = contract.ID.Date
                expiry_date = expiry_dt.date() if hasattr(expiry_dt, "date") else expiry_dt
                dte = int((expiry_date - algo_date).days)
                if min_seen is None or dte < min_seen:
                    min_seen = dte
                if max_seen is None or dte > max_seen:
                    max_seen = dte
                if dte < self.min_dte or dte > self.max_dte:
                    continue
                filtered.append(contract)
            except:
                continue

        if len(filtered) == 0:
            if is_high_vol:
                self.Debug(
                    f"[HEDGE] High-vol but no contracts in DTE window {self.min_dte}-{self.max_dte} for {long_symbol.Value}. "
                    f"Returned={len(contracts)} dte_range={min_seen}-{max_seen} time={self.Time}"
                )
            return

        # --- Strategy Execution ---
        # Requirement: when HMM indicates high volatility, execute BOTH:
        # - Buy protective put
        # - Sell covered call
        # Expiry should be the nearest Friday.

        if not is_high_vol:
            # Not in high-vol regime: close hedges (keep the equity pair position)
            if self.active_put is not None and self.Portfolio[self.active_put].Invested:
                self.Liquidate(self.active_put)
            if self.active_call is not None and self.Portfolio[self.active_call].Invested:
                self.Liquidate(self.active_call)
            self.active_put = None
            self.active_call = None
            return

        target_expiry = self._get_nearest_friday_expiry(filtered)
        if target_expiry is None:
            self.Debug(f"[HEDGE] High-vol but no Friday expiry found within DTE window for {long_symbol.Value}.")
            return

        contracts_on_expiry = [c for c in filtered if (c.ID.Date.date() if hasattr(c.ID.Date, "date") else c.ID.Date) == target_expiry]
        if len(contracts_on_expiry) == 0:
            self.Debug(f"[HEDGE] High-vol but no contracts found for Friday expiry {target_expiry.date()} on {long_symbol.Value}.")
            return

        # 1) Ensure we have a protective put for the nearest Friday
        if self._needs_roll(self.active_put, long_symbol, target_expiry):
            if self.active_put is not None and self.Portfolio[self.active_put].Invested:
                self.Liquidate(self.active_put)
            self.active_put = None

            best_put = self._select_atm_put(contracts_on_expiry, underlying_price)
            if best_put is not None:
                if not self.Securities.ContainsKey(best_put):
                    self.AddOptionContract(best_put, Resolution.Daily)
                self.Buy(best_put, required_contracts)
                self.active_put = best_put
                self.Debug(f"[HEDGE] Bought PUT {best_put.ID.StrikePrice} exp {best_put.ID.Date.date()} x{required_contracts} on {long_symbol.Value}")
            else:
                self.Debug(f"[HEDGE] High-vol but could not select ATM put for {long_symbol.Value} exp {target_expiry.date()}.")

        # 2) Ensure we have a covered call for the nearest Friday
        if self._needs_roll(self.active_call, long_symbol, target_expiry):
            if self.active_call is not None and self.Portfolio[self.active_call].Invested:
                self.Liquidate(self.active_call)
            self.active_call = None

            best_call = self._select_otm_call(contracts_on_expiry, underlying_price)
            if best_call is not None:
                if not self.Securities.ContainsKey(best_call):
                    self.AddOptionContract(best_call, Resolution.Daily)
                self.Sell(best_call, required_contracts)
                self.active_call = best_call
                self.Debug(f"[HEDGE] Sold CALL {best_call.ID.StrikePrice} exp {best_call.ID.Date.date()} x{required_contracts} on {long_symbol.Value}")
            else:
                self.Debug(f"[HEDGE] High-vol but could not select OTM call for {long_symbol.Value} exp {target_expiry.date()}.")

    def UpdateHMMRegime(self):
        """Fit HMM on SPY daily returns and update self.is_high_vol_regime."""
        if MarkovRegression is None:
            # If statsmodels isn't available in the environment, we will never detect regimes.
            # Keep this quiet after first notice.
            if self._last_hmm_date is None:
                self.Debug("[HMM] statsmodels.MarkovRegression not available; HMM regime detection disabled.")
            return

        try:
            history = self.History([self.spy], self.hmm_lookback_days, Resolution.Daily)
            if history is None or history.empty:
                return

            close = history['close'].unstack(level=0)
            returns = close.pct_change().dropna()
            if returns.empty:
                return

            model = MarkovRegression(returns, k_regimes=2, switching_variance=True)
            res = model.fit(disp=False)

            probs = res.smoothed_marginal_probabilities.values[-1]
            if len(probs) != 2:
                return

            # Match hmm.py mapping: regime==1 is treated as high volatility
            high_regime = 1

            regime = int(probs.argmax())
            confidence = float(max(probs))

            # Daily visibility (once per HMM update)
            self._last_hmm_date = self.Time.date()
            self.Debug(f"[HMM] {self.Time.date()} predicted={regime} conf={confidence:.4f} (regime==1 means high-vol)")

            if self.current_regime is None:
                self.current_regime = regime
                self.predicted_regime = regime
                self.regime_day_count = 1
            else:
                if regime == self.predicted_regime:
                    self.regime_day_count += 1
                else:
                    self.predicted_regime = regime
                    self.regime_day_count = 1

                should_switch = (
                    self.predicted_regime != self.current_regime and
                    self.regime_day_count >= self.min_regime_days and
                    confidence >= self.min_confidence
                )

                if should_switch:
                    self.Debug(f"[HMM] REGIME SWITCH {self.current_regime} -> {self.predicted_regime} (days={self.regime_day_count}, conf={confidence:.4f})")
                    self.current_regime = self.predicted_regime
                    self.regime_day_count = 0

            self.is_high_vol_regime = (self.current_regime == high_regime)

            # Print on transitions into/out of high vol
            if self._last_is_high_vol is None or self._last_is_high_vol != self.is_high_vol_regime:
                state = "HIGH" if self.is_high_vol_regime else "LOW"
                self.Debug(f"[HMM] Volatility regime now: {state} (current_regime={self.current_regime})")
                self._last_is_high_vol = self.is_high_vol_regime

        except:
            # Keep last known regime
            return

    def _get_nearest_friday_expiry(self, contracts):
        """Return the nearest future Friday expiry date from a list of option symbols."""
        best = None  # python datetime.date
        for c in contracts:
            try:
                expiry_dt = c.ID.Date
                expiry_date = expiry_dt.date() if hasattr(expiry_dt, "date") else expiry_dt
                # Friday: python weekday()==4
                if hasattr(expiry_date, "weekday") and expiry_date.weekday() != 4:
                    continue
                if expiry_date < self.Time.date():
                    continue
                if best is None or expiry_date < best:
                    best = expiry_date
            except:
                continue
        return best

    def _needs_roll(self, option_symbol, underlying_symbol, target_expiry):
        """True if we don't have an active option, or it doesn't match underlying/expiry."""
        if option_symbol is None:
            return True
        try:
            if option_symbol.Underlying != underlying_symbol:
                return True
            expiry_dt = option_symbol.ID.Date
            expiry_date = expiry_dt.date() if hasattr(expiry_dt, "date") else expiry_dt
            if expiry_date != target_expiry:
                return True
            return False
        except:
            return True

    def _select_atm_put(self, contracts, underlying_price):
        best = None
        best_dist = None
        for c in contracts:
            if c.ID.OptionRight != OptionRight.Put:
                continue
            strike = float(c.ID.StrikePrice)
            dist = abs(strike - underlying_price)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = c
        return best

    def _select_otm_call(self, contracts, underlying_price):
        best = None
        best_strike = None
        strike_floor = underlying_price * self.call_otm_mult
        for c in contracts:
            if c.ID.OptionRight != OptionRight.Call:
                continue
            strike = float(c.ID.StrikePrice)
            if strike <= strike_floor:
                continue
            if best_strike is None or strike < best_strike:
                best_strike = strike
                best = c
        return best

