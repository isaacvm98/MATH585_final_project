# Copula-Based Pairs Trading with Options Hedging (Multi-Pair Version)
# ---------------------------------------------------------------------
# This strategy extends the Copula Pairs Trading logic by:
# 1. Supporting multiple pairs trading simultaneously
# 2. Adding an options hedging layer per pair
#
# Core Strategy (Copula):
# - Uses Clayton Copula to identify mispricing between paired assets.
# - Trades based on conditional probabilities (Mispricing Index).
#
# Hedging Layer (Options):
# - Monitors market volatility (using SPY HMM regime detection).
# - High Volatility Regime: Buys Protective Puts + Sells Covered Calls on long leg.

from AlgorithmImports import *
import math
from collections import deque

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
except Exception:
    MarkovRegression = None


class MomentumStrategy:
    """Dual Confirmation Momentum Strategy using ROC and SMA."""
    
    def __init__(self, algo, ticker="SPY", weight=0.3, roc_period=252, sma_period=200):
        """
        Initialize momentum strategy.
        
        Args:
            algo: The parent QCAlgorithm instance
            ticker: Target asset ticker (default: SPY)
            weight: Portfolio weight allocated to this strategy
            roc_period: Rate of Change lookback period (default: 252 days = 12 months)
            sma_period: Simple Moving Average period (default: 200 days)
        """
        self.algo = algo
        self.ticker = ticker
        self.weight = weight
        self.roc_period = roc_period
        self.sma_period = sma_period
        
        # --- Asset ---
        self.symbol = self.algo.AddEquity(self.ticker, Resolution.Daily).Symbol
        self.shv = self.algo.AddEquity("SHV", Resolution.Daily).Symbol  # Cash parking asset
        
        # --- Indicators ---
        self.roc = self.algo.ROC(self.symbol, self.roc_period, Resolution.Daily)
        self.sma = self.algo.SMA(self.symbol, self.sma_period, Resolution.Daily)
        self.sma_fast = self.algo.SMA(self.symbol, 63, Resolution.Daily)  # 63-day SMA for sensitive exit
        self.adx = self.algo.ADX(self.symbol, 14, Resolution.Daily)  # ADX(14) for trend strength
        
        # --- State ---
        self.is_invested = False
        self.initialized = False  # Track if we've done initial allocation
    
    def OnData(self, data):
        """Process new data for momentum strategy."""
        # Wait for indicators to be ready
        if not self.roc.IsReady or not self.sma.IsReady or not self.sma_fast.IsReady or not self.adx.IsReady:
            return
        
        if not data.ContainsKey(self.symbol) or data[self.symbol] is None:
            return
        
        current_price = float(data[self.symbol].Close)
        roc_value = float(self.roc.Current.Value)
        sma_value = float(self.sma.Current.Value)
        sma_fast_value = float(self.sma_fast.Current.Value)
        adx_value = float(self.adx.Current.Value)
        
        # Check current position status
        self.is_invested = self.algo.Portfolio[self.symbol].Invested
        is_in_shv = self.algo.Portfolio[self.shv].Invested
        
        # Initial allocation: if not invested in either, park in SHV
        if not self.initialized and not self.is_invested and not is_in_shv:
            self.algo.SetHoldings(self.shv, self.weight)
            self.algo.Debug(f"[MOMENTUM] Initial: Parking in SHV @ {self.weight:.0%}")
            self.initialized = True
            return
        
        # Entry: Price > SMA AND ROC > 0 AND ADX > 20
        if not self.is_invested:
            if current_price > sma_value and roc_value > 0 and adx_value > 20:
                # Sell SHV first, then buy SPY
                if is_in_shv:
                    self.algo.Liquidate(self.shv)
                self.algo.SetHoldings(self.symbol, self.weight)
                self.algo.Debug(f"[MOMENTUM] Entry: {self.ticker} @ {current_price:.2f} (SMA={sma_value:.2f}, ROC={roc_value:.2%}, ADX={adx_value:.1f})")
        
        # Exit: Price < SMA(200) OR ROC < 0 OR Price < SMA(63) OR ADX < 15
        else:
            if current_price < sma_value or roc_value < 0 or adx_value < 15:
                exit_reason = []
                if current_price < sma_value:
                    exit_reason.append("<SMA200")
                if roc_value < 0:
                    exit_reason.append("ROC<0")
                # if current_price < sma_fast_value:
                #     exit_reason.append("<SMA63")
                if adx_value < 15:
                    exit_reason.append("ADX<15")
                #     exit_reason.append("<SMA63")
                # Sell SPY and park in SHV
                self.algo.Liquidate(self.symbol)
                self.algo.SetHoldings(self.shv, self.weight)
                self.algo.Debug(f"[MOMENTUM] Exit: {self.ticker} @ {current_price:.2f} -> SHV (Reason: {', '.join(exit_reason)})")


class CopulaPair:
    """Encapsulates a single pair trading strategy with copula logic and options hedging."""
    
    def __init__(self, algo, tickers, weight=0.5):
        """
        Initialize a copula pair.
        
        Args:
            algo: The parent QCAlgorithm instance
            tickers: List of two ticker symbols [ticker1, ticker2]
            weight: Portfolio weight allocated to this pair (e.g., 0.5 = 50%)
        """
        self.algo = algo
        self.ticker1 = tickers[0]
        self.ticker2 = tickers[1]
        self.weight = weight
        
        # --- Assets ---
        self.s1 = self.algo.AddEquity(self.ticker1, Resolution.Daily).Symbol
        self.s2 = self.algo.AddEquity(self.ticker2, Resolution.Daily).Symbol
        
        # Options require Raw normalization
        self.algo.Securities[self.s1].SetDataNormalizationMode(DataNormalizationMode.Raw)
        self.algo.Securities[self.s2].SetDataNormalizationMode(DataNormalizationMode.Raw)
        
        self.symbols = [self.s1, self.s2]

        # --- Copula Strategy Parameters ---
        self.lookback = 252
        self.cap_threshold = 0.95
        self.floor_threshold = 0.05
        self.exit_threshold = 0.5
        self.exit_tolerance = 0.1
        
        # --- Risk Management ---
        self.stop_loss_pct = 0.05
        self.max_holding_days = 60
        self.entry_time = None

        # --- Options Params ---
        self.min_dte = 1
        self.max_dte = 40
        self.call_otm_mult = 1.02

        # --- Data Structures ---
        self.window_size = self.lookback + 1
        self.s1_window = deque(maxlen=self.window_size)
        self.s2_window = deque(maxlen=self.window_size)

        # --- Hedge State ---
        self.active_put = None
        self.active_call = None
        self.active_underlying = None

    @staticmethod
    def _to_uniform_ranks(values):
        """Convert a 1D sequence to pseudo-uniforms in (0, 1) using ordinal ranks."""
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
            return 1 if v > 0 else -1 if v < 0 else 0

        concordant = 0
        discordant = 0
        for i in range(n - 1):
            xi = x[i]
            yi = y[i]
            for j in range(i + 1, n):
                s = sign(x[j] - xi) * sign(y[j] - yi)
                if s > 0:
                    concordant += 1
                elif s < 0:
                    discordant += 1

        return (concordant - discordant) / (n * (n - 1) / 2.0)

    def OnData(self, data):
        """Process new data for this pair."""
        # 1. Update Data Windows
        if data.ContainsKey(self.s1) and data[self.s1] is not None:
            self.s1_window.append(float(data[self.s1].Close))
        if data.ContainsKey(self.s2) and data[self.s2] is not None:
            self.s2_window.append(float(data[self.s2].Close))

        if len(self.s1_window) < self.window_size or len(self.s2_window) < self.window_size:
            return

        # 2. Calculate Copula Mispricing
        prices_x = list(self.s1_window)
        prices_y = list(self.s2_window)

        ret_x, ret_y = [], []
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
            if term_base <= 0:
                return
            mispricing_prob = pow(term_base, (-1 / theta - 1)) * pow(v_today, (-theta - 1))
            if math.isnan(mispricing_prob) or math.isinf(mispricing_prob):
                return
        except:
            return

        # 3. Execute Trading Signals
        invested_s1 = self.algo.Portfolio[self.s1].Invested
        invested_s2 = self.algo.Portfolio[self.s2].Invested
        is_invested = invested_s1 or invested_s2

        # Per-leg weight: half of pair weight (long one, short the other)
        leg_weight = self.weight * 0.5

        if not is_invested:
            # Entry Logic
            if mispricing_prob < self.floor_threshold:
                # Long S1, Short S2
                self.algo.SetHoldings(self.s1, leg_weight)
                self.algo.SetHoldings(self.s2, -leg_weight)
                self.entry_time = self.algo.Time
            elif mispricing_prob > self.cap_threshold:
                # Short S1, Long S2
                self.algo.SetHoldings(self.s1, -leg_weight)
                self.algo.SetHoldings(self.s2, leg_weight)
                self.entry_time = self.algo.Time
        else:
            # Exit Logic
            days_held = (self.algo.Time - self.entry_time).days if self.entry_time else 0

            # Time Stop
            if days_held > self.max_holding_days:
                self.algo.Debug(f"[{self.ticker1}-{self.ticker2}] Time Stop Hit: Held {days_held} days")
                self.LiquidatePair()
                return

            # Mean Reversion Exit
            if (0.5 - self.exit_tolerance) < mispricing_prob < (0.5 + self.exit_tolerance):
                self.LiquidatePair()
                return

            # Correlation Breakdown Exit
            if tau < 0.2 and self.algo.is_high_vol_regime:
                self.LiquidatePair()
                return

        # 4. Options Hedging
        if is_invested:
            self.ManageOptionsHedge()

    def LiquidatePair(self):
        """Liquidate all positions for this pair."""
        self.algo.Liquidate(self.s1)
        self.algo.Liquidate(self.s2)
        if self.active_put:
            self.algo.Liquidate(self.active_put)
        if self.active_call:
            self.algo.Liquidate(self.active_call)
        self.active_put = None
        self.active_call = None
        self.entry_time = None

    def ManageOptionsHedge(self):
        """Manage options hedging for this pair."""
        is_high_vol = self.algo.is_high_vol_regime

        # Identify Long Leg
        long_symbol = None
        if self.algo.Portfolio[self.s1].IsLong:
            long_symbol = self.s1
        elif self.algo.Portfolio[self.s2].IsLong:
            long_symbol = self.s2

        if not long_symbol:
            return

        # If underlying flipped, clear old hedges
        if self.active_underlying is not None and self.active_underlying != long_symbol:
            if self.active_put:
                self.algo.Liquidate(self.active_put)
            if self.active_call:
                self.algo.Liquidate(self.active_call)
            self.active_put = None
            self.active_call = None
        self.active_underlying = long_symbol

        shares_held = self.algo.Portfolio[long_symbol].Quantity
        required_contracts = int(shares_held // 100)

        if required_contracts <= 0:
            return

        # Get Contracts
        try:
            contracts = self.algo.OptionChainProvider.GetOptionContractList(long_symbol, self.algo.Time)
        except:
            return

        if not contracts:
            return

        # Filter by DTE
        algo_date = self.algo.Time.date()
        filtered = []
        for c in contracts:
            try:
                expiry_dt = c.ID.Date
                expiry_date = expiry_dt.date() if hasattr(expiry_dt, "date") else expiry_dt
                dte = int((expiry_date - algo_date).days)
                if self.min_dte <= dte <= self.max_dte:
                    filtered.append(c)
            except:
                continue

        if not filtered:
            return

        # Execution
        if not is_high_vol:
            # Close hedges in low vol
            if self.active_put and self.algo.Portfolio[self.active_put].Invested:
                self.algo.Liquidate(self.active_put)
            if self.active_call and self.algo.Portfolio[self.active_call].Invested:
                self.algo.Liquidate(self.active_call)
            self.active_put = None
            self.active_call = None
            return

        target_expiry = self._get_nearest_friday_expiry(filtered)
        if not target_expiry:
            return

        contracts_on_expiry = [
            c for c in filtered
            if (c.ID.Date.date() if hasattr(c.ID.Date, "date") else c.ID.Date) == target_expiry
        ]
        if not contracts_on_expiry:
            return

        underlying_price = self.algo.Securities[long_symbol].Price

        # Buy Put
        if self._needs_roll(self.active_put, long_symbol, target_expiry):
            if self.active_put:
                self.algo.Liquidate(self.active_put)
            self.active_put = None

            best_put = self._select_atm_put(contracts_on_expiry, underlying_price)
            if best_put:
                self.algo.AddOptionContract(best_put, Resolution.Daily)
                self.algo.Buy(best_put, required_contracts)
                self.active_put = best_put

        # Sell Call
        if self._needs_roll(self.active_call, long_symbol, target_expiry):
            if self.active_call:
                self.algo.Liquidate(self.active_call)
            self.active_call = None

            best_call = self._select_otm_call(contracts_on_expiry, underlying_price)
            if best_call:
                self.algo.AddOptionContract(best_call, Resolution.Daily)
                self.algo.Sell(best_call, required_contracts)
                self.active_call = best_call

    def _get_nearest_friday_expiry(self, contracts):
        """Return the nearest future Friday expiry date."""
        best = None
        for c in contracts:
            try:
                expiry_dt = c.ID.Date
                expiry_date = expiry_dt.date() if hasattr(expiry_dt, "date") else expiry_dt
                if hasattr(expiry_date, "weekday") and expiry_date.weekday() != 4:
                    continue
                if expiry_date < self.algo.Time.date():
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
            return expiry_date != target_expiry
        except:
            return True

    def _select_atm_put(self, contracts, underlying_price):
        """Select ATM put option."""
        best = None
        best_dist = None
        for c in contracts:
            if c.ID.OptionRight != OptionRight.Put:
                continue
            dist = abs(float(c.ID.StrikePrice) - underlying_price)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = c
        return best

    def _select_otm_call(self, contracts, underlying_price):
        """Select OTM call option."""
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


class HedgedCopulaPairsTrading(QCAlgorithm):
    """Main algorithm managing multiple copula pairs."""

    def Initialize(self):
        self.SetStartDate(2019, 1, 1)
        self.SetEndDate(2026, 9, 1)
        self.SetCash(100000)

        # --- HMM Regime Detection (Volatility) ---
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

        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.BeforeMarketClose(self.spy, 5),
            self.UpdateHMMRegime
        )

        # --- Strategy Weight Allocation ---
        # Adjust these weights to balance between strategies
        self.momentum_weight = 0.75  # 30% to momentum strategy
        self.pairs_weight = 1     # 70% to pairs trading (split among pairs)
        
        # --- Momentum Strategy Setup ---
        # Dual Confirmation: ROC(252) + SMA(200)
        self.momentum = MomentumStrategy(
            self, 
            ticker="QQQ", 
            weight=self.momentum_weight,
            roc_period=120,  # 12-month momentum
            sma_period=200   # 200-day moving average
        )

        # --- Pairs Setup ---
        # Split pairs_weight equally among pairs
        num_pairs = 1
        per_pair_weight = self.pairs_weight / num_pairs
        
        self.pairs = []
        self.pairs.append(CopulaPair(self, ["GDX", "GLD"], weight=per_pair_weight))
        #self.pairs.append(CopulaPair(self, ["BAC", "JPM"], weight=per_pair_weight))

        self.SetWarmUp(253)

    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return
        
        # Run momentum strategy
        self.momentum.OnData(data)
        
        # Delegate to each pair strategy
        for pair in self.pairs:
            pair.OnData(data)

    def UpdateHMMRegime(self):
        """Fit HMM on SPY daily returns and update self.is_high_vol_regime."""
        if MarkovRegression is None:
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

            high_regime = 1
            regime = int(probs.argmax())
            confidence = float(max(probs))

            self._last_hmm_date = self.Time.date()

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
                    self.Debug(f"[HMM] REGIME SWITCH {self.current_regime} -> {self.predicted_regime}")
                    self.current_regime = self.predicted_regime
                    self.regime_day_count = 0

            self.is_high_vol_regime = (self.current_regime == high_regime)

            if self._last_is_high_vol is None or self._last_is_high_vol != self.is_high_vol_regime:
                state = "HIGH" if self.is_high_vol_regime else "LOW"
                self.Debug(f"[HMM] Volatility regime now: {state}")
                self._last_is_high_vol = self.is_high_vol_regime

        except:
            return

