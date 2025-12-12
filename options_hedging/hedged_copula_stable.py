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
        self.SetEndDate(2023, 1, 1)
        self.SetCash(100000)

        # --- Copula Strategy Parameters ---
        self.lookback = 252
        self.cap_threshold = 0.95
        self.floor_threshold = 0.05
        self.exit_threshold = 0.5
        self.exit_tolerance = 0.1
        
        # --- Assets ---
        self.pair = ["GDX", "GLD"] 
        self.symbols = [self.AddEquity(ticker, Resolution.Daily).Symbol for ticker in self.pair]
        self.s1 = self.symbols[0] # GDX
        self.s2 = self.symbols[1] # GLD

        # --- Options Setup ---
        # We will NOT subscribe to full option chains (often unstable locally).
        # Instead, we select specific contracts via OptionChainProvider and then
        # subscribe with AddOptionContract only when needed.
        self.min_dte = 25
        self.max_dte = 40
        self.call_otm_mult = 1.02

        # --- Volatility Indicator (Regime Detection) ---
        # We use SPY as a proxy for market volatility
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.vol_window = 21 # 1 month rolling volatility
        self.std = self.STD(self.spy, self.vol_window, Resolution.Daily)
        
        # Threshold for "High Volatility" (Annualized Vol > ~20-25%)
        # Daily Vol 0.015 ~= 23.8% Annualized
        self.high_vol_threshold = 0.015 

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
            elif tau < 0.2:
                self.Liquidate()
                self.active_put = None
                self.active_call = None

        # ------------------------------------------------
        # 3. Options Hedging Logic
        # ------------------------------------------------
        # Only hedge if we have an equity position
        if self.Portfolio.Invested:
            self.ManageOptionsHedge(data)

    def ManageOptionsHedge(self, data):
        if not self.std.IsReady: return
        
        # Determine Volatility Regime
        current_vol = self.std.Current.Value
        is_high_vol = current_vol > self.high_vol_threshold
        
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
            if self.active_call is not None and self.Portfolio[self.active_call].Invested:
                self.Liquidate(self.active_call)
            self.active_put = None
            self.active_call = None
        self.active_underlying = long_symbol
        
        # Calculate required contracts (1 contract = 100 shares)
        # Round down to avoid over-hedging
        shares_held = self.Portfolio[long_symbol].Quantity
        required_contracts = int(shares_held // 100)
        
        if required_contracts <= 0: return

        underlying_price = self.Securities[long_symbol].Price

        # Pull available contracts from provider (no OptionChains dependency)
        try:
            contracts = self.OptionChainProvider.GetOptionContractList(long_symbol, self.Time)
        except:
            return

        if contracts is None or len(contracts) == 0:
            return

        # Filter contracts by DTE window
        filtered = []
        for contract in contracts:
            try:
                expiry = contract.ID.Date
                dte = int((expiry - self.Time).TotalDays)
                if dte < self.min_dte or dte > self.max_dte:
                    continue
                filtered.append(contract)
            except:
                continue

        if len(filtered) == 0:
            return

        # --- Strategy Execution ---
        
        if is_high_vol:
            # === High Volatility: Buy Protective Put ===
            # 1. Close any existing covered call we opened
            if self.active_call is not None and self.Portfolio[self.active_call].Invested:
                self.Liquidate(self.active_call)
            self.active_call = None

            # 2. Buy Put if not already hedged
            if self.active_put is None or not self.Portfolio[self.active_put].Invested:
                # Select ATM Put (strike closest to spot)
                best_put = None
                best_dist = None
                for contract in filtered:
                    if contract.ID.OptionRight != OptionRight.Put:
                        continue
                    strike = float(contract.ID.StrikePrice)
                    dist = abs(strike - underlying_price)
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_put = contract

                if best_put is None:
                    return

                # Subscribe to the specific contract then buy
                if not self.Securities.ContainsKey(best_put):
                    self.AddOptionContract(best_put, Resolution.Daily)
                self.Buy(best_put, required_contracts)
                self.active_put = best_put
                # self.Debug(f"High Vol ({current_vol:.4f}): Bought Protection {atm_put.Symbol}")

        else:
            # === Low/Normal Volatility: Sell Covered Call ===
            # 1. Close any protective put we opened
            if self.active_put is not None and self.Portfolio[self.active_put].Invested:
                self.Liquidate(self.active_put)
            self.active_put = None

            # 2. Sell Call if not already short
            if self.active_call is None or not self.Portfolio[self.active_call].Invested:
                # Select OTM Call (Strike > 102% of Price)
                best_call = None
                best_strike = None
                strike_floor = underlying_price * self.call_otm_mult
                for contract in filtered:
                    if contract.ID.OptionRight != OptionRight.Call:
                        continue
                    strike = float(contract.ID.StrikePrice)
                    if strike <= strike_floor:
                        continue
                    if best_strike is None or strike < best_strike:
                        best_strike = strike
                        best_call = contract

                if best_call is None:
                    return

                # Subscribe to the specific contract then sell
                if not self.Securities.ContainsKey(best_call):
                    self.AddOptionContract(best_call, Resolution.Daily)
                self.Sell(best_call, required_contracts)
                self.active_call = best_call
                    # self.Debug(f"Low Vol ({current_vol:.4f}): Sold Covered Call {target_call.Symbol}")
