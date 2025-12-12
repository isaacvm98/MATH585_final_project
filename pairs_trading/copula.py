
# Copula-Based Pairs Trading Algorithm
# -------------------------------------
# This strategy uses a Clayton Copula to model the joint dependence of a stock pair.
# It calculates a "Mispricing Index" based on conditional probabilities derived
# from the fitted copula.
#
# Strategy Logic:
# 1. Compute log-returns for a lookback period.
# 2. Transform returns to Uniform [0,1] using ECDF.
# 3. Estimate Clayton Copula parameter (theta) using Kendall's Tau.
# 4. Calculate Conditional Probability P(U1 <= u1 | U2 = u2).
# 5. Trade based on thresholds (0.05, 0.95) and exit at mean (0.5).

from AlgorithmImports import *
from scipy.stats import kendalltau, rankdata
import numpy as np
import math

class CopulaPairsTrading(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2018, 1, 1)   # Set Start Date
        self.SetEndDate(2025, 1, 1)     # Set End Date
        self.SetCash(100000)            # Set Strategy Cash

        # --- Strategy Parameters ---
        self.lookback = 252             # Trading days for fitting (1 year)
        self.cap_threshold = 0.95       # Upper bound for short entry
        self.floor_threshold = 0.05     # Lower bound for long entry
        self.exit_threshold = 0.5       # Mean reversion target
        self.exit_tolerance = 0.1       # Exit band (0.4 to 0.6)
        
        # --- Asset Selection ---
        # Gold Miners vs Gold ETF is a classic cointegrated pair
        self.pair = ["GDX", "GLD"] 
        self.symbols = [self.AddEquity(ticker, Resolution.Daily).Symbol for ticker in self.pair]
        self.s1 = self.symbols[0] # GDX
        self.s2 = self.symbols[1] # GLD

        # --- Data Structures ---
        # FIX: Use RollingWindow of floats (prices) instead of Slices.
        # This prevents C#/Python interop crashes (PAL_SEHException).
        self.window_size = self.lookback + 1 # Need +1 to calc returns from prices
        
        self.s1_window = RollingWindow[float](self.window_size)
        self.s2_window = RollingWindow[float](self.window_size)

        # Warm up period
        self.SetWarmUp(self.window_size)

    def OnData(self, data: Slice):
        # 1. Add data to rolling windows
        if data.ContainsKey(self.s1) and data[self.s1] is not None:
            self.s1_window.Add(data[self.s1].Close)
            
        if data.ContainsKey(self.s2) and data[self.s2] is not None:
            self.s2_window.Add(data[self.s2].Close)

        # Check if we are ready to trade
        if self.IsWarmingUp or not self.s1_window.IsReady or not self.s2_window.IsReady:
            return

        # 2. Extract History from Rolling Window
        # list(window) returns [MostRecent, ..., Oldest]. 
        # We convert to numpy and reverse to [Oldest, ..., MostRecent] for time-series math.
        prices_x = np.array(list(self.s1_window))[::-1]
        prices_y = np.array(list(self.s2_window))[::-1]

        # 3. Calculate Log Returns
        # R_t = ln(P_t / P_{t-1})
        # Add safe check for zeros just in case
        with np.errstate(divide='ignore', invalid='ignore'):
            ret_x = np.log(prices_x[1:] / prices_x[:-1])
            ret_y = np.log(prices_y[1:] / prices_y[:-1])
        
        # Clean any potential NaNs/Infs from data gaps
        valid_mask = np.isfinite(ret_x) & np.isfinite(ret_y)
        ret_x = ret_x[valid_mask]
        ret_y = ret_y[valid_mask]

        if len(ret_x) < 100: # Ensure we still have enough data after cleaning
            return

        # 4. Fit Marginal Distributions (ECDF) & Transform to Uniforms [0, 1]
        # rankdata/len gives us the percentile (u, v)
        u_series = rankdata(ret_x) / (len(ret_x) + 1)
        v_series = rankdata(ret_y) / (len(ret_y) + 1)

        # The specific u, v for today (the most recent data point)
        u_today = u_series[-1]
        v_today = v_series[-1]

        # 5. Fit Copula (Clayton)
        # We estimate Theta using Kendall's Tau to avoid expensive MLE optimization
        try:
            tau, _ = kendalltau(ret_x, ret_y)
        except:
            return 
        
        # Clayton copula requires Theta > 0 (positive dependence)
        if tau <= 0 or tau >= 1 or np.isnan(tau):
            return

        theta = (2 * tau) / (1 - tau)

        # 6. Calculate Mispricing Index (Conditional Probability)
        # We compute P(U <= u | V = v). This tells us: Given the return of Asset B (v),
        # what is the probability that Asset A would be lower than it currently is?
        
        try:
            # Safe computation to avoid overflow with power functions
            # Term base: (u^-theta + v^-theta - 1)
            u_pow = np.power(u_today, -theta)
            v_pow = np.power(v_today, -theta)
            
            term_base = u_pow + v_pow - 1
            
            if term_base <= 0: return 

            # Formula: P(U|V) = term_base^(-1/theta - 1) * v^(-theta - 1)
            part_a = np.power(term_base, (-1/theta - 1))
            part_b = np.power(v_today, (-theta - 1))
            
            mispricing_prob = part_a * part_b
            
            if np.isnan(mispricing_prob) or np.isinf(mispricing_prob):
                return
            
        except Exception as e:
            # Catch math domain errors silently
            return

        # 7. Trading Signals
        
        # --- Entry Logic ---
        if not self.Portfolio.Invested:
            # Low Probability: S1 is "too low" given S2 -> Long S1, Short S2
            if mispricing_prob < self.floor_threshold:
                self.SetHoldings(self.s1, 0.5)
                self.SetHoldings(self.s2, -0.5)
                # self.Debug(f"Entry Long: Prob {mispricing_prob:.4f}")

            # High Probability: S1 is "too high" given S2 -> Short S1, Long S2
            elif mispricing_prob > self.cap_threshold:
                self.SetHoldings(self.s1, -0.5)
                self.SetHoldings(self.s2, 0.5)
                # self.Debug(f"Entry Short: Prob {mispricing_prob:.4f}")

        # --- Exit Logic ---
        else:
            # Check if probability has reverted to mean (0.5) within tolerance
            if (0.5 - self.exit_tolerance) < mispricing_prob < (0.5 + self.exit_tolerance):
                self.Liquidate()
                # self.Debug(f"Exit Mean Reversion: Prob {mispricing_prob:.4f}")

            # Stop Loss: If correlation breaks down, exit
            if tau < 0.2:
                self.Liquidate()
                # self.Debug("Exit: Correlation Breakdown")