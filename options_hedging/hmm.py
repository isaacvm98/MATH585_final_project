# region imports
from AlgorithmImports import *
# endregion
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

class HMMDemo(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2025, 1, 1)
        self.set_end_date(2025, 11, 10) 
        self.set_cash(1000000)
        self.set_benchmark("SPY")    
        self.assets = ["SPY", "TLT"]    # "TLT" as fixed income in out-of-market period (high volatility)
        
        # Add Equity ------------------------------------------------ 
        for ticker in self.assets:
            self.add_equity(ticker, Resolution.MINUTE).symbol
        
        # Regime tracking variables
        self.current_regime = None  # Current regime we're trading on
        self.predicted_regime = None  # Most recent prediction
        self.regime_day_count = 0  # Days in current predicted regime
        self.min_regime_days = 2  # Minimum days before switching
        self.min_confidence = 0.52  # Minimum confidence threshold
        
        # Set Scheduled Event Method For HMM updating
        self.schedule.on(self.date_rules.every_day(), 
            self.time_rules.before_market_close("SPY", 5), 
            self.every_day_before_market_close)
            
            
    def every_day_before_market_close(self):
        qb = self
        
        # Get history, include COVID at least
        history = qb.history(["SPY"], 252*3, Resolution.DAILY)
            
        # Get the close price daily return
        close = history['close'].unstack(level=0)
        
        # Call pct_change to obtain the daily return
        returns = close.pct_change().iloc[1:]
                
        # Initialize the HMM, then fit by the standard deviation data
        model = MarkovRegression(returns, k_regimes=2, switching_variance=True).fit()
            
        # Obtain the market regime and confidence
        regime_probs = model.smoothed_marginal_probabilities.values[-1]
        regime = regime_probs.argmax()
        confidence = regime_probs.max()
        
        # Log current prediction
        self.debug(f"Date: {self.time}, Predicted Regime: {regime}, Confidence: {confidence:.4f}")
        
        # ====== Regime Switch Logic with Persistence ======
        
        # Initialize current regime on first run
        if self.current_regime is None:
            self.current_regime = regime
            self.predicted_regime = regime
            self.regime_day_count = 1
            self.debug(f"Initial regime set to: {regime}")
        
        # Check if predicted regime matches previous prediction
        if regime == self.predicted_regime:
            # Same regime predicted, increment counter
            self.regime_day_count += 1
        else:
            # Different regime predicted, reset counter
            self.predicted_regime = regime
            self.regime_day_count = 1
            self.debug(f"New regime predicted: {regime}, resetting counter")
        
        # Check if we should switch regimes
        should_switch = (
            self.predicted_regime != self.current_regime and  # Different from current
            self.regime_day_count >= self.min_regime_days and  # Enough persistence
            confidence >= self.min_confidence  # High enough confidence
        )
        
        if should_switch:
            self.debug(f"REGIME SWITCH: {self.current_regime} → {self.predicted_regime} " +
                      f"(Days: {self.regime_day_count}, Confidence: {confidence:.4f})")
            self.current_regime = self.predicted_regime
            self.regime_day_count = 0  # Reset after switch
        
        # ====== Portfolio Allocation Based on Current Regime ======
        
        if self.current_regime == 0:
            # Low volatility regime - invest in SPY
            self.set_holdings([
                PortfolioTarget("SPY", 1.0),
                PortfolioTarget("TLT", 0.0)
            ])
            self.debug(f"Holdings: SPY 100%, TLT 0%")
        else:
            # High volatility regime - invest in TLT
            self.set_holdings([
                PortfolioTarget("SPY", 0.0),
                PortfolioTarget("TLT", 1.0)
            ])
            self.debug(f"Holdings: SPY 0%, TLT 100%")
    
    def on_end_of_algorithm(self):
        self.debug(f"Final regime: {self.current_regime}")
        self.debug(f"Final portfolio value: ${self.portfolio.total_portfolio_value:,.2f}")