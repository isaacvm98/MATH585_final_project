# Copula-Based Pairs Trading Algorithm (Crypto Version)
# ---------------------------------------------------
# Target: ETHUSD / BTCUSD
# Adaptations:
# 1. Switched to Resolution.Hour for crypto market speed.
# 2. Added Margin Account & Fee Model for realistic backtesting.
# 3. Adjusted Lookback Window to 720 hours (30 days) for distribution fitting.

from AlgorithmImports import *
import numpy as np

def numpy_rankdata(x):
    """Pure numpy implementation of rankdata (average method)"""
    n = len(x)
    sorted_indices = np.argsort(x)
    ranks = np.empty(n, dtype=float)
    ranks[sorted_indices] = np.arange(1, n + 1)
    return ranks

def numpy_kendalltau(x, y):
    """Pure numpy implementation of Kendall's tau"""
    n = len(x)
    if n < 2:
        return np.nan
    
    concordant = 0
    discordant = 0
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            x_diff = x[j] - x[i]
            y_diff = y[j] - y[i]
            product = x_diff * y_diff
            
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return np.nan
    
    tau = (concordant - discordant) / total_pairs
    return tau

class CopulaCryptoPairs(QCAlgorithm):
    
    # Binance 最小订单金额 (USDT)
    MIN_ORDER_VALUE = 10  # 设置为 10 USDT，比最小值 5 更保守

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)   # 设置回测开始时间
        self.SetEndDate(2024, 1, 1)     # 设置回测结束时间
        self.SetCash(100000)            # 设置初始资金 (增加至 100k 以满足最小订单要求)

        # --- 关键修改 1: 设置保证金账户和手续费 ---
        # 如果不设置这个，做空 BTC 会被拒绝，且回测结果会虚高
        # 使用 Binance 支持保证金交易
        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)

        # --- Strategy Parameters ---
        # 修改: 由于改为小时线，我们需要调整窗口大小
        # 股票用 252 (1年日线)，加密货币建议用 30天 * 24小时 = 720小时
        # 窗口太小会导致 Copula 拟合失败，窗口太大会对近期变化不敏感
        self.lookback = 720             
        
        self.cap_threshold = 0.95       # 概率 > 95% 时做空
        self.floor_threshold = 0.05     # 概率 < 5% 时做多
        self.exit_threshold = 0.5       # 均值回归目标
        self.exit_tolerance = 0.1       
        
        # --- Asset Selection (修改为 ETH/BTC) ---
        # 使用 Binance 数据源，小时级别分辨率
        self.res = Resolution.Hour
        self.pair = ["ETHUSDT", "BTCUSDT"]  # Binance 使用 USDT 交易对
        
        self.s1 = self.AddCrypto(self.pair[0], self.res, Market.Binance).Symbol # ETH
        self.s2 = self.AddCrypto(self.pair[1], self.res, Market.Binance).Symbol # BTC

        # --- Data Structures ---
        self.window_size = self.lookback + 1
        
        # 使用 RollingWindow 保存历史价格
        self.s1_window = RollingWindow[float](self.window_size)
        self.s2_window = RollingWindow[float](self.window_size)

        # 预热期
        self.SetWarmUp(self.window_size)

    def SafeLiquidate(self):
        """安全平仓：只平仓价值超过最小订单金额的持仓"""
        for symbol in [self.s1, self.s2]:
            if self.Portfolio[symbol].Invested:
                position_value = abs(self.Portfolio[symbol].HoldingsValue)
                if position_value >= self.MIN_ORDER_VALUE:
                    self.Liquidate(symbol)
                # 如果持仓价值太小，忽略（避免错误）

    def OnData(self, data: Slice):
        # 1. 数据填充 (Data Ingestion)
        if data.ContainsKey(self.s1) and data[self.s1] is not None:
            self.s1_window.Add(data[self.s1].Close)
            
        if data.ContainsKey(self.s2) and data[self.s2] is not None:
            self.s2_window.Add(data[self.s2].Close)

        # 检查预热和数据就绪状态
        if self.IsWarmingUp or not self.s1_window.IsReady or not self.s2_window.IsReady:
            return

        # 2. 提取历史数据 (Extract History)
        # 将 RollingWindow 转换为 numpy 数组并反转顺序 (Oldest -> Newest)
        prices_x = np.array(list(self.s1_window))[::-1] # ETH Prices
        prices_y = np.array(list(self.s2_window))[::-1] # BTC Prices

        # 3. 计算对数收益率 (Log Returns)
        with np.errstate(divide='ignore', invalid='ignore'):
            ret_x = np.log(prices_x[1:] / prices_x[:-1])
            ret_y = np.log(prices_y[1:] / prices_y[:-1])
        
        # 清洗数据 (去除 NaN 和 Inf)
        valid_mask = np.isfinite(ret_x) & np.isfinite(ret_y)
        ret_x = ret_x[valid_mask]
        ret_y = ret_y[valid_mask]

        if len(ret_x) < 100: return

        # 4. 拟合边缘分布 (ECDF) 并转换为均匀分布 [0, 1]
        # 这一步将收益率数据“标准化”，去除量纲影响
        u_series = numpy_rankdata(ret_x) / (len(ret_x) + 1)
        v_series = numpy_rankdata(ret_y) / (len(ret_y) + 1)

        u_today = u_series[-1] # 最新一个点的 ETH 排名分位
        v_today = v_series[-1] # 最新一个点的 BTC 排名分位

        # 5. 拟合 Copula (Clayton Copula)
        # 使用 Kendall's Tau 估算参数 Theta
        try:
            tau = numpy_kendalltau(ret_x, ret_y)
        except:
            return 
        
        # Clayton Copula 只适用于正相关 (tau > 0)
        # 如果相关性破裂或为负，暂停交易
        if tau <= 0 or tau >= 1 or np.isnan(tau):
            self.SafeLiquidate() # 风控：相关性失效时平仓
            return

        theta = (2 * tau) / (1 - tau)

        # 6. 计算错误定价指数 (Mispricing Index / Conditional Probability)
        # 计算 P(U <= u | V = v)
        try:
            # 避免幂运算溢出
            u_pow = np.power(u_today, -theta)
            v_pow = np.power(v_today, -theta)
            term_base = u_pow + v_pow - 1
            
            if term_base <= 0: return 

            part_a = np.power(term_base, (-1/theta - 1))
            part_b = np.power(v_today, (-theta - 1))
            
            mispricing_prob = part_a * part_b
            
            if np.isnan(mispricing_prob) or np.isinf(mispricing_prob):
                return
            
        except Exception:
            return

        # 7. 交易信号执行 (Execution)
        
        # 获取当前持仓
        invested_s1 = self.Portfolio[self.s1].Invested
        invested_s2 = self.Portfolio[self.s2].Invested
        is_invested = invested_s1 or invested_s2

        # --- 开仓逻辑 ---
        if not is_invested:
            # 情况 A: 概率极低 (< 0.05)
            # 意味着在给定 BTC 走势下，ETH 的表现"太差了"，概率上应该反弹
            # Action: 做多 ETH (S1), 做空 BTC (S2)
            if mispricing_prob < self.floor_threshold:
                self.SetHoldings(self.s1, 0.4)
                self.SetHoldings(self.s2, -0.4) 
                # self.Debug(f"Long Entry: P={mispricing_prob:.4f}, Tau={tau:.2f}")

            # 情况 B: 概率极高 (> 0.95)
            # 意味着在给定 BTC 走势下，ETH 的表现"太好了"，概率上应该回调
            # Action: 做空 ETH (S1), 做多 BTC (S2)
            elif mispricing_prob > self.cap_threshold:
                self.SetHoldings(self.s1, -0.4)
                self.SetHoldings(self.s2, 0.4)
                # self.Debug(f"Short Entry: P={mispricing_prob:.4f}, Tau={tau:.2f}")

        # --- 平仓逻辑 ---
        else:
            # 均值回归：当概率回到 0.5 附近时平仓
            if (self.exit_threshold - self.exit_tolerance) < mispricing_prob < (self.exit_threshold + self.exit_tolerance):
                self.SafeLiquidate()
                # self.Debug(f"Exit (Mean Reversion): P={mispricing_prob:.4f}")

            # 止损风控：如果相关性 (Tau) 突然暴跌 (< 0.2)，说明市场乱了，立即离场
            if tau < 0.2:
                self.SafeLiquidate()
                self.Debug("Exit (Correlation Breakdown)")