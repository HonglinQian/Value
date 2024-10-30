import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.stats as sps
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import stats

class StockData:

    def __init__(self, ticker, startYear, startMonth, startDay, endYear, endMonth, endDay, lookback = 250):
        """
        Initializes the StockData object with a specified stock ticker, date range, and lookback period.
        
        Parameters:
        ticker (str): The stock ticker symbol.
        startYear (int): The start year for fetching historical data.
        startMonth (int): The start month for fetching historical data.
        startDay (int): The start day for fetching historical data.
        endYear (int): The end year for fetching historical data.
        endMonth (int): The end month for fetching historical data.
        endDay (int): The end day for fetching historical data.
        lookback (int, optional): The lookback period for calculating metrics. Default is 250.
        """
        self.ticker = ticker
        self.startDate = datetime.date(startYear, startMonth, startDay)
        self.endDate = datetime.date(endYear, endMonth, endDay)
        self.historicalData = yf.download(self.ticker, self.startDate, self.endDate)
        self.closePrices = np.array(self.historicalData['Close'])
        self.lookback = lookback

############################################
############################################
########close plot，return plot， pnl########
############################################
############################################


    def Plot(self): 
        """
        Plots the closing prices of the stock along with specific historical events.
        
        The historical events plotted are:
        - Covid Wuhan (2019-11-18)
        - Covid Europe (2020-03-11)
        - USA Elections (2020-11-05)
        - War in Ukraine (2022-02-24)
        - War in Gaza (2023-10-06)
        """
        self.historicalData.index = pd.to_datetime(self.historicalData.index, format = "%d/%m/%Y")
        self.closePrices = np.array(self.historicalData['Close'])
        plt.figure(dpi = 300)

        # covid Wuhan
        single_date_war = pd.to_datetime('18/11/2019', format="%d/%m/%Y")
        specific_date = pd.to_datetime('2019-11-18', format="%Y-%m-%d")
        index_of_date = self.historicalData.index.get_loc(specific_date)
        single_price_war = self.historicalData.Close[index_of_date]
        plt.scatter(single_date_war, single_price_war, color='darkgreen', facecolors='none', s=250, zorder=10, label='Covid Wuhan')

        # covid Europe
        single_date_war = pd.to_datetime('11/03/2020', format="%d/%m/%Y")
        specific_date = pd.to_datetime('2020-03-11', format="%Y-%m-%d")
        index_of_date = self.historicalData.index.get_loc(specific_date)
        single_price_war = self.historicalData.Close[index_of_date]
        plt.scatter(single_date_war, single_price_war, color='lightgreen', facecolors='none', s=250, zorder=10, label='Covid Europe')

        # USA elections
        single_date_war = pd.to_datetime('5/11/2020', format="%d/%m/%Y")
        specific_date = pd.to_datetime('2020-11-05', format="%Y-%m-%d")
        index_of_date = self.historicalData.index.get_loc(specific_date)
        single_price_war = self.historicalData.Close[index_of_date]
        plt.scatter(single_date_war, single_price_war, color='blue', facecolors='none', s=250, zorder=10, label='USA Elections')

        # war on Ukraine
        single_date_war = pd.to_datetime('24/02/2022', format="%d/%m/%Y")
        specific_date = pd.to_datetime('2022-02-24', format="%Y-%m-%d")
        index_of_date = self.historicalData.index.get_loc(specific_date)
        single_price_war = self.historicalData.Close[index_of_date]
        plt.scatter(single_date_war, single_price_war, color='red', facecolors='none', s=250, zorder=10, label='War in Ukraine')

        # war in Gaza
        single_date_war = pd.to_datetime('6/10/2023', format="%d/%m/%Y")
        specific_date = pd.to_datetime('2023-10-6', format="%Y-%m-%d")
        index_of_date = self.historicalData.index.get_loc(specific_date)
        single_price_war = self.historicalData.Close[index_of_date]
        plt.scatter(single_date_war, single_price_war, color='brown', facecolors='none', s=250, zorder=10, label='War in Gaza')

        plt.plot(self.historicalData.index, self.closePrices)
        plt.title('Close Price')
        plt.xlabel('Date')
        plt.ylabel('Close')
        plt.legend()
        plt.grid()
        plt.show()

    def Rates(self): # percentage return
        """
        Calculates the daily return rate as the percentage change between consecutive closing prices.
        
        Returns:
        np.array: An array of daily return rates.
        """
        return np.divide(self.closePrices[1:], self.closePrices[:len(self.closePrices) - 1]) - 1
    
    def ReturnPlot(self):
        """
        Plots the daily return rates of the stock over the specified date range.
        """
        self.historicalData.index = pd.to_datetime(self.historicalData.index, format = "%d/%m/%Y")
        historicalData = self.historicalData.drop(self.historicalData.index[-1])
        
        plt.figure(dpi = 300)
        plt.plot(historicalData.index, self.Rates())
        plt.title('Percentage Return Plot')
        plt.grid()
        plt.xlabel("Date")
        plt.ylabel("Return rate")
        plt.show()

    # def ReturnHistogram(self, bins=10):
    #     """
    #     Plots a histogram of the daily return rates with a specified number of bins.
        
    #     Parameters:
    #     bins (int, optional): The number of bins for the histogram. Default is 10.
    #     """
    #     plt.hist(self.Rates(), bins=bins)
    #     plt.show()

    def PnL(self):
        """
        Calculates the daily profit and loss (PnL) as the difference between consecutive closing prices.
        
        Returns:
        np.array: An array of daily PnL values.
        """
        return self.closePrices[1:] - self.closePrices[:len(self.closePrices) - 1]

    def SimPnL(self, index):
        """
        Simulates the PnL for a given index based on the lookback period.
        
        Parameters:
        index (int): The index at which to simulate the PnL.
        
        Returns:
        np.array or str: The simulated PnL or an error message if the index is less than the lookback period.
        """
        if index < self.lookback:
            return "Zle"
        else:
            return self.closePrices[index] * self.Rates()[index - self.lookback:index]

    def mu(self, index):
        """
        Calculates the mean of the simulated PnL over the lookback period for a given index.
        
        Parameters:
        index (int): The index at which to calculate the mean.
        
        Returns:
        float: The mean of the simulated PnL.
        """
        return np.mean(self.SimPnL(index))

    def sigma(self, index):
        """
        Calculates the standard deviation (sigma) of the simulated PnL over the lookback period for a given index.
        
        Parameters:
        index (int): The index at which to calculate the standard deviation.
        
        Returns:
        float: The standard deviation of the simulated PnL.
        """
        return np.sqrt(1 / (self.lookback - 1) * np.sum((self.SimPnL(index) - self.mu(index))**2))


############################################
############################################
###############value at risk################
############################################
############################################

    def Historical_Var(self):
        """
        Calculates the historical Value at Risk (VaR) based on historical data using the lookback period.
        
        Returns:
        np.array: An array of VaR values.
        """
        lookback = self.lookback
        Y = np.zeros(lookback) # lookback window
        V = np.zeros(len(self.closePrices) - lookback - 1) # VaR vector
        
        # int(lookback * 0.01) 计算的是 1% 分位数的位置（即排序后Pnl序列中前1%的那个位置）。
        # 选择这个位置的 PnL 值并取负，得到在置信水平为99%（即1% 分位数）的 VaR。
        # 这意味着，在99%的情况下，损失将不超过该值。
        for i in range(lookback, len(self.closePrices) - 1):
            Y = self.SimPnL(i).copy()
            Y.sort() # 将模拟的 PnL 序列从低到高排序。排序后的 PnL 序列中的较小值代表更大的潜在损失。
            V[i - lookback] = -Y[int(lookback * 0.01) + 1]
        return V

    def VarNorm(self):
        """
        Calculates the Gaussian (Normal) Value at Risk (VaR) using the mean and standard deviation.
        
        Returns:
        np.array: An array of Gaussian VaR values.
        """
        lookback = self.lookback
        V = np.zeros(len(self.closePrices) - lookback - 1)
        for i in range(lookback, len(self.closePrices) - 1):

            # 是正态分布在 1% 置信水平下的分位数。由于正态分布的对称性，这个值为负，表示极端情况下的损失水平。
            V[i - lookback] = -(self.mu(i) + self.sigma(i) * sps.norm.ppf(0.01, loc=0, scale=1))
        return V

    # def VarUnbNorm(self):
    #     """
    #     Calculates the Unbiased Gaussian (Normal) Value at Risk (VaR) using the mean and standard deviation, 
    #     adjusted for the lookback period.
        
    #     Returns:
    #     np.array: An array of unbiased Gaussian VaR values.
    #     """
    #     lookback = self.lookback
    #     V = np.zeros(len(self.closePrices) - lookback - 1)
    #     for i in range(lookback, len(self.closePrices) - 1):
    #         V[i - lookback] = -(self.mu(i) + self.sigma(i) * np.sqrt((lookback + 1)/lookback) * sps.t.ppf(0.01, df=lookback-1))
    #     return V

    def VarWeighted(self, lam=0.9):
        """
        Calculates the Exponentially Weighted Empirical Value at Risk (VaR).
        
        Parameters:
        lam (float, optional): The lambda parameter for the exponential weighting. Default is 0.9.
        
        Returns:
        np.array: An array of weighted empirical VaR values.
        """
        lookback = self.lookback
        Y = np.zeros(lookback)
        V = np.zeros(len(self.closePrices) - lookback - 1)
        weights = np.zeros(lookback)

        for i in range(lookback):
            weights[i] = (1-lam)*(lam**i)/(1-lam**lookback)
        weights = np.flip(weights) # reversed weights

        # 步骤 1: 对于每一个时间点，计算回溯期内的模拟 PnL (Y)。
        # 步骤 2: 使用 np.vstack() 将模拟 PnL 和对应的权重组合成一个二维数组，然后按 PnL 值对数组进行排序，使得损失较大的 PnL 值排在前面。
        # 步骤 3: 通过累积权重 (CumWeights) 来确定 VaR 值的位置。当累积权重达到 1% 的置信水平时，对应的 PnL 值就是 VaR。
        # 步骤 4: 记录 VaR 值（取负值，因为 VaR 通常表示的是损失）。
        for i in range(lookback, len(self.closePrices) - 1):
            Y = self.SimPnL(i).copy()
            Y = np.vstack((Y, weights))
            Y = Y[:, Y[0, :].argsort()]
            CumWeights = 0
            index = 0
            while CumWeights < 0.01:
                CumWeights += Y[1, index]
                index = index + 1
            V[i - lookback] = -Y[0, index-1]
        return V

    def MonteCarloVar(self, num_simulations=10000, time_horizon=1):
        """
        Calculates the Monte Carlo Value at Risk (VaR) using simulated price paths.
        
        Parameters:
        num_simulations (int, optional): The number of Monte Carlo simulations to run. Default is 10,000.
        time_horizon (int, optional): The time horizon over which the VaR is calculated (e.g., 1 day). Default is 1.
        
        Returns:
        np.array: An array of Monte Carlo VaR values.
        """
        lookback = self.lookback
        V = np.zeros(len(self.closePrices) - lookback - 1)
        
        for i in range(lookback, len(self.closePrices) - 1):
            # Get historical mean and standard deviation for the lookback period
            mu = self.mu(i)
            sigma = self.sigma(i)
            
            # Simulate future returns based on the mean and volatility
            simulated_returns = np.random.normal(mu * time_horizon, sigma * np.sqrt(time_horizon), num_simulations)
            
            # Calculate simulated PnL for each path based on the portfolio's current value
            current_price = self.closePrices[i]
            simulated_prices = current_price * np.exp(simulated_returns)
            simulated_pnl = simulated_prices - current_price
            
            # Sort simulated PnL and get VaR at the desired percentile (e.g., 1%)
            simulated_pnl.sort()
            VaR_index = int(num_simulations * 0.01)
            V[i - lookback] = -simulated_pnl[VaR_index]
        
        return V

############################################
############################################
###############backtesting##################
############################################
############################################

    def Backtest(self, Var):
        """
        Performs a backtest of the Value at Risk (VaR) over a 250-day lookback period to assess model accuracy.
        
        Parameters:
        Var (np.array): An array of VaR values to be tested.
        """

        # 对于每一天的盈亏值 (self.PnL()[self.lookback:])，加上相应的 VaR 值 (Var)。
        # 如果这个和（盈亏 + VaR）小于 0，则说明这一天实际损失超出了 VaR 的预期损失，将其记为一个异常，返回 1。
        # 否则返回 0，不记为异常。
        # sum(...) 对所有天数的异常进行求和，计算出总的异常天数（errors），即在给定的 VaR 下，模型未能捕捉到的异常损失天数。
        errors = sum(np.where(self.PnL()[self.lookback:] + Var < 0, 1, 0))
        print("Number of exceptions =", errors)
        if errors < 14:
            print("Green zone, model looks to be correct")
        elif errors < 24 and errors >= 14:
            print("Yellow zone, watch out!")
        else:
            print("Red zone, stay away!")

    def Backtest2(self, Var):
        """
        Performs a backtest of the Value at Risk (VaR) over a 500-day lookback period to assess model accuracy.
        
        Parameters:
        Var (np.array): An array of VaR values to be tested.
        """
        errors = sum(np.where(self.PnL()[self.lookback:] + Var < 0, 1, 0))
        print("Number of exceptions =", errors)
        if errors < 11:
            print("Green zone, model looks to be correct")
        elif errors < 20 and errors >= 11:
            print("Yellow zone, watch out!")
        else:
            print("Red zone, stay away!")

    def Backtest_daily(self, Var):
        """
        Performs a daily backtest of the Value at Risk (VaR) over a 250-day window to assess model accuracy.
        
        Parameters:
        Var (np.array): An array of VaR values to be tested.
        
        Returns:
        np.array: An array representing the proportion of exceptions over the testing period.
        """
        length = len(self.PnL()) - self.lookback - 250
        errors = np.zeros(length)
        for i in range(length):
            errors[i] = sum(np.where(self.PnL()[self.lookback+i:self.lookback+i+250] + Var[i:i+250] < 0, 1, 0))
        return errors/250 # 将异常天数除以 250，得到每个窗口内的异常比例。


############################################
############################################
###############statistic test###############
############################################
############################################

    def lags(self): # 计算在 Ljung-Box 测试中使用的滞后数量。
            """
            Calculates the number of lags to be used in the Ljung-Box test, 
            determined as the square root of the number of observations.
            
            Returns:
            int: The number of lags to use in the Ljung-Box test.
            """
            return int(np.sqrt(len(self.Rates())))

    def ShapiroWilk(self, alpha): # 使用 Shapiro-Wilk 测试检查回报率是否符合正态分布。
        """
        Performs the Shapiro-Wilk test to check for normality in the return rates.
        
        Parameters:
        alpha (float): The significance level for the test.
        """
        pValue = sps.shapiro(self.Rates()).pvalue
        print(f"P-value of Shapiro-Wilk test is: {pValue}")
        if pValue <= alpha:
            print(f"Null hypothesis of normality is rejected at the {alpha} level of significance")
        else:
            print(f"Null hypothesis of normality couldn't be rejected at the {alpha} level of significance")

 
    def LjungBox(self): # 使用 Ljung-Box 测试检查回报率的独立性（即是否存在自相关性）。
        """
        Performs the Ljung-Box test to check for independence in the return rates.
        
        Returns:
        DataFrame: Results of the Ljung-Box test.
        """
        ljung_box_test = sm.stats.acorr_ljungbox(self.Rates(), lags=[self.lags()], return_df=True)
        print(f"Ljung-Box test:\n{ljung_box_test}")

    def Levene(self): # 使用 Levene 测试检查回报率的同方差性（即各组间的方差是否相同）。
        """
        Performs the Levene test to check for homoskedasticity (constant variance) in the return rates.
        
        Returns:
        tuple: The W-statistic and p-value of the Levene test.
        """
        split_data = np.array_split(self.Rates(), 5)
        levene_test = stats.levene(split_data[0], split_data[1], split_data[2], split_data[3], split_data[4])
        print(f"Levene test: W-statistic={levene_test[0]}, p-value={levene_test[1]}")

    def Adf(self): # 使用扩展的 Dickey-Fuller (ADF) 测试检查回报率的平稳性。
        """
        Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity in the return rates.
        
        Returns:
        tuple: The ADF statistic, p-value, and critical values.
        """
        adf_test = adfuller(self.Rates())
        print(f"ADF test: ADF statistic={adf_test[0]}, p-value={adf_test[1]}")
        for key, value in adf_test[4].items():
            print(f"Critical value ({key}): {value}")
