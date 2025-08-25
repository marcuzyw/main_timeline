import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

#plt.style.use("seaborn-v0_8")

class ML_Backtester():

    def __init__(self,symbol,start,end,tc = 0.00075):
        """ 
        
        Class for the vectorized backtesting of machine learning based trading strategies.
        
        Parameters
        =================================================================================
        symbol: str
            ticker symbol of instrument to be backtested.
        start: str
            start date of data retrieval.
        end: str
            end date of data retrieval.
        tc: float
            approximate transaction costs per trade (e.g. 0.00075 for 7.5 bps)     
        ---------------------------------------------------------------------------------

        Methods
        =================================================================================
        show_data():
            shows the data retrieved from Yahoo Finance.

        test_ml_strategy(train_ratio = 0.7,lags = 5):
            backtests a machine learning based trading strategy.

        show_results():
            shows a dataframe with the results of the backtest.

        plot_results():
            plots the results of the backtest.

        ---------------------------------------------------------------------------------

        Example
        =================================================================================
        df = ML_Backtester("AAPL",start="2010-01-01",end="2020-01-01",tc=0.00075)

        df.test_ml_strategy(train_ratio = 0.7,lags = 5)


        """
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.get_data()

    def get_data(self):
        """ 
        Retrieves data from Yahoo Finance 
        
        """
        raw = yf.download(self.symbol,self.start,self.end,multi_level_index=False)
        raw = raw["Close"].to_frame()
        raw["returns"] = np.log(raw["Close"]/raw["Close"].shift(1))
        self.data = raw


    def __repr__(self):
        return f"ML Backtester (symbol= {self.symbol} , start= {self.start} , end= {self.end})"

    # start of helper methods
    def split_data(self, start, end):
        data = self.data.loc[start:end].copy()
        return data

    def prepare_features(self,start,end):
        self.data_subset = self.split_data(start,end)
        self.feature_columns = []
        for lag in range(1,self.lags+1):
            col = f"lag{lag}"
            self.data_subset[col] = self.data_subset["returns"].shift(lag)
            self.feature_columns.append(col)
        self.data_subset.dropna(inplace=True)

    def scale_features(self,recalc=True):
        if recalc == True:
            self.means = self.data_subset[self.feature_columns].mean()
            self.stand_devs = self.data_subset[self.feature_columns].std()

        self.data_subset[self.feature_columns] = (self.data_subset[self.feature_columns]-self.means)/self.stand_devs
        
    def fit_model(self,start,end):
        self.prepare_features(start,end)
        self.scale_features(recalc = True)
        self.model.fit(self.data_subset[self.feature_columns],np.sign(self.data_subset["returns"]))
    # end of helper methods



    def show_data(self):
        """
        Shows retrieved data from Yahoo Finance with calculated returns.
        
        """
        return self.data

    

    def test_ml_strategy(self,train_ratio = 0.7,lags = 5):
        """
        Backtests a trading strategy using machine-learning.

        Parameters
        =================================================================================
        train_ratio: float
            ratio of data used for training (between 0 and 1).
        lags: int
            number of lagged returns used as features.
        ---------------------------------------------------------------------------------
     
        """

        # reset
        self.position = 0
        self.trades = 0
        self.n = 0
        self.results = None

        # assign variables
        self.lags = lags
        self.train_ratio = train_ratio

        # initialisation print out
        print(75*"-")
        print(f"Testing ML strategy | {self.symbol} | lags = {self.lags}")
        print(75*"-")

        # prepare model
        self.model = OneVsRestClassifier(LogisticRegression(C = 1e6, max_iter = 100000))

        # determine datetime for start, end and split (training and testing)
        full_data = self.data.copy()
        split_index = int(len(full_data)*self.train_ratio)
        split_date = full_data.index[split_index-1]
        train_start_date = full_data.index[0]
        test_end_date = full_data.index[-1]

        # fit model (training)
        self.fit_model(train_start_date,split_date)

        # prepare test set
        self.prepare_features(split_date,test_end_date)
        self.scale_features(recalc = False)

        # make predictions (testing)
        predict = self.model.predict(self.data_subset[self.feature_columns])
        self.data_subset["prediction"] = predict

        # calculate returns
        self.data_subset["strategy_returns"] = self.data_subset["prediction"]*self.data_subset["returns"]

        # calculate number of trades in each bar
        self.data_subset["trades"] = self.data_subset["prediction"].diff().fillna(0).abs()

        # subtract transaction/trading costs from returns
        self.data_subset["strategy_returns"] = self.data_subset["strategy_returns"] - self.data_subset["trades"]*self.tc

        # calculate cumulative returns for strategy
        self.data_subset["creturns"] = self.data_subset["returns"].cumsum().apply(np.exp)
        self.data_subset["cstrategy"] = self.data_subset["strategy_returns"].cumsum().apply(np.exp)
        self.results = self.data_subset

        # calculate performamce
        perf = self.results["cstrategy"].iloc[-1]
        outperf = perf - self.results["creturns"].iloc[-1]

        print(75*" ")
        print(75*"-")
        print(f"Finished testing ML strategy | {self.symbol} | lags = {self.lags}")
        print(f"Performance of Strategy = {round(perf,6)}, Outperformance = {round(outperf,6)}")
        print(75*"-")

        return round(perf,6), round(outperf,6)
    
    def show_results(self):
        """
        Shows a dataframe with the results of the backtest.

        """

        return self.results


    def plot_results(self):
        """
        Plots the results of backtest strategy.

        """

        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = f"Logistic Regression: {self.symbol} | TC = {self.tc}"
            self.results[["creturns","cstrategy"]].plot(title=title,figsize = (12,8))