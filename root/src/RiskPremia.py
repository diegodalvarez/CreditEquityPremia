# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 11:59:32 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
from   DataCollect import DataManager

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

class RiskPremia(DataManager):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.risk_premia = os.path.join(self.data_path, "RiskPremia")
        if os.path.exists(self.risk_premia) == False: os.makedirs(self.risk_premia)
        
        self.credit_premia_dict = {
            "hy_premia": ["HYG", "AGG"],
            "fa_premia": ["FALN", "HYG"]}
        
        self.equity_benchmarks = ["IWB", "IWV"]
        self.rolling_window    = 30
        
    def _get_equal_credit(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_out = pd.DataFrame()
        for premia in list(self.credit_premia_dict.keys()):
            
            tickers = self.credit_premia_dict[premia]
            
            df_tmp = (df[
                tickers].
                dropna().
                assign(
                    credit_spread = lambda x: x[tickers[0]] - x[tickers[1]],
                    credit_premia = premia).
                drop(columns = tickers))
            
            df_out = pd.concat([df_out, df_tmp])
            
        return df_out
    
    def _get_equal_equity(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        tickers = self.equity_benchmarks + ["SPY"]
        
        df_out = (df[
            tickers].
            reset_index().
            melt(id_vars = ["date", "SPY"], var_name = "eq_premia").
            dropna().
            assign(eq_spread = lambda x: x.value - x.SPY).
            drop(columns = ["SPY", "value"]))
        
        return df_out
        
    def get_equal_weight(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.risk_premia, "EqualWeightRiskPremia.parquet")
        try:
            
            if verbose == True: print("Trying to find Risk Premia data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")

        except:
        
            if verbose == True: print("Couldn't find data, collecting it now")    
        
            df_wider = (self.get_yf_data().pivot(
                index = "date", columns = "ticker", values = "Adj Close").
                pct_change())
            
            df_credit = self._get_equal_credit(df_wider)
            df_equity = self._get_equal_equity(df_wider)
            
            df_out = (df_credit.merge(
                right = df_equity, how = "inner", on = ["date"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
        
def main() -> None:
        
    RiskPremia().get_equal_weight(verbose = True)
        
if __name__ == "__main__": main()