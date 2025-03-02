# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 11:39:23 2025

@author: Diego
"""

import os
import pandas as pd
import yfinance as yf
import datetime as dt

class DataManager:
    
    def __init__(self) -> None:
        
        self.dir       = os.path.dirname(os.path.abspath(__file__))  
        self.root_path = os.path.abspath(
            os.path.join(os.path.abspath(
                os.path.join(self.dir, os.pardir)), os.pardir))
        
        self.data_path      = os.path.join(self.root_path, "data")
        self.raw_data_path  = os.path.join(self.data_path, "RawData")
        
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        if os.path.exists(self.raw_data_path) == False: os.makedirs(self.raw_data_path)
        
        self.tickers    = ["AGG", "HYG", "FALN", "IWB", "IWV", "SPY"]
        self.start_date = dt.date(year = 1990, month = 1, day = 1)
        
    def get_yf_data(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_data_path, "ETFs.parquet")
        try:
            
            if verbose == True: print("Trying to find ETF data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")

        except:
        
            if verbose == True: print("Couldn't find data, collecting it now")    
        
            df_out = (pd.concat([
                yf.Ticker(ticker = ticker).history(
                    start = self.start_date,
                    auto_adjust = False).
                    assign(ticker = ticker)
                for ticker in self.tickers]).
                reset_index().
                rename(columns = {"Date": "date"}).
                assign(date = lambda x: pd.to_datetime(x.date).dt.date))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
def main() -> None:
    
    DataManager().get_yf_data(verbose = True)
    
if __name__ == "__main__": main()