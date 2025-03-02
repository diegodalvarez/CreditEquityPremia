# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 13:21:01 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
from   tqdm import tqdm
from   RiskPremia import RiskPremia

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

class SignalResid(RiskPremia):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.signal = os.path.join(self.data_path, "Signal")
        if os.path.exists(self.signal) == False: os.makedirs(self.signal)
        
        self.sample_size = 0.3
        self.num_samples = 3_000
        
    def _get_is_resid(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_tmp = (df.set_index(
            "date").
            sort_index())
        
        df_out = (sm.OLS(
            endog = df_tmp.eq_spread,
            exog  = sm.add_constant(df_tmp.credit_spread)).
            fit().
            resid.
            to_frame(name = "resid").
            assign(lag_resid = lambda x: x.resid.shift()).
            merge(right = df_tmp, how = "inner", on = ["date"]).
            assign(signal_rtn = lambda x: -np.sign(x.lag_resid) * x.eq_spread))
        
        return df_out
        
    def get_is_resid(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal, "InSampleResid.parquet")
        try:
            
            if verbose == True: print("Trying to find In Sample Resid Signal data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")

        except:
        
            if verbose == True: print("Couldn't find data, collecting it now")    
        
            df_out = (self.get_equal_weight().groupby(
                ["credit_premia", "eq_premia"]).
                apply(self._get_is_resid).
                drop(columns = ["eq_premia", "credit_premia"]).
                reset_index())
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")

        return df_out
    
    def _get_oos_resid(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp = (df.set_index(
            "date").
            sort_index())
        
        df_out = (RollingOLS(
            endog     = df_tmp.eq_spread,
            exog      = sm.add_constant(df_tmp.credit_spread),
            expanding = True).
            fit().
            params.
            rename(columns = {
                "const"        : "alpha",
                "credit_spread": "beta"}).
            dropna().
            merge(right = df_tmp, how = "inner", on = ["date"]).
            assign(
                yhat       = lambda x: (x.credit_spread * x.beta) + x.alpha,
                resid      = lambda x: x.eq_spread - x.yhat,
                lag_resid  = lambda x: x.resid.shift(),
                signal_rtn = lambda x: -np.sign(x.lag_resid) * x.eq_spread))
        
        return df_out
    
    def get_oos_resid(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal, "OutSampleResid.parquet")
        try:
            
            if verbose == True: print("Trying to find In Sample Resid Signal data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")

        except:
        
            if verbose == True: print("Couldn't find data, collecting it now")  
        
            df_out = (self.get_equal_weight().groupby(
                ["credit_premia", "eq_premia"]).
                apply(self._get_oos_resid).
                drop(columns = ["eq_premia", "credit_premia"]).
                reset_index())
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _sample_resid(self, df: pd.DataFrame, sample_size: float) -> pd.DataFrame: 
    
        df_input = df.set_index("date").sort_index()
        df_tmp   = df.sample(frac = sample_size)
        
        model = (sm.OLS(
            endog = df_tmp.eq_spread,
            exog  = sm.add_constant(df_tmp.credit_spread)).
            fit())
        
        df_out = (model.predict(
            exog = sm.add_constant(df_input.credit_spread)).
            to_frame(name = "yhat").
            merge(right = df_input, how = "inner", on = ["date"]).
            assign(
                resid      = lambda x: x.eq_spread - x.yhat,
                lag_resid  = lambda x: x.resid.shift(),
                signal_rtn = lambda x: -np.sign(x.lag_resid) * x.eq_spread))
        
        return df_out
    
    def _get_sample_resid(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (pd.concat([
            self._sample_resid(df = df, sample_size = self.sample_size).assign(sample = i + 1)
            for i in tqdm(iterable = range(self.num_samples), desc = "Working on {}".format(df.name))]))
        
        return df_out
    
    def get_sample_resid(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal, "BootstrapSampleResid.parquet")
        try:
            
            if verbose == True: print("Trying to find Boostrapped Resid Signal data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")

        except:
        
            if verbose == True: print("Couldn't find data, collecting it now")  
            
            df_out = (self.get_equal_weight().groupby(
                ["credit_premia", "eq_premia"]).
                apply(self._get_sample_resid).
                drop(columns = ["eq_premia", "credit_premia"]).
                reset_index())
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def get_sampled_sharpe(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal, "BootsrappedSharpe.parquet")
        try:
            
            if verbose == True: print("Trying to find Boostrapped Resid Signal Sharpe data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")

        except:
        
            if verbose == True: print("Couldn't find data, collecting it now")  
        
            df_out = (self.get_sample_resid()[
                ["credit_premia", "eq_premia", "signal_rtn", "sample"]].
                groupby(["credit_premia", "eq_premia", "sample"]).
                agg(["mean", "std"])
                ["signal_rtn"].
                rename(columns = {
                    "mean": "mean_rtn",
                    "std" : "std_rtn"}).
                assign(sharpe = lambda x: x.mean_rtn / x.std_rtn * np.sqrt(252)).
                reset_index())
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_min_max_median_rtn(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_ordered = (df[
            ["sample", "sharpe"]].
            sort_values("sharpe"))
        
        halfway_mark = len(df_ordered) // 2
        
        df_median = (df_ordered.reset_index(
            drop = True).
            reset_index().
            assign(index = lambda x: x.index + 1).
            query("index == @halfway_mark")
            [["sample", "sharpe"]].
            assign(sharpe_group = "median_sharpe"))
        
        df_min_max = (df[
            ["sharpe"]].
            agg(["min", "max"]).
            merge(right = df_ordered, how = "inner", on = ["sharpe"]).
            groupby("sharpe").
            head(1).
            sort_values("sharpe").
            assign(sharpe_group = ["min_sharpe", "max_sharpe"]))
        
        df_out = (pd.concat(
            [df_min_max, df_median]).
            assign(
                credit_premia = df.credit_premia.iloc[0],
                eq_premia     = df.eq_premia.iloc[0]))
        
        return df_out

    def get_min_max_median_sampled_rtn(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal, "MinMaxMedianBootstrappedSharpeRtn.parquet")
        try:
            
            if verbose == True: print("Trying to find Min Max Median Signal Sharpe data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")

        except:
        
            if verbose == True: print("Couldn't find data, collecting it now") 
        
            df_sharpe = (self.get_sampled_sharpe().assign(
                group_var = lambda x: x.credit_premia + "_" + x.eq_premia).
                groupby("group_var").
                apply(self._get_min_max_median_rtn).
                reset_index(drop = True))
            
            df_out = (self.get_sample_resid().merge(
                right = df_sharpe, 
                how   = "inner", 
                on    = ["sample", "credit_premia", "eq_premia"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
        
def main() -> None:
            
    SignalResid().get_is_resid(verbose = True)
    SignalResid().get_oos_resid(verbose = True)
    SignalResid().get_sample_resid(verbose = True)
    SignalResid().get_sampled_sharpe(verbose = True)
    SignalResid().get_min_max_median_sampled_rtn(verbose = True)
    
if __name__ == "__main__": main()