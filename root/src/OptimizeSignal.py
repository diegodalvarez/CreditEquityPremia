# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 18:34:39 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
from   SignalGenerator import SignalResid

class SignalOptimizer(SignalResid):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.optimize_path = os.path.join(self.data_path, "SignalOptimize")
        if os.path.exists(self.optimize_path) == False: os.makedirs(self.optimize_path)
        
        self.zscore_window = 20
        self.threshold     = 0.9
        self.tc_cost       = -0.0001
        
    def _get_zscore(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(
                resid_mean = lambda x: x.resid.ewm(span = window, adjust = False).mean(),
                resid_std  = lambda x: x.resid.ewm(span = window, adjust = False).std(),
                z_score    = lambda x: (x.resid - x.resid_mean) / x.resid_std,
                lag_zscore = lambda x: x.z_score.shift()).
            drop(columns = ["resid_mean", "resid_std"]))
        
        return df_out
        
    def optimize_signal(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.optimize_path, "PreOptSignal.parquet")
        try:
            
            if verbose == True: print("Trying to find In Sample Resid Signal data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")

        except:
        
            if verbose == True: print("Couldn't find data, collecting it now")  
            
            df_out = (self.get_oos_resid().assign(
                group_var = lambda x: x.credit_premia + "_" + x.eq_premia).
                groupby("group_var").
                apply(self._get_zscore, self.zscore_window).
                reset_index(drop = True).
                drop(columns = ["group_var", "alpha", "beta", "credit_spread"]).
                assign(preopt_rtn = lambda x: -np.sign(x.lag_zscore) * x.eq_spread))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def _optimize_transaction_cost(self, df: pd.DataFrame, threshold: float, tc_cost: float) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(
                abs_signal = lambda x: np.abs(x.lag_zscore),
                tmp_signal = lambda x: np.where(x.abs_signal > threshold, x.lag_zscore, np.nan),
                opt_signal = lambda x: x.tmp_signal.ffill(),
                opt_rtn    = lambda x: -np.sign(x.opt_signal) * x.eq_spread,
                signal     = lambda x: np.sign(x.opt_signal),
                lag_signal = lambda x: x.signal.shift(),
                tc_cost    = lambda x: np.where(x.signal != x.lag_signal, tc_cost, 0),
                tc_rtn     = lambda x: x.opt_rtn + x.tc_cost).
            drop(columns = ["abs_signal", "tmp_signal", "signal", "lag_signal"]))
        
        return df_out
    
    def optimize_transaction_cost(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.optimize_path, "TCOptSignal.parquet")
        try:
            
            if verbose == True: print("Trying to find In Sample Resid Signal data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")

        except:
        
            if verbose == True: print("Couldn't find data, collecting it now")  
        
            df_out = (self.optimize_signal().assign(
                group_var = lambda x: x.credit_premia + "_" + x.eq_premia).
                groupby("group_var").
                apply(self._optimize_transaction_cost, self.threshold, self.tc_cost).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out

def main() -> None:            
    
    SignalOptimizer().optimize_signal(verbose = True)    
    SignalOptimizer().optimize_transaction_cost(verbose = True)
    
if __name__ == "__main__": main()