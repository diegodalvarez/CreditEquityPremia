# Credit Equity Premia

~2.4 sharpe in sample, ~2.3 sharpe out-of-sample, ~1.7 sharpe with transaction cost

This model will trade equity returns by using fixed income risk premia as an inputted variable.
1. The equity premia will be simply be calculated as the equal weight spread between Russell indices and SPX
2. The fixed income premia will be calculated as the equal weight return of High Yield - Investment Grade and Fallen Angel - High Yield

A later goal is to build out the same model using factor returns and also exploring other equity risk premia returns such as sector and ETF style. 

The model uses a OLS regression and trades residuals. There are three models that are tested
1. Full Sample OLS Regression
2. Randomized Sample (50% sample size) 3,000 samples OLS Regression
3. Expanding Out-of-Sample Regression

Equal weight returns is used for portfolio optimization as it yielded the same returns as equal volatility contribution. 

![image](https://github.com/user-attachments/assets/abf8b8a8-5a64-48ca-9bd9-7e97ffd120af)

To optimize the signal
1. Using a z-score which generates about the same return ```(preopt signal)```
2. Threhsolding the z-score to reduce turnover ```(opt signal)```
4. Adding transaction cost the optimized signal ```(opt signal with tc)```

![image](https://github.com/user-attachments/assets/c55c6d4b-767d-4332-8947-bbc3d793c256)

Equal weight Portfolio 

![image](https://github.com/user-attachments/assets/b17fab08-ec8b-438e-955e-b5a534ffe542)
