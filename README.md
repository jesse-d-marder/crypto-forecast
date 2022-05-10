### Forecasting Cryptocurrency Returns Using Machine Learning

## Repo contents:
### 1. This Readme:
- Project description with goals
- Inital hypothesis/questions on data, ideas
- Data dictionary
- Project planning
- Instructions for reproducing this project and findings
- Key findings and recommendations for this project
- Conclusion
### 2. Final report (predict_crypto.ipynb)
### 3. Acquire and Prepare modules (acquire.py, prepare.py)
### 4. Exploration & modeling notebooks (explore.ipynb, model.ipynb)
### 5. Functions to support exploration and modeling work (model.py)

---------------------------------------------------------------------
### Project Description and Goals

The goal of this project was to compare the forecasting ability of machine learning models in predicting cryptocurrency returns. The profitability of trading strategies built from the results was also evaluated. Three separate currencies were evaluated - Bitcoin, Ethereum, and Litecoin - over the same time period. 

### Initial Questions

1. Are past returns predictive of future returns for cryptocurrencies?
2. Is there a relationship between volatility and returns?
3. Are there differences in log returns based on the day of the week?

### Data Dictionary

| Variable    | Meaning     |
| ----------- | ----------- |
| btc   |  Bitcoin       |
| sigma |  Parkinson range volatility estimator (lag values 1- 7)     |
| RR    |  Relative price range (first lag)   |
| fwd_log_ret   |  the log return for one day in the future (regression target)   |
| fwd_close_positive    |  whether the next day close is higher than today's close (classification target)  |


### Project Plan

For this project I followed the data science pipeline:

Planning: I wanted to roughly follow the methodology used by Helder Sebastiao and Pedro Godinho in their article "Forecasting and trading cryptocurrencies with machine learning under changing market conditions," published 06 Jan 2021 (https://rdcu.be/cMaLB). The authors examined the predictability of three major crytocurrencies - Bitcoin, Ethereum, and Litecoin - using machine learning techniques for the period April 15, 2015 - March 03, 2019. In the original version of this project (completed as part of the Codeup Data Science curriculum) I focused solely on Bitcoin due to time constraints. This repository expanded on that initial effort to compare Bitcoin with Litecoin and Ethereum. 

Acquire: The data for this project consists of daily open, high, low, and close prices as well as volume data for Bitcoin, Ethereum, and Litecoin from 2016-08-24 - 2022-04-21 and was acquired using the Coinbase Pro API. An account and API key are required for access. Scripts to acquire this data are included in acquire.py. Without a Coinbase Pro account the data can be accessed via the csvs included in the repository. Data starts and ends at the same data for all currencies for better comparison. 

Prepare: The prepare.py module cleans the data and also contains a function to add features and targets (regression and classification) to the dataframes. Anomalous values (particularly with the low price on specific days) were explored and handled. 

Explore: The questions established in planning were analyzed using statistical tests including correlation and t-tests to confirm hypotheses about the data. Relationships between predictors and the target were explored. 

Model: Multiple classification and regression machine learning models were investigated to determine if returns could be predicted effectively. A simple trading strategy using the results of the models were used to calculate average trade profit, which was used as the primary metric for ranking the different models. The strategy bought at the close if the next day predicted return was positive or shorted if the next day predicted return was negative (always in the market). Transaction costs were not included for this iteration so average trade values represent breakeven values if commission are included. 

Delivery: This is in the form of this github repository. I am happy to talk through my results with anyone interested and collaborate on any projects related to trading.

### Steps to Reproduce
1. You will need an env.py file that contains the passphrase, secret_key and api_key of your Coinbase PRO account. Store that env file locally in the repository. Without an account you can read in the data from the csvs. 
2. Clone my repository. Confirm .gitignore is hiding your env.py file.
3. Libraries used are pandas, matplotlib, scipy, sklearn, seaborn, and numpy.
4. You should be able to run predict_crypto.ipynb.

### Key Findings 
- For conventional data split (50/30/20 train/validate/test):
    - Logistic Regression with C=1 is best for LTC (0.006330% average trade) but exhibits high dropoff in accuracy from train to validate. Thus C=0.1 would be preferred as it is the next best in terms of average percent trade but has about 1/5 of the dropoff in accuracy.
    - LassoLars with default hyperparameters is best for Ethereum, but it is only slightly above the baseline (0.003964% average trade)
    - Tweedie Regressor is best for Bitcoin, with an average percent trade of 0.003738%

### Conclusion and Recommendations
- Coming soon

### Future work
- Explore other features and feature combinations that may be predictive of returns. The original paper also included blockchain information (such as on-chain volume, active addresses, and block sizes) as inputs, though for most of the most successful models only returns, volatility, and daily dummies were actually used. 
- Test additional hyperparameters for the models and different algorithms. Based on time and resource constraints some models used in the paper could not be tested, particularly for the single-step prediction method. 
- Generate additional trading statistics for strategies based on the model results. Here I only included average trade but other metrics such as win rate, Sharpe ratio, and max drawdown are important to know prior to implementing in live trading. 
- Test ensemble methods of determining whether a trade should be taken, per the paper. These were shown to be more successful than using a single model alone to make trading decisions. 
- Test models with higher frequency data
