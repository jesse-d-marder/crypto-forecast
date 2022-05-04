import pandas as pd
import acquire
import numpy as np
import scipy.stats as stats

def prepare_crypto_data(results):
    """ Takes in a dictionary with keys as the symbols of different cryptocurrencies and values as a dataframe of open, high, low, close, and volume prices. Returns dictionary with the data prepared:
        -Sets time to datetime index
        -Truncates dataframes so all start at same date
        """
    
    first_dates = []
    for key in results.keys():
        results[key] = results[key].set_index(pd.to_datetime(results[key]['time']))
        first_dates.append(results[key].index.min())
        
    starting_date = max(first_dates)

    print(f'Max first date is {starting_date}, starting all dataframes at this day')

    for key in results.keys():
        df = results[key]
        df = df.loc[starting_date:]

        results[key] = df
        
        # Percent that High is Above Low
        # h_m_l = (df.high - df.low)/df.low
        
        # df[h_m_l>get_outlier_thresholds(h_m_l,20)[1]]
        
        # correct a low value for april 15 2017 that was due to an exchange error
        if key == "BTC_USD":
            # minute_data = acquire.acquire_crypto_data(acquire.get_full_product_info(['BTC-USD']),datetime(2017, 4, 15, 0,0,0), datetime(2017, 4, 15, 23, 59, 0), 60)
        
            # minute_data['BTC-USD']=minute_data['BTC-USD'].loc[(minute_data['BTC-USD'].index<'2017-04-15 23:00:00' )|(minute_data['BTC-USD'].index>'2017-04-15 23:50:00')]
            
            # To save time on subsequent loads this value is set based on prior exploration
            # results[key].loc['2017-04-15','low'] = minute_data['BTC-USD'].low.min()
            print("Corrected btc low data for 2017-04-15")
            results[key].loc['2017-04-15','low'] = 568.120000
        elif key == "ETH_USD":
            print("Corrected eth low data for 2017-06-21")
            results[key].loc['2017-06-21','low'] = 241.0
        

        results[key] = add_features(results[key])

    return results


def get_outlier_thresholds(s, k):
    """ Returns lower and upper thresholds based on IQR multiples """
    k = k
    iqr = stats.iqr(s)
    upper = np.quantile(s,0.75)+k*iqr
    lower = np.quantile(s,0.25)-k*iqr
    
    return lower, upper

def add_features(df):
    """ Adds target and additional features to dataframe. Returns dataframe with additional features """
    ###### TARGETS ######
    # forward 1 day log returns
    df["fwd_log_ret"] = np.log(df.close.shift(-1)) - np.log(df.close)
    # forward standard returns
    df["fwd_ret"] = df.close.shift(-1) - df.close
    # forward pct change
    df["fwd_pct_chg"] = df.close.pct_change(1).shift(-1)
    # binary positive vs negative next day return
    df["fwd_close_positive"] = df.fwd_ret>0
    
    ###### FEATURES ######
    # Pct change from yesterday
    df["pct_chg"] = df.close.pct_change()

    # Calculate lagged log returns 
    for i in range(1,8):
        df[f'log_ret_lag_{i}'] = np.log(df.close) - np.log(df.close.shift(i))
        
    # Volatility:
    # relative price range: RR
        
    df["RR"] = 2*(df.high.shift(1)-df.low.shift(1))/(df.high.shift(1)+df.low.shift(1))
    
    # range volatility estimator of Parkinson: sigma  - lags 1-7
    for i in range(1,8):
        df[f'sigma_lag_{i}'] = ((np.log(df.high.shift(i)/df.low.shift(i))**2)/(4*np.log(2)))**0.5
    
    # Day of the week shown to be significant from literature
    df["day_name"] = df.index.day_name()
    
    # Dummy variable for day name
    df = pd.get_dummies(df, columns=['day_name'])
        
    # Drop any remaining nulls (created due to lagged values)
    df = df.dropna()

    return df

def split_datasets(data, train_length, validate_length):
    """ Split crypto data into train, validate, and test sets. Takes in dictionary with keys as ticker symbols and values as dataframes """
    
    for k in data.keys():
        
        # segment out the individual dataframe
        cry = data[k]
        # determine indices of train, validate, test
        train_size = int(len(cry) * train_length)
        validate_size = int(len(cry) * validate_length)
        test_size = int(len(cry) - train_size - validate_size)
        validate_end_index = train_size + validate_size
        
        # split into train, validation, test
        train = cry[: train_size]
        validate = cry[train_size : validate_end_index]
        test = cry[validate_end_index: ]
        # print("Nulls after split train",train.isna().sum().sum())
        # print("Nulls after split val",validate.isna().sum().sum())
        # print("Nulls after split test",test.isna().sum().sum())

        data[k] = [train, validate, test]
        
    return data

def correct_spurious_low():
    h_m_l = (eth.high - eth.low)/eth.low

    redo = eth[h_m_l>get_outlier_thresholds(h_m_l,20)[1]].index

    redo[0]

    redo.to_pydatetime()[0]

    redo.to_pydatetime()[0]+timedelta(days=1)

    minute_data = acquire.acquire_crypto_data(acquire.get_full_product_info(['ETH-USD']), redo.to_pydatetime()[0], redo.to_pydatetime()[0]+timedelta(days=1), 60)

    minute_data = minute_data['ETH-USD']

    minute_data.sort_values(by='low').iloc[2:].low.min()