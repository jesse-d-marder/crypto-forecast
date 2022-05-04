import cbpro
from env import *
from datetime import datetime, timedelta
import pandas as pd
import time
import os

auth_client = cbpro.AuthenticatedClient(api_key, secret_key, passphrase)

def acquire_crypto_data(products_to_acquire, first_start_time, final_end_time, minute_bars):
    """ Acquires cryptocurrency price data using the Coinbase API"""
    
    # Checks if inputted granularity matches values allowed by API
    if minute_bars not in [60, 300, 900, 3600, 21600, 86400]:
        print("granularity must be following values:",[60, 300, 900, 3600, 21600, 86400])
        return
    
    all_product_data=[]
    product_list=[]
    
    for i, product in enumerate(products_to_acquire):
            
        
        print(f"\rAcquiring {i} / {len(products_to_acquire)}\n")
        data_list=[]
        start_time = first_start_time
        if minute_bars == 86400:
            end_time = start_time+timedelta(days=300)
        else:
            end_time = start_time+timedelta(minutes=300)
        
        while start_time<final_end_time:

            print(f"\r{product['id']} :Collecting range: {start_time}-{end_time}, data length: {len(data_list)}",end="")
            # API call requires isoformat
            start_time_iso = start_time.isoformat()
            end_time_iso = end_time.isoformat()
            try:
                # Acquire data from Coinbase
                data_list.extend(auth_client.get_product_historic_rates(product['id'],start=start_time_iso, end=end_time_iso,granularity=minute_bars))
            except:
                # Sometimes get error , pauses for 5 seconds
                print("Exception: sleep for 5")
                time.sleep(5)
                
            if minute_bars == 86400:
                start_time = start_time + timedelta(days=300)
                end_time = min(end_time + timedelta(days=300), final_end_time)
            else:
                
                start_time = start_time+timedelta(minutes=300)
                end_time = min(end_time + timedelta(minutes=300), final_end_time)
            
        time.sleep(5)
        df = pd.DataFrame(data = [line for line in data_list if line != 'message'], columns = ['time','low','high','open','close','volume'])
        # dataframe.to_csv(f'./daily_data/{product["id"]}_data.csv',index=None)
        
        df['time'] = (pd.to_datetime(df.time, unit='s'))

        df.index = df.time

        df = df.sort_index()
        
        all_product_data.append(df)
        product_list.append(product['id'])
        
    return dict(zip(product_list, all_product_data))

def get_full_product_info(desired_products):
    
    all_products=auth_client.get_products()

    return [crypto for crypto in all_products if crypto['id'] in desired_products]

def get_data_from_csv():
    
        filepaths = ['BTC_USD_daily.csv','ETH_USD_daily.csv','LTC_USD_daily.csv']
        key_names = [filename.split("_daily")[0] for filename in filepaths]
        data = [pd.read_csv(filen) for filen in filepaths]
        
        return dict(zip(key_names, data))