#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
os.chdir('/home/rene/Capstone2 b/Data')
import pandas as pd
import numpy as np
import seaborn as sns
import dask
import dask.dataframe as dd

def trans_to_datetime(ddf):
    ddf["transaction_date"]= dd.to_datetime(ddf["transaction_date"], format='%Y%m%d', errors='coerce')
    ddf["membership_expire_date"]= dd.to_datetime(ddf["membership_expire_date"], format='%Y%m%d', errors='coerce')
    return ddf

def new_transaction_features(ddf):
    ddf['auto_renew_&_cancel'] = ddf['is_auto_renew'] * ddf['is_cancel']
    ddf['days_to_expire'] = ddf['membership_expire_date'].dt.dayofyear  - ddf['transaction_date'].dt.dayofyear
    ddf['amount_unpaid'] = ddf['plan_list_price']- ddf['actual_amount_paid']
    ddf['pay_method_renewed'] = ddf['payment_method_id']* ddf['is_auto_renew']
    return ddf

def transaction_group_agg(ddf):
    ddf_grouped=ddf.groupby(by="msno").agg(
    {'payment_method_id': ['min', 'max', 'mean', 'std', 'sum'],
        'payment_plan_days': ['min', 'max', 'mean', 'std', 'sum'],
        'plan_list_price': ['min', 'max', 'mean', 'std', 'sum'],
        'actual_amount_paid': ['min', 'max', 'mean', 'std','sum'],
        'is_auto_renew': ['max',  'count'],
        'transaction_date': ['min', 'max',  'count'],
        'membership_expire_date': ['min', 'max'],
        'is_cancel': ['max', 'count'],
        'auto_renew_&_cancel': ['max', 'count'],
        'days_to_expire': ['min', 'max', 'mean', 'std', 'sum'],
        'amount_unpaid':['min', 'max', 'mean', 'std', 'sum'],
        'pay_method_renewed':['min', 'max', 'mean', 'std', 'sum']
          
    })
    grouped_col_names =[
    'min_payment_id', 'max_payment_id', 'mean_payment_id', 'std_payment_id', 'sum_payment_id',
    'min_plan_days', 'max_plan_days', 'mean_plan_days', 'std_plan_days', 'sum_plan_days',
    'min_price_list', 'max_price_list', 'mean_price_list', 'std_price_list', 'sum_price_list',
    'min_amount_paid', 'max_amount_paid', 'mean_amount_paid', 'std_amount_paid', 'sum_amount_paid',
    'max_renew', 'count_renew',
    'min_transaction_date', 'max_transaction_date', 'count_transaction_date',
    'min_expire_date', 'max_expire_date',
    'max_is_cancel', 'count_is_cancel',
    'max_auto_renew_&_cancel', 'count_auto_renew_&_cancel',
    'min_days_to_expire', 'max_days_to_expire', 'mean_days_to_expire', 'std_days_to_expire', 'sum_days_to_expire',
    'min_unpaid', 'max_unpaid', 'mean_unpaid', 'std_unpaid', 'sum_unpaid',
    'min_pmethod_renew', 'max_pmethod_renew', 'mean_pmethod_renew', 'std_pmethod_renew', 'sum_pmethod_renew'] 
    ddf_agg = dd.from_array(ddf_grouped.values,chunksize=25, columns=grouped_col_names)
    ddf_agg['msno']  =  ddf_grouped.index
    return ddf_agg

def power_on_client(x,y):
    cluster = LocalCluster(n_workers=x, threads_per_worker=y)
    client = Client(cluster)
    return client

#user logs train test data prep
def new_log_features(ddf):
    ddf['skip_ratio']= (ddf['num_25'] + ddf['num_50'] + ddf['num_75'] +  ddf['num_985'])                                                /                       (ddf['num_25'] + ddf['num_50']  + ddf['num_75']  + ddf['num_985'] + ddf['num_100'])


    ddf['skip25_ratio']= (ddf['num_25'])                            /                     (ddf['num_25']  +                       ddf['num_50']  +                       ddf['num_75']  +                       ddf['num_985'] +                       ddf['num_100'])
    
    
    ddf['skip50_ratio']= (ddf['num_50'])                            /                     (ddf['num_25']  +                       ddf['num_50']  +                       ddf['num_75']  +                       ddf['num_985'] +                       ddf['num_100'])
                                      
                                      
    ddf['skip75_ratio']= (ddf['num_75'])                            /                     (ddf['num_25']  +                       ddf['num_50']  +                       ddf['num_75']  +                       ddf['num_985'] +                       ddf['num_100'])
                                
    ddf['skip985_ratio']= (ddf['num_985'])                            /                      (ddf['num_25']  +                        ddf['num_50']  +                        ddf['num_75']  +                        ddf['num_985'] +                        ddf['num_100'])
                                       
    ddf['num100_ratio']= (ddf['num_100'])                            /                     (ddf['num_25']  +                       ddf['num_50']  +                       ddf['num_75']  +                       ddf['num_985'] +                       ddf['num_100'])
                                    
    ddf['unq_ratio']=   (ddf['num_unq'])                             /                    (ddf['num_25']  +                      ddf['num_50']  +                      ddf['num_75']  +                      ddf['num_985'] +                      ddf['num_100'])
                                   
    ddf['unq_secs_ratio']= (ddf['unq_ratio'] * ddf['total_secs'])                                   
    ddf['num100_secs_ratio']= (ddf['num_100'] * ddf['total_secs'])                                   
    ddf['skip_secs_ratio']= (ddf['skip_ratio'] * ddf['total_secs'])
    ddf['skip25_secs_ratio']= (ddf['skip25_ratio'] * ddf['total_secs'])
    ddf['skip50_secs_ratio']= (ddf['skip50_ratio'] * ddf['total_secs'])
    ddf['skip75_secs_ratio']= (ddf['skip75_ratio'] * ddf['total_secs'])
    ddf['skip985_secs_ratio']= (ddf['skip985_ratio'] * ddf['total_secs'])
    ddf['daily_listening_ratio']= ddf['total_secs'] / 86400
    return ddf



def log_to_datetime(ddf):
    ddf["date"]=dd.to_datetime(ddf["date"], format='%Y%m%d', errors='coerce')
    return ddf


def log_group_agg(ddf):
    ddf_grouped=ddf.groupby(by="msno").agg(
    {'date': ['min', 'max',  'count'],
     'num_25': ['min', 'max', 'mean', 'std', 'sum'],
     'num_50': ['min', 'max', 'mean', 'std', 'sum'],
     'num_75': ['min', 'max', 'mean', 'std', 'sum'],
     'num_985': ['min', 'max', 'mean', 'std','sum'],
     'num_100': ['min', 'max', 'mean', 'std','sum'],
     'num_unq': ['min', 'max', 'mean', 'std','sum'],
     'total_secs': ['min', 'max', 'mean', 'std','sum'],
     'skip_ratio': ['min', 'max', 'mean', 'std','sum'],
     'skip25_ratio': ['min', 'max', 'mean', 'std','sum'],
     'skip50_ratio': ['min', 'max', 'mean', 'std','sum'],
     'skip75_ratio': ['min', 'max', 'mean', 'std','sum'],
     'skip985_ratio': ['min', 'max', 'mean', 'std','sum'],
     'num100_ratio': ['min', 'max', 'mean', 'std','sum'],
     'unq_ratio': ['min', 'max', 'mean', 'std','sum'],
     'unq_secs_ratio': ['min', 'max', 'mean', 'std','sum'],
     'num100_secs_ratio': ['min', 'max', 'mean', 'std','sum'],
     'skip_secs_ratio': ['min', 'max', 'mean', 'std','sum'],
     'skip25_secs_ratio': ['min', 'max', 'mean', 'std','sum'],
     'skip50_secs_ratio': ['min', 'max', 'mean', 'std','sum'],
     'skip75_secs_ratio': ['min', 'max', 'mean', 'std','sum'],
     'skip985_secs_ratio': ['min', 'max', 'mean', 'std','sum'],
     'daily_listening_ratio': ['min', 'max', 'mean', 'std','sum']     
    })
    grouped_col_names =['min_date', 'max_date', 'count_date', 
                    'min_num_25', 'max_num_25', 'mean_num_25', 'std_num_25', 'sum_num_25',
                    'min_num_50', 'max_num_50', 'mean_num_50', 'std_num_50', 'sum_num_50',
                    'min_num_75', 'max_num_75', 'mean_num_75', 'std_num_75', 'sum_num_75',
                    'min_num_985', 'max_num_985', 'mean_num_985', 'std_num_985', 'sum_num_985',
                    'min_num_100', 'max_num_100', 'mean_num_100', 'std_num_100', 'sum_num_100',
                    'min_num_unq', 'max_num_unq', 'mean_num_unq', 'std_num_unq', 'sum_num_unq',
                    'min_total_secs', 'max_total_secs', 'mean_total_secs', 'std_total_secs', 'sum_total_secs',
                    'min_skip_ratio', 'max_skip_ratio', 'mean_skip_ratio', 'std_skip_ratio','sum_skip_ratio',
                    'min_skip25_ratio', 'max_skip25_ratio', 'mean_skip25_ratio', 'std_skip25_ratio','sum_skip25_ratio',
                    'min_skip50_ratio', 'max_skip50_ratio', 'mean_skip50_ratio', 'std_skip50_ratio','sum_skip50_ratio',
                    'min_skip75_ratio', 'max_skip75_ratio', 'mean_skip75_ratio', 'std_skip75_ratio','sum_skip75_ratio',
                    'min_skip985_ratio', 'max_skip985_ratio', 'mean_skip985_ratio', 'std_skip985_ratio','sum_skip985_ratio',
                    'min_num100_ratio', 'max_num100_ratio', 'mean_num100_ratio', 'std_num100_ratio','sum_num100_ratio',
                    'min_unq_ratio', 'max_unq_ratio', 'mean_unq_ratio', 'std_unq_ratio','sum_unq_ratio',
                    'min_unq_secs_ratio', 'max_unq_secs_ratio', 'mean_unq_secs_ratio', 'std_unq_secs_ratio','sum_unq_secs_ratio',
                    'min_num100_secs_ratio', 'max_num100_secs_ratio', 'mean_num100_secs_ratio', 'std_num100_secs_ratio','sum_num100_secs_ratio',
                    'min_skip_secs_ratio', 'max_skip_secs_ratio', 'mean_skip_secs_ratio', 'std_skip_secs_ratio','sum_skip_secs_ratio',
                    'min_skip25_secs_ratio', 'max_skip25_secs_ratio', 'mean_skip25_secs_ratio', 'std_skip25_secs_ratio','sum_skip25_secs_ratio',
                    'min_skip50_secs_ratio', 'max_skip50_secs_ratio', 'mean_skip50_secs_ratio', 'std_skip50_secs_ratio','sum_skip50_secs_ratio',
                    'min_skip75_secs_ratio', 'max_skip75_secs_ratio', 'mean_skip75_secs_ratio', 'std_skip75_secs_ratio','sum_skip75_secs_ratio',
                    'min_skip985_secs_ratio', 'max_skip985_secs_ratio', 'mean_skip985_secs_ratio', 'std_skip985_secs_ratio','sum_skip985_secs_ratio',
                    'min_daily_listening_ratio', 'max_daily_listening_ratio', 'mean_daily_listening_ratio', 'std_daily_listening_ratio','sum_daily_listening_ratio'] 
    ddf_agg = dd.from_array(ddf_grouped.values,chunksize=25, columns=grouped_col_names)
    ddf_agg['msno']  =  ddf_grouped.index
    return ddf_agg

def members_to_timedate(ddf):
    ddf["max_transaction_date"] = dd.to_datetime( ddf["max_transaction_date"], format='%Y%m%d', errors='coerce')
    ddf["registration_init_time"] = dd.to_datetime( ddf["registration_init_time"], format='%Y%m%d', errors='coerce')
    ddf["registration_init_year"] =  ddf["registration_init_time"].dt.year
    ddf["registration_init_month"] =  ddf["registration_init_time"].dt.month
    ddf["registration_init_day"] =  ddf["registration_init_time"].dt.day
    ddf["max_transaction_year"] =  ddf["max_transaction_date"].dt.year
    ddf["max_transaction_month"] =  ddf["max_transaction_date"].dt.month
    ddf["max_transaction_day"] =  ddf["max_transaction_date"].dt.day
    ddf["account_age"] =  ddf["max_transaction_date"].dt.year -  ddf["registration_init_time"].dt.year
    return ddf

def members_merge(ddf1, ddf2, ddf3):
    ddf = dd.merge(ddf1, ddf2, on='msno')
    ddf = dd.merge(ddf, ddf3, on='msno')
    return ddf

