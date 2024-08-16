import pandas as pd

# Engineering 3 dataframes with relevant metrics
def feature_engineering(d_f):
    if 'Timestamp' not in d_f.columns:
        raise KeyError("The 'Timestamp' column is missing from the DataFrame.")

    df = d_f.copy()
    df.set_index('Timestamp', inplace=True)
    
    grouped_ip = df.groupby('IP Address').resample('T')
    
    features_ip = pd.DataFrame()
    
    features_ip['request_count_per_min'] = grouped_ip['Request URL'].count()
    features_ip['unique_url_count_per_min'] = grouped_ip['Request URL'].nunique()
    features_ip['burstiness'] = features_ip['request_count_per_min'].diff().fillna(0)
    
    min_request_count = 1
    features_ip = features_ip[features_ip['request_count_per_min'] >= min_request_count]
    
    features_ip.fillna(0, inplace=True)
    features_ip.reset_index(inplace=True)
    
    grouped_overall = df.resample('T')
    
    features_overall = pd.DataFrame()
    features_overall['total_request_count_per_minute'] = grouped_overall['Request URL'].count()
    features_overall['total_response_time_per_minute'] = grouped_overall['Response Time'].sum()
    features_overall['total_response_size_per_minute'] = grouped_overall['Response Size'].sum()
    
    features_overall.fillna(0, inplace=True)
    features_overall.reset_index(inplace=True)
    
    grouped_status = df.groupby([pd.Grouper(freq='T'), 'Status']).size().unstack(fill_value=0)
    
    features_status = pd.DataFrame(grouped_status)
    
    features_status.fillna(0, inplace=True)
    
    return features_ip, features_overall, features_status
