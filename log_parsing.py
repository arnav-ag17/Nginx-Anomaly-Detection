import pandas as pd
import re
import numpy as np

def parse_log(file_path):
    pattern = re.compile(r'(\d+\.\d+\.\d+\.\d+)\|-\|\[(.*?)\]\|.*?\|\"(.*?)\"\|(\d+)\|(\d+)\|.*?\|.*?\|.*?\|(\d+\.\d+)\|-')
    log_data = []
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.match(line)
            if match:
                ip_address = match.group(1)
                timestamp = match.group(2)
                request = match.group(3)
                status = match.group(4)
                response_size = match.group(5)
                response_time = match.group(6)
                log_data.append([ip_address, timestamp, request, status, response_size, response_time])
    
    columns = ['IP Address', 'Timestamp', 'Request URL', 'Status', 'Response Size', 'Response Time']
    df = pd.DataFrame(log_data, columns=columns)
    
    with pd.option_context('mode.use_inf_as_na', True):
        df['Status'] = pd.to_numeric(df['Status'])
        df['Response Size'] = pd.to_numeric(df['Response Size'])
        df['Response Time'] = pd.to_numeric(df['Response Time'])
        
        df.dropna(subset=['Response Size'], inplace=True)
        df['Response Time'].fillna(0, inplace=True)
        
        response_time_mean = df['Response Time'].mean()
        response_time_std = df['Response Time'].std()
        df = df[(df['Response Time'] >= (response_time_mean - 3 * response_time_std)) & 
                (df['Response Time'] <= (response_time_mean + 3 * response_time_std))]
        
        df = df[(df['Status'] >= 200) & (df['Status'] <= 599)]
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%b/%Y:%H:%M:%S %z')
    
    return df

def agg_df(df, interval='T'):
    df_copy = df.copy()
    df_copy.set_index('Timestamp', inplace=True)

    aggregated_df = df_copy.resample(interval).agg({
        'IP Address': 'count',
        'Response Size': 'sum',
        'Response Time': 'mean',
        'Status': 'count'
    }).rename(columns={
        'IP Address': 'Request Count',
        'Status': 'Response Count'
    })

    return aggregated_df
