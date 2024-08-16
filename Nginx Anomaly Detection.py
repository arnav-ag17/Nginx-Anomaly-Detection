#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[177]:


def parse_log(file_path):
    pattern = re.compile(r'(\d+\.\d+\.\d+\.\d+)\|-\|\[(.*?)\]\|.*?\|\"(.*?)\"\|(\d+)\|(\d+)\|.*?\|.*?\|.*?\|(\d+\.\d+)\|-')
    log_data = []
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.match(line)
            if match:
                #Extract relevant metrics using regex groups 
                ip_address = match.group(1)
                timestamp = match.group(2)
                request = match.group(3)
                status = match.group(4)
                response_size = match.group(5)
                response_time = match.group(6)
                
                # Store the extracted data in the log_data list
                log_data.append([ip_address, timestamp, request, status, response_size, response_time])
    
    columns = ['IP Address', 'Timestamp', 'Request URL', 'Status', 'Response Size', 'Response Time']

    df = pd.DataFrame(log_data, columns=columns)
    
    with pd.option_context('mode.use_inf_as_na', True):
        df['Status'] = pd.to_numeric(df['Status'])
        df['Response Size'] = pd.to_numeric(df['Response Size'])
        df['Response Time'] = pd.to_numeric(df['Response Time'])
        
        df.dropna(subset=['Response Size', 'Response Time'], inplace=True)
        
        response_time_mean = df['Response Time'].mean()
        response_time_std = df['Response Time'].std()
        df = df[(df['Response Time'] >= (response_time_mean - 3 * response_time_std)) & 
                (df['Response Time'] <= (response_time_mean + 3 * response_time_std))]
        
        df = df[(df['Status'] >= 200) & (df['Status'] <= 599)]
    

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%b/%Y:%H:%M:%S %z')
     
    display(df)
    return df 


log_file = './nginx.log' 

metric_df = parse_log(log_file)

                


# In[178]:


def agg_df(df, interval='T'):
    df_copy = df.copy()
    df_copy.set_index('Timestamp', inplace=True)

    aggregated_df = df_copy.resample(interval).agg({
        'IP Address': 'count',      # Number of requests
        'Response Size': 'sum',     # Total response size
        'Response Time': 'mean',    # Average response time
        'Status': 'count'           # Number of responses
    }).rename(columns={
        'IP Address': 'Request Count',
        'Status': 'Response Count'
    })

    return aggregated_df


aggregated_df = agg_df(metric_df, interval='T')

display(aggregated_df)



# In[179]:


def feature_engineering(d_f):
    if 'Timestamp' not in d_f.columns:
        raise KeyError("The 'Timestamp' column is missing from the DataFrame.")

    df = d_f.copy()
    df.set_index('Timestamp', inplace=True)
    
    # Group by IP address and resample by minute
    grouped_ip = df.groupby('IP Address').resample('T')
    
    features_ip = pd.DataFrame()
    
    features_ip['request_count_per_min'] = grouped_ip['Request URL'].count()
    features_ip['unique_url_count_per_min'] = grouped_ip['Request URL'].nunique()
    features_ip['burstiness'] = features_ip['request_count_per_min'].diff().fillna(0)
    
    # Filter out IPs with low request counts
    min_request_count = 1
    features_ip = features_ip[features_ip['request_count_per_min'] >= min_request_count]
    
    features_ip.fillna(0, inplace=True)
    features_ip.reset_index(inplace=True)
    
    # Group by timestamp for overall request counts
    grouped_overall = df.resample('T')
    
    features_overall = pd.DataFrame()
    features_overall['total_request_count_per_minute'] = grouped_overall['Request URL'].count()
    features_overall['total_unique_ips_per_minute'] = grouped_overall['IP Address'].nunique()
    
    features_overall.fillna(0, inplace=True)
    features_overall.reset_index(inplace=True)
    
    # Group by status code and resample by minute
    grouped_status = df.groupby([pd.Grouper(freq='T'), 'Status']).size().unstack(fill_value=0)
    
    features_status = pd.DataFrame(grouped_status)
    
    features_status.fillna(0, inplace=True)
    
    return features_ip, features_overall, features_status

features_ip, features_overall, features_status= feature_engineering(metric_df)

print("Feature-Engineered DataFrame for IPs:")
display(features_ip)

print("Feature-Engineered DataFrame for Overall Request Counts:")
display(features_overall)

print("Feature-Engineered DataFrame for Status Codes:")
display(features_status)


# In[180]:


from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

def train_and_evaluate(features, feature_columns):
    # Ensure feature columns exist
    for col in feature_columns:
        if col not in features.columns:
            raise KeyError(f"'{col}' not in the DataFrame columns")
    
    X = features[feature_columns].fillna(0)
    
    # Split data into training, development, and testing sets
    X_train, X_temp = train_test_split(X, test_size=0.3, random_state=42)
    X_dev, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)
    
    # Train Isolation Forest model on the training set
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_train)
    
    # Predict anomalies on the development and testing sets
    features_dev = X_dev.copy()
    features_dev['anomaly_score'] = model.decision_function(X_dev)
    features_dev['anomaly'] = model.predict(X_dev)
    
    features_test = X_test.copy()
    features_test['anomaly_score'] = model.decision_function(X_test)
    features_test['anomaly'] = model.predict(X_test)
    
    return model, features_dev, features_test
    


# In[181]:


# Train and evaluate model for IP-related features
ip_feature_columns = ['request_count_per_min', 'unique_url_count_per_min', 'burstiness']
model_ip, dev_ip, test_ip = train_and_evaluate(features_ip, ip_feature_columns)

print("Development Set Anomaly Scores and Labels for IP Features:")
display(dev_ip)

print("Testing Set Anomaly Scores and Labels for IP Features:")
display(test_ip)


# In[182]:


# Train and evaluate model for overall request count features
overall_feature_columns = ['total_request_count_per_minute', 'total_unique_ips_per_minute']
model_overall, dev_overall, test_overall = train_and_evaluate(features_overall, overall_feature_columns)

print("Development Set Anomaly Scores and Labels for Overall Request Count Features:")
display(dev_overall)

print("Testing Set Anomaly Scores and Labels for Overall Request Count Features:")
display(test_overall)


# In[183]:


# Train and evaluate model for status code features
status_feature_columns = features_status.columns
model_status, dev_status, test_status = train_and_evaluate(features_status, status_feature_columns)

print("Development Set Anomaly Scores and Labels for Status Code Features:")
display(dev_status)

print("Testing Set Anomaly Scores and Labels for Status Code Features:")
display(test_status)


# In[184]:


def plot_anomalies(features_overall, model):
    # Ensure Timestamp is in datetime format and set as index
    features_overall['Timestamp'] = pd.to_datetime(features_overall['Timestamp'])
    features_overall.set_index('Timestamp', inplace=True)
    
    # Predict anomalies
    overall_feature_columns = ['total_request_count_per_minute', 'total_unique_ips_per_minute']
    X_overall = features_overall[overall_feature_columns].fillna(0)
    features_overall['anomaly_score'] = model.decision_function(X_overall)
    features_overall['anomaly'] = model.predict(X_overall)
    
    # Plotting
    plt.figure(figsize=(14, 7))
    
    # Plot total request count per minute
    plt.plot(features_overall.index, features_overall['total_request_count_per_minute'], label='Total Request Count Per Minute', color='black', linestyle='', marker='o', markersize=2, zorder = 1)
    
    # Highlight anomalies
    anomalies = features_overall[features_overall['anomaly'] == -1]
    plt.scatter(anomalies.index, anomalies['total_request_count_per_minute'], color='red', label='Anomalies', zorder=2)
    
    # Highlight normal range
    normal_range = features_overall[features_overall['anomaly'] == 1]
    plt.fill_between(normal_range.index, 0, normal_range['total_request_count_per_minute'], color='lightblue', alpha=0.5, label='Normal Range', zorder=0)
    
    plt.xlabel('Timestamp')
    plt.ylabel('Total Request Count Per Minute')
    plt.title('Total Request Count Per Minute with Anomalies Highlighted')
    plt.legend()
    plt.show()

# Example usage
plot_anomalies(features_overall, model_overall)


# In[ ]:





# In[ ]:




