import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from adjustText import adjust_text
import numpy as np

# Function to plot anomalies detected in overall features
def plot_anomalies_overall(features_overall, model):
    features_overall['Timestamp'] = pd.to_datetime(features_overall['Timestamp'])
    features_overall.set_index('Timestamp', inplace=True)
    
    overall_feature_columns = ['total_request_count_per_minute', 'total_response_time_per_minute', 'total_response_size_per_minute']
    X_overall = features_overall[overall_feature_columns].fillna(0)

    features_overall['anomaly_score'] = model.decision_function(X_overall)
    features_overall['anomaly'] = model.predict(X_overall)
    plt.figure(figsize=(20, 10))
    
    plt.scatter(features_overall.index, features_overall['anomaly_score'], label='Anomaly Score', color='lightblue', alpha=0.8, s=70)
    
    anomalies = features_overall[features_overall['anomaly'] == -1]
    plt.scatter(anomalies.index, anomalies['anomaly_score'], color='red', label='Anomalies', s=100, zorder=3)
    
    texts = []
    for i, row in anomalies.iterrows():
        annotation = f"Requests: {row['total_request_count_per_minute']}"
        texts.append(plt.text(row.name, row['anomaly_score'], annotation, ha='center', color='red', fontsize=8))
    
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='red'))
    
    plt.gca().invert_yaxis()

    plt.title('Anomaly Detection in Overall Features')
    plt.xlabel('Timestamp')
    plt.ylabel('Anomaly Score')
    plt.legend(loc='upper right')

    date_form = DateFormatter("%m-%d %H:%M")
    plt.gca().xaxis.set_major_formatter(date_form)
    plt.gcf().autofmt_xdate()

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    
    plt.tight_layout()
    plt.show()

# Function to plot anomalies detected in IP address features
def plot_anomalies_ip(features, timestamp_col='Timestamp', anomaly_col='anomaly_score', ip_col='IP Address'):
    if timestamp_col not in features.columns or anomaly_col not in features.columns or ip_col not in features.columns:
        raise KeyError("One or more required columns are missing from the DataFrame.")
    
    plt.figure(figsize=(15, 8))

    # Create a scatter plot
    scatter = sns.scatterplot(
        x=features[timestamp_col],
        y=features[anomaly_col],
        hue=features['anomaly'],
        size=features[anomaly_col],
        sizes=(20, 200),
        palette={-1: 'red', 1: 'lightblue'},
        alpha=0.7,
        legend=False
    )
    
    anomaly_points = features[features['anomaly'] == -1]
    plt.scatter(anomaly_points[timestamp_col], anomaly_points[anomaly_col], color='red', s=100, label='Anomalous IP')

    plt.gca().invert_yaxis()

    plt.axhline(y=np.percentile(features[anomaly_col], 5), color='green', linestyle='--', label='95th Percentile Threshold')
    
    plt.title('Anomaly Detection in IP Addresses')
    plt.xlabel('Timestamp')
    plt.ylabel('Anomaly Score')
    plt.legend(loc='upper right')
    plt.xticks(rotation=45)
    
    texts = []
    for i, row in anomaly_points.iterrows():
        texts.append(plt.text(row[timestamp_col], row[anomaly_col], 
                              f"{row[ip_col]}",
                              fontsize=8, ha='right', color='red'))
    
    adjust_text(texts)
    
    plt.tight_layout()
    plt.show()

def plot_anomalies_status(features, timestamp_col='Timestamp', anomaly_col='anomaly_score'):
    # Check for required columns
    if timestamp_col not in features.columns or anomaly_col not in features.columns:
        raise KeyError("One or more required columns are missing from the DataFrame.")
    
    plt.figure(figsize=(20, 10))  

   
    scatter = sns.scatterplot(
        x=features[timestamp_col],
        y=features[anomaly_col],
        hue=features['anomaly'],
        size=features[anomaly_col],
        sizes=(20, 200),
        palette={-1: 'red', 1: 'lightblue'},
        alpha=0.7,
        legend=False  
    )
    
    anomaly_points = features[features['anomaly'] == -1]
    plt.scatter(anomaly_points[timestamp_col], anomaly_points[anomaly_col], color='red', s=100, label='Anomalies')
    

    texts = []
    for i, row in anomaly_points.iterrows():
        # Get status codes with non-zero values
        contributing_statuses = {col: row[col] for col in features.columns if isinstance(col, int) and row[col] > 0}
        annotation_text = ', '.join([f"{status}: {count}" for status, count in contributing_statuses.items()])
        
        texts.append(plt.text(row[timestamp_col], row[anomaly_col], annotation_text,
                     fontsize=8, ha='center', color='red'))
    
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='red')) # Iteratively adjusts the positions of text annotations to minimze overlap 

    plt.gca().invert_yaxis()  # Invert the y-axis

    plt.axhline(y=np.percentile(features[anomaly_col], 5), color='green', linestyle='--', label='95th Percentile Threshold')
    
    plt.title('Anomaly Detection in Status Codes')
    plt.xlabel('Timestamp')
    plt.ylabel('Anomaly Score')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend to the upper left, outside the plot area
    plt.xticks(rotation=45)

    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.2)
    
    plt.tight_layout()
    plt.show()
