import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Function to train and evaluate an Isolation Forest model on the engineered features
def train_and_evaluate(features, feature_cols):
    for col in feature_cols:
        if col not in features.columns:
            raise KeyError(f"'{col}' not in the DataFrame columns")
    
    X = features[feature_cols].fillna(0)
    
    X_train, X_temp = train_test_split(X, test_size=0.3, random_state=42)
    X_dev, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)
    
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_train)
    
    features_dev = X_dev.copy()
    features_dev['anomaly_score'] = model.decision_function(X_dev)
    features_dev['anomaly'] = model.predict(X_dev)

    features_test = X_test.copy()
    features_test['anomaly_score'] = model.decision_function(X_test)
    features_test['anomaly'] = model.predict(X_test)

    return model, features_dev, features_test
