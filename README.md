# Nginx Anomaly Detection

## Description
Inspired by ELK stack, this project focuses on detecting anomalies in Nginx access logs. It includes modules for parsing log files, engineering features, training machine learning models, and visualizing anomalies. The goal is to identify unusual patterns in the server's behavior that could indicate issues such as attacks, failures, or performance degradation.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Module Details](#module-details)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
  
---

## Installation
To set up the project on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/arnav-ag17/Nginx-Anomaly-Detection.git
   cd Nginx-Anomaly-Detection
2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
---
## Project Structure

### 📁 Model

    ├── 📂 model/
    │   ├── 📝 log_parsing.py
    │   ├── 📝 feature_engineering.py
    │   ├── 📝 model_training.py
    │   ├── 📝 visualization.py
    │   └── 📝 __init__.py

---
## 🔍 Module Details

### 1. Log Parsing Module
The `parse_log` function is responsible for extracting and cleaning data from Nginx log files. The function processes the logs to extract key metrics such as IP address, timestamp, request URL, status code, response size, and response time.
- **Regex Matching:**
  ```python
  pattern = re.compile(r'(\d+\.\d+\.\d+\.\d+)\|-\|\[(.*?)\]\|.*?\|\"(.*?)\"\|(\d+)\|(\d+)\|.*?\|.*?\|.*?\|(\d+\.\d+)\|-')
 This regular expression (regex) pattern is used to match and extract relevant fields from each line in the log file.
- **Data Cleaning:**
  ```python
  df['Status'] = pd.to_numeric(df['Status'])
  df['Response Size'] = pd.to_numeric(df['Response Size'])
  df['Response Time'] = pd.to_numeric(df['Response Time'])
  df.dropna(subset=['Response Size'], inplace=True)
  df['Response Time'].fillna(0, inplace=True)
 The extracted data is cleaned by converting relevant columns to numeric types and handling missing values.

- **Statistical Filtering:**
  ```python
  response_time_mean = df['Response Time'].mean()
  response_time_std = df['Response Time'].std()
  df = df[(df['Response Time'] >= (response_time_mean - 3 * response_time_std)) & 
        (df['Response Time'] <= (response_time_mean + 3 * response_time_std))]
 The function applies a statistical filter to remove outliers in response times, retaining only the data within three standard deviations of the mean.

### 2. Feature Engineering Module
The 'feature_engineering' function creates three distinct datframes that are responsible for detecting anomalies of specific nature. 

- **Overall Request-Based Features:**  
  This dataframe (`features_overall`) aggregates overall request data by minute, calculating the total request count, total response time, and total response size. These features are useful for identifying spikes or drops in server activity.

- **IP-Based Features:**  
  This dataframe (`features_ip`) groups the data by IP address and resamples it by minute, generating features like the number of requests per minute, the count of unique URLs accessed, and the "burstiness" of requests, which helps in detecting unusual patterns of requests from specific IPs.

- **Status Code-Based Features:**  
  This dataframe (`features_status`) groups the data by status code and resamples it by minute. It is essential for detecting anomalies related to specific HTTP status codes, such as an increase in error responses.

### 3. Train and Evaluate Module

The `train_and_evaluate` function leverages the Isolation Forest machine learning model to identify anomalies in the features generated by the feature engineering module.

- **Isolation Forest Model:**  
  The Isolation Forest model is trained on the features, with a contamination rate of 1%, meaning it expects about 1% of the data to be anomalous. The model works by isolating outliers in the dataset, making it particularly effective for anomaly detection.
  ```python
  model = IsolationForest(contamination=0.01, random_state=42)
  model.fit(X_train)

### 4. Visualization Module

The Visualization Module provides a set of functions designed to visually represent the anomalies detected by the model. Each function is tailored to plot specific features, offering insights into patterns and irregularities within the data. 

---
## Examples

You can find examples of how to use the project in the Jupyter Notebook provided (`your-notebook.ipynb`). This notebook includes exploratory data analysis and demonstrates how to use the different components of the project.

---
## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. **Fork the repository.**
2. **Create a new branch** (`git checkout -b feature-branch`).
3. **Make your changes and commit them** (`git commit -m 'Add new feature'`).
4. **Push to the branch** (`git push origin feature-branch`).
5. **Open a Pull Request.**

---
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
