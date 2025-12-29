#  General

### Description

Anomaly detector for web logs. It parses raw logs, extracts features and uses Isolation Forest model to detect suspicious requests.

### Technology stack

*   **Language:** Python
*   **Libraries:**
    *   Pandas - for processing data
    *   Scikit-learn - to build IsolationForest-model
    *   joblib - saving the model

### How to run

0. Create a virtual environment (optional):
    ```
    python -m venv venv
    ```
    ```
    source venv/bin/activate    # On Linux/Mac
    source venv\Scripts\activate    # On Windows
    ```

1. Logs for model training (optional):

    I used logs from https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs

    You can use any logs for training the model

    Recommended to put them in data/raw/

    Or you can use small logs sample that is ready for use in data/sample, you can leave it there

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Run run.py
    ```
    python -m run
    ```

    Command-line arguments:
    ```
    --log-file (required) - logs for analysis
    
    --train-file - path to logs used for training the model, if not stated will used the saved model
    
    --IF-model-path - path to the saved model, if not stated modelsSaved/isolation_forest.pkl will be used
    
    --output-all - path for CSV-table with all the unique IP and their features, if not stated data/processed/allLogs.csv will be used
    
    --output-anomalies - path for CSV-table with all anomaly requests, if not stated data/processed/anomalies.csv will be used
    
    --contamination - percentage of expected anomaly logs, if not mentioned will equal 0.1 (10%)

    --estimators - number of estimators for Isolation Forest, if not mentioned will equal 200

    --n-jobs - number of jobs for Isolation Forest, if not mentioned will equal 3
    ```

    **Note:** If you want to use the default model paths and settings, you can simply run:

    ```
    python -m run --log-file data/raw/logs.csv
    ```

# Autoencoder

### Description

Anomaly detection using Autoencoder neural networks.

### Technology stack

*   **Language:** Python
*   **Libraries:**
    *   TensorFlow - for building and training the Autoencoder model
    *   Keras - high-level API for TensorFlow
    *   Pandas - for processing data
    *   NumPy - for numerical operations
    *   Matplotlib - for plotting

### How to run

0. Follow steps 0-1 from the previous section to set up your environment and install dependencies.

1. Install dependencies:
   ```
   pip install -r full_requirements.txt
   ```

2. Run run.py:
   ```
   python -m run
   ```

   Command-line arguments:
   ```
    --log-file (required) - logs for analysis

    --autoencoder - flag to use autoencoder model

    --train-file - path to logs used for training the model, if not stated will used the saved model

    --AE-model-path - path to the saved model, if not stated modelsSaved/autoencoder.keras will be used

    --output-all - path for CSV-table with all the unique IP and their features, if not stated data/processed/allLogs.csv will be used

    --output-anomalies - path for CSV-table with all anomaly requests, if not stated data/processed/anomalies.csv will be used

    --percentile - percentile for anomaly detection, if not stated will be used 90

    --plot-mse - flag to plot mean squared error during training
   ```