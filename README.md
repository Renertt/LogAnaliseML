### Description

Anomaly detector for web logs. It parses raw-logs, extacts features and uses Isolation Forest model to detect suspicious requests.

### Technology stack

*   **Language:** Python
*   **Libraries:**
    *   Pandas - for proccessing data
    *   Scikit-learn - to build IsolationForest-model
    *   joblib - saving the model

### How to run

0. Create a virual environment (optional):
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
    Reccomended to put them in data/raw/
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
    --log-file (necessarily) - logs for analysis

    --train-file - logs used for training the model, if not stated will used the saved model

    --model-path - path to the saved model, if not stated modelsSaved/isolation_forest.pkl will be used

    --output-all - path for CSV-table with all the unique IP and their features, if not stated data/processed/all_results.csv will be used
    
    --output-anomalies - path for CSV-table with all anomaly requests, if not stated data/processed/anomalies_only.csv will be used
