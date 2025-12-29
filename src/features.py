import pandas as pd

def extract_features(log_entries):
    df = pd.DataFrame(log_entries)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') 
    df = df.dropna(subset=['timestamp'])

    features = df.groupby('ip').agg(
        total_requests=('ip', 'count'),
        error_rate=('status', lambda x: (x >= 400).mean()),
        unique_paths=('path', 'nunique'),
        avg_size=('size', 'mean'),
        methods=('method', 'nunique'),
        first_seen=('timestamp', 'min'),
        last_seen=('timestamp', 'max')
    )
    features['duration_sec'] = (features['last_seen'] - features['first_seen']).dt.total_seconds()
    features['requests_per_sec'] = features['total_requests'] / (features['duration_sec'] + 1)
    features = features.reset_index()

    return features, df