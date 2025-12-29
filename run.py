import argparse
from src.parser import parse_log_line
from src.features import extract_features
from src.models.IsolFor import train_isolation_forest, predict_anomalies, load_model
from src.models.autoEncode import train_autoencoder, detect_anomalies, AE_load_model
import joblib
import os

def main():
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('modelsSaved', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', required=True, help="File to analyze for anomalies")
    parser.add_argument('--train-file', help="File to train the model on. If not provided, uses saved model.")
    parser.add_argument('--IF-model-path', default='modelsSaved/isolation_forest.pkl', help="Path to saved model")
    parser.add_argument('--AE-model-path', default='modelsSaved/autoencoder.keras', help="Path to saved model")
    parser.add_argument('--output-all', default='data/processed/allLog.csv', help="Path for CSV table for all unique IP")
    parser.add_argument('--output-anomalies', default='data/processed/anomalies.csv', help="Path for CSV table for all anomaly requests")
    parser.add_argument('--contamination', type=float, default=0.1, help="Percent of anomaly logs")
    parser.add_argument('--estimators', type=int, default=200, help="Number of estimators for Isolation Forest")
    parser.add_argument('--n-jobs', type=int, default=3, help="Number of jobs for Isolation Forest")
    parser.add_argument('--threshold', type=float, default=0.1, help="Threshold for anomaly detection")
    parser.add_argument('--autoencode', action='store_true', help="If true - use autoencoder")
    parser.add_argument('--plot-mse', action='store_true', help="If true - plot MSE")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Log file {args.log_file} does not exist.")
        return

    logs = []
    with open(args.log_file) as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                logs.append(parsed)

    if not logs:
        print("No valid logs parsed")
        return

    features, raw_df = extract_features(logs)
    if features.empty:
        print("No valid logs parsed")
        return

    if args.train_file:
        print("Training new model...")
        train_logs = []
        with open(args.train_file) as f:
            for line in f:
                parsed = parse_log_line(line)
                if parsed:
                    train_logs.append(parsed)

        if not train_logs:
            print("No valid logs in train file")
            return

        train_features, _ = extract_features(train_logs)
        if train_features.empty:
            print("No valid logs in train file")
            return

        X_train = train_features.select_dtypes(include='number').drop(columns=['ip'], errors='ignore')

        if args.autoencode:
            model = train_autoencoder(X_train)
        else:
            model = train_isolation_forest(
                X_train,
                model_path=args.IF_model_path,
                contamination=args.contamination,
                estimators=args.estimators,
                n_jobs=args.n_jobs
            )

    else:
        if args.autoencode:
            print("Loading existing model...")
            model = AE_load_model(args.AE_model_path)
        else:
            print("Loading existing model...")
            model = load_model(args.IF_model_path)

    X = features.select_dtypes(include='number').drop(columns=['ip'], errors='ignore')

    if args.autoencode:
        preds, scores = detect_anomalies(features, model, args.threshold, args.plot_mse)
    else:
        preds, scores = predict_anomalies(model, X)

    features['anomaly_score'] = scores

    if args.autoencode:
        features['is_anomaly'] = preds
    else:
        features['is_anomaly'] = preds == -1

    features.to_csv(args.output_all, index=False)

    raw_df = raw_df.merge(features[['ip', 'anomaly_score', 'is_anomaly']], on='ip', how='left')
    anomalies_full = raw_df[raw_df['is_anomaly']].copy()
    anomalies_full.to_csv(args.output_anomalies, index=False)

    print(f"Saved {len(features)} records to {args.output_all}")
    print(f"Saved {len(anomalies_full)} anomaly records to {args.output_anomalies}")

if __name__ == '__main__':
    main()