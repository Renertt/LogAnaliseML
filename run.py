import argparse
from src.parser import parse_log_file
from src.features import extract_features
from src.models.IsolFor import train_isolation_forest, predict_anomalies, load_model
from src.models.autoEncode import train_autoencoder, detect_anomalies, AE_load_model
import polars as pl
import polars.selectors as cs
import os

def main():
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
    parser.add_argument('--sensitivity', type=float, default=3.0, help="Detector sensitivity, recommended range: 2.0-5.0. Default: 3.0(3-sigma rule)")
    parser.add_argument('--autoencode', action='store_true', help="If true - use autoencoder")
    parser.add_argument('--plot-mse', action='store_true', help="If true - plot MSE")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Log file {args.log_file} does not exist.")
        return

    if not args.train_file:
        if args.autoencode and not os.path.exists(args.AE_model_path):
            print(f"Autoencoder model file {args.AE_model_path} does not exist.")
            return
        if not args.autoencode and not os.path.exists(args.IF_model_path):
            print(f"Isolation Forest model file {args.IF_model_path} does not exist.")
            return

    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('modelsSaved', exist_ok=True)
    os.makedirs(os.path.dirname(args.output_all), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_anomalies), exist_ok=True)

    #logs = [] # Отказываемся от этой хуйни
    try:
        lf = parse_log_file(args.log_file) # LazyFrame, ещё не паршено

        logs = lf.collect(engine="streaming") # Парсим в таблицу, в больших данных может всё равно быть OOM
    except PermissionError:
        print(f"Permission denied when trying to read {args.log_file}")
        return

    if logs.is_empty():
        print("No valid logs parsed")
        return

    features, raw_df = extract_features(logs)
    if features.is_empty():
        print("No valid logs parsed")
        return

    if args.train_file:
        print("Training new model...")
        lf = parse_log_file(args.log_file)
        train_logs = lf.collect(engine="streaming")

        if train_logs.is_empty():
            print("No valid logs in train file")
            return

        train_features, _ = extract_features(train_logs)
        if train_features.is_empty():
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

    X = features.select(
        cs.numeric()
    ).drop("ip", strict=False)    # strict=False — аналог errors='ignore'

    if args.autoencode:
        preds, scores = detect_anomalies(features, model, args.sensitivity, args.plot_mse)
    else:
        preds, scores = predict_anomalies(model, X)

    features = features.with_columns (
        pl.Series("anomaly_score", scores)
    )

    is_anomaly_bool = (preds == -1)

    if args.autoencode:
        features = features.with_columns (
            pl.Series("is_anomaly", preds)
        )
    else:
        features = features.with_columns(
            pl.Series("is_anomaly", is_anomaly_bool)
        )

    features.write_csv(args.output_all)

    raw_df = raw_df.join(
        features.select(['ip', 'anomaly_score', 'is_anomaly']), 
        on='ip', 
        how='left'
    )
    anomalies_full = raw_df.filter(pl.col("is_anomaly"))
    anomalies_full.write_csv(args.output_anomalies)

    print(f"Saved {len(features)} records to {args.output_all}")
    print(f"Saved {len(anomalies_full)} anomaly records to {args.output_anomalies}")

if __name__ == '__main__':
    main()