import polars as pl

def extract_features(df):
    if df.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    df = df.drop_nulls(subset=["timestamp"])

    features = df.group_by("ip").agg([
        pl.len().alias("total_requests"),
        (pl.col("status") >= 400).mean().alias("error_rate"),
        pl.col("path").n_unique().alias("unique_paths"),
        pl.col("size").mean().alias("avg_size"),
        pl.col("method").n_unique().alias("methods"),
        pl.col("timestamp").min().alias("first_seen"),
        pl.col("timestamp").max().alias("last_seen")
    ])

    features = features.with_columns([
        ((pl.col("last_seen") - pl.col("first_seen")).dt.total_seconds() + 1).alias("duration_sec")
    ]).with_columns([
        (pl.col("total_requests") / (pl.col("duration_sec") + 1)).alias("requests_per_sec")
    ])

    return features, df