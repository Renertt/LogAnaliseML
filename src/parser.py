import polars as pl

def parse_log_file(file_path):
    pattern = r'(?P<ip>\d+\.\d+\.\d+\.\d+) .* \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+)\s+(?P<path>\S+)\s+(?P<protocol>\S+)" (?P<status>\d+) (?P<size>\d+|-)'

    data = pl.scan_csv(
        file_path,
        has_header=False,
        new_columns=["raw"], # запись в единый столбец
        separator="\n"
    ).select(
        pl.col("raw").str.extract_groups(pattern).alias("fields") # парс внутри поларс
    ).unnest("fields").with_columns([ #unnest раскрывает всё во множество столбцов
        pl.col("status").cast(pl.Int32),
        pl.col("size").replace("-", "0").cast(pl.Int64), # дефисы меняем ну нули, записываем в int64
        pl.col("timestamp").str.to_datetime("%d/%b/%Y:%H:%M:%S %z", strict=False)
    ])
    
    return data

# Я в ахуе..