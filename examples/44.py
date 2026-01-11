import polars as pl

df = pl.DataFrame(
    {
        "group": ["A", "A", "A", "B", "B", "B"],
        "value": [1, 2, 3, 4, 5, 6],
    }
)
df = df.with_columns(pl.col("value").first().over("group").alias("mean_value"))
print(df)
