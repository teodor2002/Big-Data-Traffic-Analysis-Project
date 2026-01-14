# utils.py
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# --- CONSTANTS ---
# Centralize your data paths here so you only change them in one place
PATHS = {
    "raw_parquet": "hdfs:///user/ubuntu/largest_parquet/",
    "meta_csv": "hdfs:///user/ubuntu/largeST/ca_meta.csv",
    "clean_enriched": "hdfs:///user/ubuntu/largest_parquet_clean_enriched/clean_enriched",
    "results_spatio_temporal": "hdfs:///user/ubuntu/analysis_results/spatio_temporal",
    "sensor_stats": "hdfs:///user/ubuntu/analysis_results/spatio_temporal/sensor_statistics",
    "adj_matrix": "/home/ubuntu/data/largeST/ca_rn_adj.npy",
    "network_clusters": "hdfs:///user/ubuntu/analysis_results/network_clusters"
}

# --- SPARK SETUP ---
def get_spark_session(app_name):
    """
    Creates a standardized SparkSession with optimized configs used across the project.
    """
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.sql.adaptive.enabled", "true")
            # These configs were repeated in multiple notebooks
            .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
            .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
            .getOrCreate())

# --- PLOTTING FUNCTIONS ---
def plot_hourly_distribution_per_year(cached_hourly_df):
    """
    Visualizes hourly traffic distribution.
    Refactored from 01_data_visualization.ipynb
    """
    hourly_pd = (
        cached_hourly_df
        .orderBy("year", "hour")
        .toPandas()
    )

    plt.figure(figsize=(12, 6))

    for yr in sorted(hourly_pd["year"].unique()):
        subset = hourly_pd[hourly_pd["year"] == yr]
        plt.plot(subset["hour"], subset["avg_value"], label=str(yr))

    plt.xlabel("Hour of Day")
    plt.ylabel("Average Traffic Value")
    plt.title("Hourly Traffic Distribution per Year")
    plt.legend()
    plt.grid(True)
    plt.show()

def print_data_quality_report(df):
    """
    Calculates and prints the count and ratio of missing/null values 
    for all columns in a Spark DataFrame.
    """
    total = df.count()
    print(f"Total Rows: {total}")
    
    # Calculate missing values per column
    missing_exprs = [
        F.sum(
            (
                (F.col(c).isNull()) | 
                (F.trim(F.col(c)) == "") | 
                (F.lower(F.trim(F.col(c))).isin("na", "null", "n/a"))
            ).cast("int")  # <--- CRITICAL FIX: Cast to int must be INSIDE sum()
        ).alias(c)
        for c in df.columns
    ]
    
    missing_df = df.select(missing_exprs)
    
    # Transpose for readability (Columns -> Rows)
    missing_long = (
        missing_df.select(
            F.explode(
                F.map_from_arrays(
                    F.array([F.lit(c) for c in df.columns]),
                    F.array([F.col(c) for c in df.columns])
                )
            ).alias("column", "missing_count")
        )
        .withColumn("missing_ratio", F.col("missing_count") / total)
        .orderBy(F.desc("missing_ratio"))
    )
    
    missing_long.show(50, truncate=False)