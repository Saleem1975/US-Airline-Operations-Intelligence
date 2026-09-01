# Databricks notebook source
from pyspark.sql import functions as F

SOURCE_DIRECTORY = (
    "/Volumes/workspace/default/my_files/"
    "us_airline_operations_intelligence/source/bts_ontime_2025"
)

BRONZE_TABLE = "workspace.airline_bronze.flights_raw"

EXPECTED_FIELD_COUNT = 71

# COMMAND ----------

import os

source_files = sorted(
    file_name
    for file_name in os.listdir(SOURCE_DIRECTORY)
    if file_name.lower().endswith(".asc")
)

print("ASC source files found:", len(source_files))

for file_name in source_files:
    print(file_name)

# COMMAND ----------

import re

file_metadata = []

for file_name in source_files:
    match = re.search(r"(\d{6})", file_name)

    if match is None:
        raise ValueError(f"Could not identify YYYYMM in file: {file_name}")

    year_month = match.group(1)

    source_year = int(year_month[:4])
    source_month = int(year_month[4:6])

    file_metadata.append(
        {
            "file_name": file_name,
            "source_year": source_year,
            "source_month": source_month
        }
    )

for item in file_metadata:
    print(item)

# COMMAND ----------

loaded_months = {
    (row["source_year"], row["source_month"])
    for row in (
        spark.table(BRONZE_TABLE)
        .select("source_year", "source_month")
        .distinct()
        .collect()
    )
}

pending_files = [
    item
    for item in file_metadata
    if (item["source_year"], item["source_month"]) not in loaded_months
]

print("Already loaded months:")
for year, month in sorted(loaded_months):
    print(f"  {year}-{month:02d}")

print("\nFiles waiting to be ingested:")
for item in pending_files:
    print(
        f"  {item['source_year']}-{item['source_month']:02d} "
        f"-> {item['file_name']}"
    )

print("\nPending file count:", len(pending_files))

# COMMAND ----------

def ingest_monthly_file(file_info):
    file_name = file_info["file_name"]
    source_year = file_info["source_year"]
    source_month = file_info["source_month"]

    file_path = f"{SOURCE_DIRECTORY}/{file_name}"

    print(
        f"Processing {source_year}-{source_month:02d}: "
        f"{file_name}"
    )

    # Read the original BTS file as raw text
    monthly_df = (
        spark.read.text(file_path)
        .withColumnRenamed("value", "raw_record")
        .withColumn("source_file", F.lit(file_path))
        .withColumn("source_year", F.lit(source_year))
        .withColumn("source_month", F.lit(source_month))
        .withColumn(
            "field_count",
            F.size(F.split(F.col("raw_record"), "\\|"))
        )
        .withColumn(
            "ingestion_timestamp",
            F.current_timestamp()
        )
    )

    # Structural validation
    invalid_records = (
        monthly_df
        .filter(F.col("field_count") != EXPECTED_FIELD_COUNT)
        .count()
    )

    if invalid_records > 0:
        raise ValueError(
            f"{file_name} contains {invalid_records} records "
            f"that do not have {EXPECTED_FIELD_COUNT} fields."
        )

    record_count = monthly_df.count()

    # Append the validated month to Bronze
    (
        monthly_df.write
        .format("delta")
        .mode("append")
        .saveAsTable(BRONZE_TABLE)
    )

    print(
        f"Successfully loaded {record_count:,} records "
        f"for {source_year}-{source_month:02d}."
    )

# COMMAND ----------

for item in pending_files:
    ingest_monthly_file(item)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     source_year,
# MAGIC     source_month,
# MAGIC     COUNT(*) AS month_records,
# MAGIC     MIN(field_count) AS minimum_field_count,
# MAGIC     MAX(field_count) AS maximum_field_count,
# MAGIC     COUNT(DISTINCT field_count) AS distinct_field_counts,
# MAGIC     SUM(COUNT(*)) OVER () AS total_bronze_records
# MAGIC FROM workspace.airline_bronze.flights_raw
# MAGIC GROUP BY
# MAGIC     source_year,
# MAGIC     source_month
# MAGIC ORDER BY
# MAGIC     source_year,
# MAGIC     source_month;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC DESCRIBE HISTORY workspace.airline_bronze.flights_raw;

# COMMAND ----------

for item in pending_files:
    year = item["source_year"]
    month = item["source_month"]

    already_loaded = (
        spark.table(BRONZE_TABLE)
        .filter(
            (F.col("source_year") == year) &
            (F.col("source_month") == month)
        )
        .limit(1)
        .count() > 0
    )

    if already_loaded:
        print(f"Skipping {year}-{month:02d}: already loaded.")
    else:
        ingest_monthly_file(item)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     source_month,
# MAGIC     COUNT(*) AS record_count,
# MAGIC     MIN(field_count) AS min_field_count,
# MAGIC     MAX(field_count) AS max_field_count,
# MAGIC     COUNT(DISTINCT field_count) AS distinct_field_counts
# MAGIC FROM workspace.airline_bronze.flights_raw
# MAGIC WHERE source_year = 2025
# MAGIC GROUP BY source_month
# MAGIC ORDER BY source_month;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_records,
# MAGIC     COUNT(DISTINCT source_month) AS months_loaded,
# MAGIC     MIN(source_month) AS first_month,
# MAGIC     MAX(source_month) AS last_month,
# MAGIC     MIN(field_count) AS min_field_count,
# MAGIC     MAX(field_count) AS max_field_count
# MAGIC FROM workspace.airline_bronze.flights_raw
# MAGIC WHERE source_year = 2025;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC DESCRIBE HISTORY workspace.airline_bronze.flights_raw;