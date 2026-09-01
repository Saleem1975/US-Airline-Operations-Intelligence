# Databricks notebook source
print("Python environment: WORKING")

print("Apache Spark version:", spark.version)

display(spark.range(5))

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     current_catalog() AS current_catalog,
# MAGIC     current_schema()  AS current_schema;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SHOW SCHEMAS IN workspace;

# COMMAND ----------

source_file = "/Volumes/workspace/default/my_files/us_airline_operations_intelligence/source/bts_ontime_2025/ontime.td.202501 (3).asc"

raw_preview = spark.read.text(source_file)

display(raw_preview.limit(5))

# COMMAND ----------

from pyspark.sql import functions as F

field_count_check = (
    raw_preview
    .select(
        F.size(F.split(F.col("value"), "\\|")).alias("field_count")
    )
    .groupBy("field_count")
    .count()
    .orderBy("field_count")
)

display(field_count_check)

# COMMAND ----------

first_record_fields = (
    raw_preview
    .limit(1)
    .select(
        F.posexplode(
            F.split(F.col("value"), "\\|")
        ).alias("position", "value")
    )
    .select(
        (F.col("position") + 1).alias("field_number"),
        F.col("value")
    )
)

display(first_record_fields)

# COMMAND ----------

first_line = raw_preview.first()["value"]
values = first_line.split("|")

print("Number of fields:", len(values))
print()

for start in range(0, len(values), 8):
    end = min(start + 8, len(values))

    chunk = [
        f"{i + 1}={repr(values[i])}"
        for i in range(start, end)
    ]

    print(" | ".join(chunk))

# COMMAND ----------

form_distribution = (
    raw_preview
    .select(
        F.split(F.col("value"), "\\|").alias("fields")
    )
    .select(
        F.col("fields").getItem(69).alias("form_type"),
        F.col("fields").getItem(70).alias("field_71")
    )
    .groupBy("form_type", "field_71")
    .count()
    .orderBy(F.desc("count"))
)

display(form_distribution)

# COMMAND ----------

form_examples = (
    raw_preview
    .select(
        F.split(F.col("value"), "\\|").alias("fields")
    )
    .select(
        F.col("fields").getItem(0).alias("field_1"),
        F.col("fields").getItem(1).alias("field_2"),
        F.col("fields").getItem(2).alias("field_3"),
        F.col("fields").getItem(3).alias("field_4"),
        F.col("fields").getItem(4).alias("field_5"),
        F.col("fields").getItem(5).alias("field_6"),
        F.col("fields").getItem(6).alias("origin"),
        F.col("fields").getItem(7).alias("destination"),
        F.col("fields").getItem(8).alias("flight_date"),
        F.col("fields").getItem(9).alias("day_of_week"),
        F.col("fields").getItem(69).alias("form_type"),
        F.col("fields").getItem(70).alias("duplicate_flag")
    )
    .dropDuplicates(["form_type", "duplicate_flag"])
    .orderBy("form_type", "duplicate_flag")
)

display(form_examples)

# COMMAND ----------

records_check = (
    raw_preview
    .select(
        F.split(F.col("value"), "\\|").alias("fields")
    )
    .select(
        F.col("fields").getItem(0).alias("marketing_carrier"),
        F.col("fields").getItem(1).alias("marketing_flight"),
        F.col("fields").getItem(2).alias("scheduled_operator"),
        F.col("fields").getItem(3).alias("scheduled_operator_flight"),
        F.col("fields").getItem(4).alias("actual_operator"),
        F.col("fields").getItem(5).alias("actual_operator_flight"),
        F.col("fields").getItem(6).alias("origin"),
        F.col("fields").getItem(7).alias("destination"),
        F.col("fields").getItem(8).alias("flight_date"),
        F.col("fields").getItem(69).alias("form_type"),
        F.col("fields").getItem(70).alias("duplicate_flag")
    )
)

display(
    records_check.filter(
        (F.col("marketing_carrier") == "AA") &
        (F.col("marketing_flight") == "3822") &
        (F.col("origin") == "DCA") &
        (F.col("destination") == "JAX") &
        (F.col("flight_date") == "20250101")
    )
)

# COMMAND ----------

swap_example = (
    records_check
    .filter(
        (F.col("actual_operator") == "9E") &
        (F.col("actual_operator_flight") == "5538") &
        (F.col("origin") == "ATL") &
        (F.col("destination") == "MGM") &
        (F.col("flight_date") == "20250116")
    )
    .select(
        "marketing_carrier",
        "marketing_flight",
        "scheduled_operator",
        "scheduled_operator_flight",
        "actual_operator",
        "actual_operator_flight",
        "form_type",
        "duplicate_flag"
    )
    .orderBy("form_type", "duplicate_flag")
)

display(swap_example)

# COMMAND ----------

swap_relationship_check = (
    records_check
    .filter(
        (F.col("origin") == "ATL") &
        (F.col("destination") == "MGM") &
        (F.col("flight_date") == "20250116") &
        (
            (
                (F.col("actual_operator") == "OO") &
                (F.col("actual_operator_flight") == "3860")
            )
            |
            (
                (F.col("scheduled_operator") == "OO") &
                (F.col("scheduled_operator_flight") == "3860")
            )
        )
    )
    .select(
        "marketing_carrier",
        "marketing_flight",
        "scheduled_operator",
        "scheduled_operator_flight",
        "actual_operator",
        "actual_operator_flight",
        "origin",
        "destination",
        "flight_date",
        "form_type",
        "duplicate_flag"
    )
    .orderBy("form_type", "duplicate_flag")
)

display(swap_relationship_check)

# COMMAND ----------

display(
    swap_relationship_check.select(
        "marketing_carrier",
        "marketing_flight",
        "scheduled_operator",
        "scheduled_operator_flight",
        "actual_operator",
        "actual_operator_flight",
        "form_type",
        "duplicate_flag"
    )
)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.airline_bronze;

# COMMAND ----------

from pyspark.sql import functions as F

bronze_january = (
    raw_preview
    .withColumnRenamed("value", "raw_record")
    .withColumn("source_file", F.lit(source_file))
    .withColumn("source_year", F.lit(2025))
    .withColumn("source_month", F.lit(1))
    .withColumn(
        "field_count",
        F.size(F.split(F.col("raw_record"), "\\|"))
    )
    .withColumn("ingestion_timestamp", F.current_timestamp())
)

display(bronze_january.limit(5))

# COMMAND ----------

display(
    bronze_january.select(
        "source_year",
        "source_month",
        "field_count",
        "ingestion_timestamp"
    ).limit(5)
)

# COMMAND ----------

(
    bronze_january.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("workspace.airline_bronze.flights_raw")
)

print("Bronze Delta table created successfully.")

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_records,
# MAGIC     MIN(field_count) AS minimum_field_count,
# MAGIC     MAX(field_count) AS maximum_field_count,
# MAGIC     COUNT(DISTINCT field_count) AS distinct_field_counts
# MAGIC FROM workspace.airline_bronze.flights_raw;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC DESCRIBE TABLE workspace.airline_bronze.flights_raw;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC DESCRIBE HISTORY workspace.airline_bronze.flights_raw;