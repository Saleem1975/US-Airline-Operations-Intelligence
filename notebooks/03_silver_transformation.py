# Databricks notebook source
from pyspark.sql import functions as F

BRONZE_TABLE = "workspace.airline_bronze.flights_raw"

SILVER_SCHEMA = "workspace.airline_silver"
SILVER_TABLE = "workspace.airline_silver.flights"

bronze_df = spark.table(BRONZE_TABLE)

print(f"Bronze records: {bronze_df.count():,}")
print("Bronze columns:", len(bronze_df.columns))

# COMMAND ----------

RAW_FIELD_NAMES = [
    "marketing_carrier_raw",
    "marketing_flight_number_raw",
    "scheduled_operating_carrier_raw",
    "scheduled_operating_flight_number_raw",
    "operating_carrier_raw",
    "operating_flight_number_raw",

    "origin_raw",
    "destination_raw",
    "flight_date_raw",
    "day_of_week_raw",

    "scheduled_departure_oag_raw",
    "scheduled_departure_crs_raw",
    "actual_gate_departure_raw",

    "scheduled_arrival_oag_raw",
    "scheduled_arrival_crs_raw",
    "actual_gate_arrival_raw",

    "departure_schedule_diff_raw",
    "arrival_schedule_diff_raw",

    "scheduled_elapsed_minutes_raw",
    "actual_elapsed_minutes_raw",
    "departure_delay_minutes_raw",
    "arrival_delay_minutes_raw",
    "elapsed_time_diff_raw",

    "wheels_off_raw",
    "wheels_on_raw",
    "tail_number_raw",

    "taxi_out_minutes_raw",
    "taxi_in_minutes_raw",
    "air_time_minutes_raw",

    "cancellation_code_raw",

    "carrier_delay_minutes_raw",
    "weather_delay_minutes_raw",
    "nas_delay_minutes_raw",
    "security_delay_minutes_raw",
    "late_aircraft_delay_minutes_raw",

    "first_gate_departure_raw",
    "total_gate_return_ground_minutes_raw",
    "longest_gate_return_ground_minutes_raw",

    "diversion_code_raw",

    "diverted_airport_1_raw",
    "diverted_wheels_on_1_raw",
    "diverted_ground_time_1_raw",
    "diverted_longest_ground_time_1_raw",
    "diverted_wheels_off_1_raw",
    "diverted_tail_number_1_raw",

    "diverted_airport_2_raw",
    "diverted_wheels_on_2_raw",
    "diverted_ground_time_2_raw",
    "diverted_longest_ground_time_2_raw",
    "diverted_wheels_off_2_raw",
    "diverted_tail_number_2_raw",

    "diverted_airport_3_raw",
    "diverted_wheels_on_3_raw",
    "diverted_ground_time_3_raw",
    "diverted_longest_ground_time_3_raw",
    "diverted_wheels_off_3_raw",
    "diverted_tail_number_3_raw",

    "diverted_airport_4_raw",
    "diverted_wheels_on_4_raw",
    "diverted_ground_time_4_raw",
    "diverted_longest_ground_time_4_raw",
    "diverted_wheels_off_4_raw",
    "diverted_tail_number_4_raw",

    "diverted_airport_5_raw",
    "diverted_wheels_on_5_raw",
    "diverted_ground_time_5_raw",
    "diverted_longest_ground_time_5_raw",
    "diverted_wheels_off_5_raw",
    "diverted_tail_number_5_raw",

    "form_type_raw",
    "duplicate_flag_raw"
]

assert len(RAW_FIELD_NAMES) == 71

silver_fields = F.split(
    F.col("raw_record"),
    r"\|",
    -1
)

silver_raw_df = bronze_df.select(
    "source_file",
    "source_year",
    "source_month",
    "ingestion_timestamp",

    *[
        silver_fields.getItem(i).alias(field_name)
        for i, field_name in enumerate(RAW_FIELD_NAMES)
    ]
)

print("Mapped BTS fields:", len(RAW_FIELD_NAMES))
print("Silver staging columns:", len(silver_raw_df.columns))

# COMMAND ----------

def blank_to_null(column_name):
    return F.when(
        F.trim(F.col(column_name)) == "",
        F.lit(None)
    ).otherwise(
        F.trim(F.col(column_name))
    )


silver_core_df = (
    silver_raw_df
    .select(
        # Lineage
        "source_file",
        "source_year",
        "source_month",
        "ingestion_timestamp",

        # Flight identity
        blank_to_null("marketing_carrier_raw")
            .alias("marketing_carrier"),

        blank_to_null("marketing_flight_number_raw")
            .alias("marketing_flight_number"),

        blank_to_null("scheduled_operating_carrier_raw")
            .alias("scheduled_operating_carrier"),

        blank_to_null("scheduled_operating_flight_number_raw")
            .alias("scheduled_operating_flight_number"),

        blank_to_null("operating_carrier_raw")
            .alias("operating_carrier"),

        blank_to_null("operating_flight_number_raw")
            .alias("operating_flight_number"),

        blank_to_null("origin_raw").alias("origin"),
        blank_to_null("destination_raw").alias("destination"),

        F.to_date(
            F.col("flight_date_raw"),
            "yyyyMMdd"
        ).alias("flight_date"),

        F.col("day_of_week_raw")
            .cast("int")
            .alias("day_of_week"),

        # Reporting metadata
        blank_to_null("form_type_raw")
            .alias("form_type"),

        blank_to_null("duplicate_flag_raw")
            .alias("duplicate_flag"),

        # Cancellation / diversion status
        blank_to_null("cancellation_code_raw")
            .alias("cancellation_code"),

        F.col("diversion_code_raw")
            .cast("int")
            .alias("diversion_code"),

        # Diversion airports needed for consistency logic
        blank_to_null("diverted_airport_1_raw")
            .alias("diverted_airport_1"),

        blank_to_null("diverted_airport_2_raw")
            .alias("diverted_airport_2"),

        blank_to_null("diverted_airport_3_raw")
            .alias("diverted_airport_3"),

        blank_to_null("diverted_airport_4_raw")
            .alias("diverted_airport_4"),

        blank_to_null("diverted_airport_5_raw")
            .alias("diverted_airport_5")
    )

    # Reported cancellation
    .withColumn(
        "reported_cancellation_flag",
        F.col("cancellation_code").isin("A", "B", "C", "D")
    )

    # Special BTS air-return/cancellation state
    .withColumn(
        "air_return_cancelled_flag",
        F.col("diversion_code") == 9
    )

    # Operational cancellation:
    # reported cancellation OR BTS code 9
    .withColumn(
        "operational_cancellation_flag",
        F.col("reported_cancellation_flag") |
        F.col("air_return_cancelled_flag")
    )

    # Human-readable cancellation reason
    .withColumn(
        "cancellation_reason",
        F.when(F.col("cancellation_code") == "A", "Carrier")
         .when(F.col("cancellation_code") == "B", "Weather")
         .when(F.col("cancellation_code") == "C", "NAS")
         .when(F.col("cancellation_code") == "D", "Security")
         .when(F.col("air_return_cancelled_flag"), "Unknown / not reported")
         .otherwise(F.lit(None))
    )

    # Detect whether any diversion-event airport was reported
    .withColumn(
        "has_diversion_event_data",
        F.col("diverted_airport_1").isNotNull() |
        F.col("diverted_airport_2").isNotNull() |
        F.col("diverted_airport_3").isNotNull() |
        F.col("diverted_airport_4").isNotNull() |
        F.col("diverted_airport_5").isNotNull()
    )

    # The one consistency anomaly we discovered
    .withColumn(
        "diversion_consistency_anomaly_flag",
        (F.col("diversion_code") == 0) &
        F.col("has_diversion_event_data")
    )

    # Duplicate = Y is superseded for canonical operational counts
    .withColumn(
        "canonical_operation_flag",
        F.col("duplicate_flag") == "N"
    )
)

print(f"Silver core records: {silver_core_df.count():,}")
print("Silver core columns:", len(silver_core_df.columns))

# COMMAND ----------

silver_status_validation = (
    silver_core_df
    .agg(
        F.count("*").alias("total_records"),

        F.sum(
            F.when(
                F.col("reported_cancellation_flag"),
                1
            ).otherwise(0)
        ).alias("reported_cancellations"),

        F.sum(
            F.when(
                F.col("air_return_cancelled_flag"),
                1
            ).otherwise(0)
        ).alias("air_return_code_9"),

        F.sum(
            F.when(
                F.col("operational_cancellation_flag"),
                1
            ).otherwise(0)
        ).alias("operational_cancellations"),

        F.sum(
            F.when(
                F.col("cancellation_reason") == "Unknown / not reported",
                1
            ).otherwise(0)
        ).alias("cancelled_reason_unknown"),

        F.sum(
            F.when(
                F.col("has_diversion_event_data"),
                1
            ).otherwise(0)
        ).alias("records_with_diversion_event_data"),

        F.sum(
            F.when(
                F.col("diversion_consistency_anomaly_flag"),
                1
            ).otherwise(0)
        ).alias("diversion_consistency_anomalies"),

        F.sum(
            F.when(
                F.col("canonical_operation_flag"),
                1
            ).otherwise(0)
        ).alias("canonical_operations"),

        F.sum(
            F.when(
                ~F.col("canonical_operation_flag"),
                1
            ).otherwise(0)
        ).alias("superseded_duplicate_records")
    )
)

result = silver_status_validation.first()

for field_name, value in result.asDict().items():
    print(f"{field_name}: {value:,}")

# COMMAND ----------

def int_or_null(column_name):
    return F.expr(
        f"try_cast(nullif(trim(`{column_name}`), '') AS INT)"
    )


numeric_field_mapping = {
    "departure_schedule_diff_raw": "departure_schedule_diff_minutes",
    "arrival_schedule_diff_raw": "arrival_schedule_diff_minutes",
    "scheduled_elapsed_minutes_raw": "scheduled_elapsed_minutes",
    "actual_elapsed_minutes_raw": "actual_elapsed_minutes",
    "departure_delay_minutes_raw": "departure_delay_minutes",
    "arrival_delay_minutes_raw": "arrival_delay_minutes",
    "elapsed_time_diff_raw": "elapsed_time_diff_minutes",

    "taxi_out_minutes_raw": "taxi_out_minutes",
    "taxi_in_minutes_raw": "taxi_in_minutes",
    "air_time_minutes_raw": "air_time_minutes",

    "carrier_delay_minutes_raw": "carrier_delay_minutes",
    "weather_delay_minutes_raw": "weather_delay_minutes",
    "nas_delay_minutes_raw": "nas_delay_minutes",
    "security_delay_minutes_raw": "security_delay_minutes",
    "late_aircraft_delay_minutes_raw": "late_aircraft_delay_minutes",

    "total_gate_return_ground_minutes_raw":
        "total_gate_return_ground_minutes",

    "longest_gate_return_ground_minutes_raw":
        "longest_gate_return_ground_minutes",

    "diversion_code_raw": "diversion_code"
}

silver_numeric_df = silver_raw_df.select(
    "source_year",
    "source_month",

    *[
        int_or_null(raw_name).alias(typed_name)
        for raw_name, typed_name in numeric_field_mapping.items()
    ]
)

print("Numeric fields typed:", len(numeric_field_mapping))
print("Numeric staging columns:", len(silver_numeric_df.columns))

# COMMAND ----------

numeric_cast_checks = []

for raw_name, typed_name in numeric_field_mapping.items():

    check = (
        silver_raw_df
        .select(
            F.trim(F.col(raw_name)).alias("raw_value"),
            int_or_null(raw_name).alias("typed_value")
        )
        .agg(
            F.sum(
                F.when(
                    (F.col("raw_value").isNotNull()) &
                    (F.col("raw_value") != "") &
                    F.col("typed_value").isNull(),
                    1
                ).otherwise(0)
            ).alias("failed_casts")
        )
        .first()
    )

    numeric_cast_checks.append(
        (
            raw_name,
            typed_name,
            check["failed_casts"]
        )
    )

numeric_cast_validation_df = spark.createDataFrame(
    numeric_cast_checks,
    [
        "raw_field",
        "typed_field",
        "failed_casts"
    ]
)

display(
    numeric_cast_validation_df
    .orderBy(F.desc("failed_casts"))
)

# COMMAND ----------

numeric_cast_summary = (
    numeric_cast_validation_df
    .agg(
        F.count("*").alias("numeric_fields_checked"),
        F.sum("failed_casts").alias("total_failed_casts"),
        F.max("failed_casts").alias("maximum_failed_casts_in_any_field")
    )
)

result = numeric_cast_summary.first()

print("Numeric fields checked:", result["numeric_fields_checked"])
print("Total failed casts:", result["total_failed_casts"])
print(
    "Maximum failed casts in any field:",
    result["maximum_failed_casts_in_any_field"]
)

# COMMAND ----------

def normalize_bts_clock(column_name):
    """
    Convert BTS HHMM values to normalized HH:mm strings.

    Rules established from data-quality profiling:
      blank -> NULL
      0     -> NULL
      2400  -> 00:00
      other valid HHMM -> HH:mm
    """

    value = F.expr(
        f"try_cast(nullif(trim(`{column_name}`), '') AS INT)"
    )

    return (
        F.when(
            value.isNull() | (value == 0),
            F.lit(None)
        )
        .when(
            value == 2400,
            F.lit("00:00")
        )
        .otherwise(
            F.format_string(
                "%02d:%02d",
                F.floor(value / 100),
                value % 100
            )
        )
    )


clock_field_mapping = {
    "scheduled_departure_oag_raw": "scheduled_departure_oag",
    "scheduled_departure_crs_raw": "scheduled_departure_crs",
    "actual_gate_departure_raw": "actual_gate_departure",

    "scheduled_arrival_oag_raw": "scheduled_arrival_oag",
    "scheduled_arrival_crs_raw": "scheduled_arrival_crs",
    "actual_gate_arrival_raw": "actual_gate_arrival",

    "wheels_off_raw": "wheels_off",
    "wheels_on_raw": "wheels_on"
}


silver_clock_df = silver_raw_df.select(
    "source_year",
    "source_month",

    *[
        normalize_bts_clock(raw_name).alias(clean_name)
        for raw_name, clean_name in clock_field_mapping.items()
    ],

    *[
        (
            F.expr(
                f"try_cast(nullif(trim(`{raw_name}`), '') AS INT)"
            ) == 2400
        ).alias(f"{clean_name}_was_2400")
        for raw_name, clean_name in clock_field_mapping.items()
    ]
)

print("Clock fields normalized:", len(clock_field_mapping))
print("Clock staging columns:", len(silver_clock_df.columns))

# COMMAND ----------

clock_validation_rows = []

for raw_name, clean_name in clock_field_mapping.items():

    temp = (
        silver_raw_df
        .select(
            F.expr(
                f"try_cast(nullif(trim(`{raw_name}`), '') AS INT)"
            ).alias("raw_value"),

            normalize_bts_clock(raw_name)
                .alias("clean_value")
        )
    )

    summary = temp.agg(

        F.sum(
            F.when(
                F.col("raw_value") == 0,
                1
            ).otherwise(0)
        ).alias("raw_zero_count"),

        F.sum(
            F.when(
                F.col("raw_value") == 2400,
                1
            ).otherwise(0)
        ).alias("raw_2400_count"),

        F.sum(
            F.when(
                F.col("clean_value").isNull(),
                1
            ).otherwise(0)
        ).alias("normalized_null_count"),

        F.sum(
            F.when(
                (F.col("raw_value") == 2400) &
                (F.col("clean_value") != "00:00"),
                1
            ).otherwise(0)
        ).alias("bad_2400_conversion"),

        F.sum(
            F.when(
                F.col("raw_value").between(1, 2359) &
                (~F.col("clean_value").rlike(
                    r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$"
                )),
                1
            ).otherwise(0)
        ).alias("invalid_normalized_format")
    ).first()

    clock_validation_rows.append(
        (
            clean_name,
            summary["raw_zero_count"],
            summary["raw_2400_count"],
            summary["normalized_null_count"],
            summary["bad_2400_conversion"],
            summary["invalid_normalized_format"]
        )
    )


clock_normalization_validation_df = spark.createDataFrame(
    clock_validation_rows,
    [
        "field",
        "raw_zero_count",
        "raw_2400_count",
        "normalized_null_count",
        "bad_2400_conversion",
        "invalid_normalized_format"
    ]
)

display(clock_normalization_validation_df)

# COMMAND ----------

clock_validation_summary = (
    clock_normalization_validation_df
    .agg(
        F.sum("bad_2400_conversion")
            .alias("total_bad_2400_conversions"),

        F.sum("invalid_normalized_format")
            .alias("total_invalid_normalized_formats"),

        F.sum(
            F.abs(
                F.col("raw_zero_count") -
                F.col("normalized_null_count")
            )
        ).alias("zero_null_count_difference")
    )
)

result = clock_validation_summary.first()

print(
    "Bad 2400 conversions:",
    result["total_bad_2400_conversions"]
)

print(
    "Invalid normalized formats:",
    result["total_invalid_normalized_formats"]
)

print(
    "Zero-to-NULL count difference:",
    result["zero_null_count_difference"]
)

# COMMAND ----------

special_ops_columns = [

    # Gate-return information
    normalize_bts_clock(
        "first_gate_departure_raw"
    ).alias("first_gate_departure"),

    (
        F.expr(
            "try_cast(nullif(trim(`first_gate_departure_raw`), '') AS INT)"
        ) == 2400
    ).alias("first_gate_departure_was_2400"),

    int_or_null(
        "total_gate_return_ground_minutes_raw"
    ).alias("total_gate_return_ground_minutes"),

    int_or_null(
        "longest_gate_return_ground_minutes_raw"
    ).alias("longest_gate_return_ground_minutes"),

    int_or_null(
        "diversion_code_raw"
    ).alias("diversion_code")
]


# Add the five repeated diversion-event blocks
for i in range(1, 6):

    special_ops_columns.extend([

        blank_to_null(
            f"diverted_airport_{i}_raw"
        ).alias(
            f"diverted_airport_{i}"
        ),

        normalize_bts_clock(
            f"diverted_wheels_on_{i}_raw"
        ).alias(
            f"diverted_wheels_on_{i}"
        ),

        (
            F.expr(
                f"try_cast("
                f"nullif(trim(`diverted_wheels_on_{i}_raw`), '') "
                f"AS INT)"
            ) == 2400
        ).alias(
            f"diverted_wheels_on_{i}_was_2400"
        ),

        int_or_null(
            f"diverted_ground_time_{i}_raw"
        ).alias(
            f"diverted_ground_time_{i}_minutes"
        ),

        int_or_null(
            f"diverted_longest_ground_time_{i}_raw"
        ).alias(
            f"diverted_longest_ground_time_{i}_minutes"
        ),

        normalize_bts_clock(
            f"diverted_wheels_off_{i}_raw"
        ).alias(
            f"diverted_wheels_off_{i}"
        ),

        (
            F.expr(
                f"try_cast("
                f"nullif(trim(`diverted_wheels_off_{i}_raw`), '') "
                f"AS INT)"
            ) == 2400
        ).alias(
            f"diverted_wheels_off_{i}_was_2400"
        ),

        blank_to_null(
            f"diverted_tail_number_{i}_raw"
        ).alias(
            f"diverted_tail_number_{i}"
        )
    ])


silver_special_ops_df = silver_raw_df.select(

    # Lineage
    "source_file",
    "source_year",
    "source_month",
    "ingestion_timestamp",

    *special_ops_columns
)


print(
    f"Special-operation records: "
    f"{silver_special_ops_df.count():,}"
)

print(
    "Special-operation columns:",
    len(silver_special_ops_df.columns)
)

# COMMAND ----------

silver_df = (
    silver_raw_df
    .select(
        # -------------------------------------------------
        # 1. Lineage
        # -------------------------------------------------
        "source_file",
        "source_year",
        "source_month",
        "ingestion_timestamp",

        # -------------------------------------------------
        # 2. Flight identity
        # -------------------------------------------------
        blank_to_null("marketing_carrier_raw")
            .alias("marketing_carrier"),

        blank_to_null("marketing_flight_number_raw")
            .alias("marketing_flight_number"),

        blank_to_null("scheduled_operating_carrier_raw")
            .alias("scheduled_operating_carrier"),

        blank_to_null("scheduled_operating_flight_number_raw")
            .alias("scheduled_operating_flight_number"),

        blank_to_null("operating_carrier_raw")
            .alias("operating_carrier"),

        blank_to_null("operating_flight_number_raw")
            .alias("operating_flight_number"),

        blank_to_null("origin_raw")
            .alias("origin"),

        blank_to_null("destination_raw")
            .alias("destination"),

        F.to_date(
            F.col("flight_date_raw"),
            "yyyyMMdd"
        ).alias("flight_date"),

        F.col("day_of_week_raw")
            .cast("int")
            .alias("day_of_week"),

        # -------------------------------------------------
        # 3. Scheduled / actual clock fields
        # -------------------------------------------------
        normalize_bts_clock("scheduled_departure_oag_raw")
            .alias("scheduled_departure_oag"),

        normalize_bts_clock("scheduled_departure_crs_raw")
            .alias("scheduled_departure_crs"),

        normalize_bts_clock("actual_gate_departure_raw")
            .alias("actual_gate_departure"),

        normalize_bts_clock("scheduled_arrival_oag_raw")
            .alias("scheduled_arrival_oag"),

        normalize_bts_clock("scheduled_arrival_crs_raw")
            .alias("scheduled_arrival_crs"),

        normalize_bts_clock("actual_gate_arrival_raw")
            .alias("actual_gate_arrival"),

        normalize_bts_clock("wheels_off_raw")
            .alias("wheels_off"),

        normalize_bts_clock("wheels_on_raw")
            .alias("wheels_on"),

        blank_to_null("tail_number_raw")
            .alias("tail_number"),

        # -------------------------------------------------
        # 4. Schedule / delay / elapsed measures
        # -------------------------------------------------
        int_or_null("departure_schedule_diff_raw")
            .alias("departure_schedule_diff_minutes"),

        int_or_null("arrival_schedule_diff_raw")
            .alias("arrival_schedule_diff_minutes"),

        int_or_null("scheduled_elapsed_minutes_raw")
            .alias("scheduled_elapsed_minutes"),

        int_or_null("actual_elapsed_minutes_raw")
            .alias("actual_elapsed_minutes"),

        int_or_null("departure_delay_minutes_raw")
            .alias("departure_delay_minutes"),

        int_or_null("arrival_delay_minutes_raw")
            .alias("arrival_delay_minutes"),

        int_or_null("elapsed_time_diff_raw")
            .alias("elapsed_time_diff_minutes"),

        int_or_null("taxi_out_minutes_raw")
            .alias("taxi_out_minutes"),

        int_or_null("taxi_in_minutes_raw")
            .alias("taxi_in_minutes"),

        int_or_null("air_time_minutes_raw")
            .alias("air_time_minutes"),

        # -------------------------------------------------
        # 5. Delay-cause minutes
        # -------------------------------------------------
        int_or_null("carrier_delay_minutes_raw")
            .alias("carrier_delay_minutes"),

        int_or_null("weather_delay_minutes_raw")
            .alias("weather_delay_minutes"),

        int_or_null("nas_delay_minutes_raw")
            .alias("nas_delay_minutes"),

        int_or_null("security_delay_minutes_raw")
            .alias("security_delay_minutes"),

        int_or_null("late_aircraft_delay_minutes_raw")
            .alias("late_aircraft_delay_minutes"),

        # -------------------------------------------------
        # 6. Cancellation
        # -------------------------------------------------
        blank_to_null("cancellation_code_raw")
            .alias("cancellation_code"),

        # -------------------------------------------------
        # 7. Gate-return information
        # -------------------------------------------------
        normalize_bts_clock("first_gate_departure_raw")
            .alias("first_gate_departure"),

        int_or_null("total_gate_return_ground_minutes_raw")
            .alias("total_gate_return_ground_minutes"),

        int_or_null("longest_gate_return_ground_minutes_raw")
            .alias("longest_gate_return_ground_minutes"),

        # -------------------------------------------------
        # 8. Diversion status
        # -------------------------------------------------
        int_or_null("diversion_code_raw")
            .alias("diversion_code"),

        # Diversion event 1
        blank_to_null("diverted_airport_1_raw")
            .alias("diverted_airport_1"),

        normalize_bts_clock("diverted_wheels_on_1_raw")
            .alias("diverted_wheels_on_1"),

        int_or_null("diverted_ground_time_1_raw")
            .alias("diverted_ground_time_1_minutes"),

        int_or_null("diverted_longest_ground_time_1_raw")
            .alias("diverted_longest_ground_time_1_minutes"),

        normalize_bts_clock("diverted_wheels_off_1_raw")
            .alias("diverted_wheels_off_1"),

        blank_to_null("diverted_tail_number_1_raw")
            .alias("diverted_tail_number_1"),

        # Diversion event 2
        blank_to_null("diverted_airport_2_raw")
            .alias("diverted_airport_2"),

        normalize_bts_clock("diverted_wheels_on_2_raw")
            .alias("diverted_wheels_on_2"),

        int_or_null("diverted_ground_time_2_raw")
            .alias("diverted_ground_time_2_minutes"),

        int_or_null("diverted_longest_ground_time_2_raw")
            .alias("diverted_longest_ground_time_2_minutes"),

        normalize_bts_clock("diverted_wheels_off_2_raw")
            .alias("diverted_wheels_off_2"),

        blank_to_null("diverted_tail_number_2_raw")
            .alias("diverted_tail_number_2"),

        # Diversion event 3
        blank_to_null("diverted_airport_3_raw")
            .alias("diverted_airport_3"),

        normalize_bts_clock("diverted_wheels_on_3_raw")
            .alias("diverted_wheels_on_3"),

        int_or_null("diverted_ground_time_3_raw")
            .alias("diverted_ground_time_3_minutes"),

        int_or_null("diverted_longest_ground_time_3_raw")
            .alias("diverted_longest_ground_time_3_minutes"),

        normalize_bts_clock("diverted_wheels_off_3_raw")
            .alias("diverted_wheels_off_3"),

        blank_to_null("diverted_tail_number_3_raw")
            .alias("diverted_tail_number_3"),

        # Diversion event 4
        blank_to_null("diverted_airport_4_raw")
            .alias("diverted_airport_4"),

        normalize_bts_clock("diverted_wheels_on_4_raw")
            .alias("diverted_wheels_on_4"),

        int_or_null("diverted_ground_time_4_raw")
            .alias("diverted_ground_time_4_minutes"),

        int_or_null("diverted_longest_ground_time_4_raw")
            .alias("diverted_longest_ground_time_4_minutes"),

        normalize_bts_clock("diverted_wheels_off_4_raw")
            .alias("diverted_wheels_off_4"),

        blank_to_null("diverted_tail_number_4_raw")
            .alias("diverted_tail_number_4"),

        # Diversion event 5
        blank_to_null("diverted_airport_5_raw")
            .alias("diverted_airport_5"),

        normalize_bts_clock("diverted_wheels_on_5_raw")
            .alias("diverted_wheels_on_5"),

        int_or_null("diverted_ground_time_5_raw")
            .alias("diverted_ground_time_5_minutes"),

        int_or_null("diverted_longest_ground_time_5_raw")
            .alias("diverted_longest_ground_time_5_minutes"),

        normalize_bts_clock("diverted_wheels_off_5_raw")
            .alias("diverted_wheels_off_5"),

        blank_to_null("diverted_tail_number_5_raw")
            .alias("diverted_tail_number_5"),

        # -------------------------------------------------
        # 9. Reporting metadata
        # -------------------------------------------------
        blank_to_null("form_type_raw")
            .alias("form_type"),

        blank_to_null("duplicate_flag_raw")
            .alias("duplicate_flag")
    )

    # -----------------------------------------------------
    # 10. Business / operational flags
    # -----------------------------------------------------
    .withColumn(
        "reported_cancellation_flag",
        F.col("cancellation_code").isin("A", "B", "C", "D")
    )

    .withColumn(
        "air_return_cancelled_flag",
        F.col("diversion_code") == 9
    )

    .withColumn(
        "operational_cancellation_flag",
        F.col("reported_cancellation_flag") |
        F.col("air_return_cancelled_flag")
    )

    .withColumn(
        "cancellation_reason",
        F.when(F.col("cancellation_code") == "A", "Carrier")
         .when(F.col("cancellation_code") == "B", "Weather")
         .when(F.col("cancellation_code") == "C", "NAS")
         .when(F.col("cancellation_code") == "D", "Security")
         .when(
             F.col("air_return_cancelled_flag"),
             "Unknown / not reported"
         )
         .otherwise(F.lit(None))
    )

    .withColumn(
        "has_diversion_event_data",
        F.col("diverted_airport_1").isNotNull() |
        F.col("diverted_airport_2").isNotNull() |
        F.col("diverted_airport_3").isNotNull() |
        F.col("diverted_airport_4").isNotNull() |
        F.col("diverted_airport_5").isNotNull()
    )

    .withColumn(
        "diversion_consistency_anomaly_flag",
        (F.col("diversion_code") == 0) &
        F.col("has_diversion_event_data")
    )

    .withColumn(
        "canonical_operation_flag",
        F.col("duplicate_flag") == "N"
    )

    .withColumn(
        "arrival_delay_15_plus_flag",
        F.col("arrival_delay_minutes") >= 15
    )
)

print(f"Final Silver DataFrame records: {silver_df.count():,}")
print("Final Silver DataFrame columns:", len(silver_df.columns))

# COMMAND ----------

silver_df = (
    silver_df

    .withColumn(
        "reported_cancellation_flag",
        F.coalesce(
            F.col("cancellation_code").isin("A", "B", "C", "D"),
            F.lit(False)
        )
    )

    .withColumn(
        "air_return_cancelled_flag",
        F.coalesce(
            F.col("diversion_code") == 9,
            F.lit(False)
        )
    )

    .withColumn(
        "operational_cancellation_flag",
        F.col("reported_cancellation_flag") |
        F.col("air_return_cancelled_flag")
    )

    .withColumn(
        "has_diversion_event_data",
        F.col("diverted_airport_1").isNotNull() |
        F.col("diverted_airport_2").isNotNull() |
        F.col("diverted_airport_3").isNotNull() |
        F.col("diverted_airport_4").isNotNull() |
        F.col("diverted_airport_5").isNotNull()
    )

    .withColumn(
        "diversion_consistency_anomaly_flag",
        F.coalesce(
            (F.col("diversion_code") == 0) &
            F.col("has_diversion_event_data"),
            F.lit(False)
        )
    )

    .withColumn(
        "canonical_operation_flag",
        F.coalesce(
            F.col("duplicate_flag") == "N",
            F.lit(False)
        )
    )

    .withColumn(
        "arrival_delay_15_plus_flag",
        F.coalesce(
            F.col("arrival_delay_minutes") >= 15,
            F.lit(False)
        )
    )
)

# COMMAND ----------

flag_columns = [
    "reported_cancellation_flag",
    "air_return_cancelled_flag",
    "operational_cancellation_flag",
    "has_diversion_event_data",
    "diversion_consistency_anomaly_flag",
    "canonical_operation_flag",
    "arrival_delay_15_plus_flag"
]

flag_validation = silver_df.agg(

    F.count("*").alias("total_records"),

    *[
        F.sum(
            F.when(
                F.col(column_name).isNull(),
                1
            ).otherwise(0)
        ).alias(f"{column_name}_nulls")
        for column_name in flag_columns
    ],

    F.sum(
        F.when(
            F.col("operational_cancellation_flag"),
            1
        ).otherwise(0)
    ).alias("operational_cancellations"),

    F.sum(
        F.when(
            ~F.col("canonical_operation_flag"),
            1
        ).otherwise(0)
    ).alias("superseded_records"),

    F.sum(
        F.when(
            F.col("diversion_consistency_anomaly_flag"),
            1
        ).otherwise(0)
    ).alias("diversion_anomalies"),

    F.sum(
        F.when(
            F.col("arrival_delay_15_plus_flag"),
            1
        ).otherwise(0)
    ).alias("arrival_delay_15_plus")
)

result = flag_validation.first()

for field_name, value in result.asDict().items():
    print(f"{field_name}: {value:,}")

# COMMAND ----------

core_silver_fields = [
    "marketing_carrier",
    "marketing_flight_number",
    "operating_carrier",
    "operating_flight_number",
    "origin",
    "destination",
    "flight_date",
    "day_of_week",
    "form_type",
    "duplicate_flag"
]

final_integrity_check = (
    silver_df
    .agg(
        F.count("*").alias("total_records"),

        F.countDistinct("source_month")
            .alias("months_loaded"),

        F.min("source_month")
            .alias("first_month"),

        F.max("source_month")
            .alias("last_month"),

        F.min("flight_date")
            .alias("first_flight_date"),

        F.max("flight_date")
            .alias("last_flight_date"),

        *[
            F.sum(
                F.when(
                    F.col(field_name).isNull(),
                    1
                ).otherwise(0)
            ).alias(f"{field_name}_nulls")
            for field_name in core_silver_fields
        ],

        F.sum(
            F.when(
                F.col("canonical_operation_flag"),
                1
            ).otherwise(0)
        ).alias("canonical_operations"),

        F.sum(
            F.when(
                F.col("operational_cancellation_flag"),
                1
            ).otherwise(0)
        ).alias("operational_cancellations"),

        F.sum(
            F.when(
                F.col("diversion_consistency_anomaly_flag"),
                1
            ).otherwise(0)
        ).alias("diversion_anomalies")
    )
)

result = final_integrity_check.first()

print("=== FINAL SILVER INTEGRITY CHECK ===")

print(f"Records: {result['total_records']:,}")
print(f"Columns: {len(silver_df.columns)}")
print(f"Unique column names: {len(set(silver_df.columns))}")

print()
print(f"Months loaded: {result['months_loaded']}")
print(
    f"Month range: "
    f"{result['first_month']} -> {result['last_month']}"
)
print(
    f"Flight-date range: "
    f"{result['first_flight_date']} -> "
    f"{result['last_flight_date']}"
)

print()
print("Core-field NULL counts:")

for field_name in core_silver_fields:
    print(
        f"  {field_name}: "
        f"{result[field_name + '_nulls']:,}"
    )

print()
print(
    "Canonical operations:",
    f"{result['canonical_operations']:,}"
)

print(
    "Operational cancellations:",
    f"{result['operational_cancellations']:,}"
)

print(
    "Diversion anomalies:",
    f"{result['diversion_anomalies']:,}"
)

# COMMAND ----------

# Create the Silver schema if it does not already exist
spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS {SILVER_SCHEMA}"
)

# Persist the validated Silver DataFrame as a Delta table
(
    silver_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE)
)

print("Silver Delta table created successfully.")
print("Table:", SILVER_TABLE)

# COMMAND ----------

# Read the persisted Silver Delta table back from storage
silver_table_df = spark.table(SILVER_TABLE)

persisted_silver_check = (
    silver_table_df
    .agg(
        F.count("*").alias("total_records"),

        F.countDistinct("source_month")
            .alias("months_loaded"),

        F.min("source_month")
            .alias("first_month"),

        F.max("source_month")
            .alias("last_month"),

        F.min("flight_date")
            .alias("first_flight_date"),

        F.max("flight_date")
            .alias("last_flight_date"),

        F.sum(
            F.when(
                F.col("canonical_operation_flag"),
                1
            ).otherwise(0)
        ).alias("canonical_operations"),

        F.sum(
            F.when(
                F.col("operational_cancellation_flag"),
                1
            ).otherwise(0)
        ).alias("operational_cancellations"),

        F.sum(
            F.when(
                F.col("diversion_consistency_anomaly_flag"),
                1
            ).otherwise(0)
        ).alias("diversion_anomalies"),

        F.sum(
            F.when(
                F.col("arrival_delay_15_plus_flag"),
                1
            ).otherwise(0)
        ).alias("arrival_delay_15_plus")
    )
)

result = persisted_silver_check.first()

print("=== PERSISTED SILVER TABLE CHECK ===")
print(f"Records: {result['total_records']:,}")
print(f"Columns: {len(silver_table_df.columns)}")
print(f"Unique column names: {len(set(silver_table_df.columns))}")

print()
print(f"Months loaded: {result['months_loaded']}")
print(
    f"Month range: "
    f"{result['first_month']} -> {result['last_month']}"
)
print(
    f"Flight-date range: "
    f"{result['first_flight_date']} -> "
    f"{result['last_flight_date']}"
)

print()
print(
    "Canonical operations:",
    f"{result['canonical_operations']:,}"
)

print(
    "Operational cancellations:",
    f"{result['operational_cancellations']:,}"
)

print(
    "Diversion anomalies:",
    f"{result['diversion_anomalies']:,}"
)

print(
    "Arrival delay 15+ minutes:",
    f"{result['arrival_delay_15_plus']:,}"
)

# COMMAND ----------

# Key columns whose data types matter analytically
important_columns = [
    "source_year",
    "source_month",
    "ingestion_timestamp",
    "marketing_carrier",
    "marketing_flight_number",
    "flight_date",
    "day_of_week",
    "scheduled_departure_crs",
    "actual_gate_departure",
    "arrival_delay_minutes",
    "actual_elapsed_minutes",
    "diversion_code",
    "reported_cancellation_flag",
    "operational_cancellation_flag",
    "canonical_operation_flag",
    "arrival_delay_15_plus_flag"
]

schema_lookup = dict(silver_table_df.dtypes)

print("=== KEY SILVER DATA TYPES ===")

for column_name in important_columns:
    print(
        f"{column_name}: "
        f"{schema_lookup[column_name]}"
    )

print()
print("=== DATA TYPE DISTRIBUTION ===")

type_distribution = (
    spark.createDataFrame(
        silver_table_df.dtypes,
        ["column_name", "data_type"]
    )
    .groupBy("data_type")
    .count()
    .orderBy("data_type")
)

display(type_distribution)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC DESCRIBE HISTORY workspace.airline_silver.flights;