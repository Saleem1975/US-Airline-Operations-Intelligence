# Databricks notebook source
from pyspark.sql import functions as F

BRONZE_TABLE = "workspace.airline_bronze.flights_raw"

bronze_df = spark.table(BRONZE_TABLE)

print("Bronze record count:", f"{bronze_df.count():,}")
print("Bronze column count:", len(bronze_df.columns))
print("Bronze columns:", bronze_df.columns)

# COMMAND ----------

split_fields = F.split(F.col("raw_record"), "\\|")

positional_df = bronze_df.select(
    "source_year",
    "source_month",
    "source_file",
    "ingestion_timestamp",
    *[
        split_fields.getItem(i).alias(f"field_{i + 1:02d}")
        for i in range(71)
    ]
)

print("Positional column count:", len(positional_df.columns))

display(
    positional_df.select(
        "source_month",
        "field_01",
        "field_02",
        "field_03",
        "field_04",
        "field_05",
        "field_06",
        "field_07",
        "field_08",
        "field_09",
        "field_70",
        "field_71"
    ).limit(10)
)

# COMMAND ----------

mapped_preview = positional_df.select(
    "source_month",

    F.col("field_01").alias("marketing_carrier"),
    F.col("field_02").alias("marketing_flight_number"),
    F.col("field_03").alias("scheduled_operating_carrier"),
    F.col("field_04").alias("scheduled_operating_flight_number"),
    F.col("field_05").alias("operating_carrier"),
    F.col("field_06").alias("operating_flight_number"),
    F.col("field_07").alias("origin"),
    F.col("field_08").alias("destination"),
    F.col("field_09").alias("flight_date_raw"),
    F.col("field_10").alias("day_of_week_raw"),

    F.col("field_70").alias("form_type"),
    F.col("field_71").alias("duplicate_flag")
)

display(mapped_preview.limit(10))

# COMMAND ----------

timing_preview = positional_df.select(
    "source_month",

    F.col("field_07").alias("origin"),
    F.col("field_08").alias("destination"),
    F.col("field_09").alias("flight_date_raw"),

    F.col("field_11").alias("scheduled_departure_oag_raw"),
    F.col("field_12").alias("scheduled_departure_crs_raw"),
    F.col("field_13").alias("actual_gate_departure_raw"),

    F.col("field_14").alias("scheduled_arrival_oag_raw"),
    F.col("field_15").alias("scheduled_arrival_crs_raw"),
    F.col("field_16").alias("actual_gate_arrival_raw"),

    F.col("field_19").alias("scheduled_elapsed_minutes_raw"),
    F.col("field_20").alias("actual_elapsed_minutes_raw"),

    F.col("field_21").alias("departure_delay_minutes_raw"),
    F.col("field_22").alias("arrival_delay_minutes_raw"),

    F.col("field_24").alias("wheels_off_raw"),
    F.col("field_25").alias("wheels_on_raw"),
    F.col("field_26").alias("tail_number_raw")
)

display(timing_preview.limit(10))

# COMMAND ----------

operations_preview = positional_df.select(
    "source_month",

    F.col("field_07").alias("origin"),
    F.col("field_08").alias("destination"),
    F.col("field_09").alias("flight_date_raw"),

    F.col("field_27").alias("taxi_out_minutes_raw"),
    F.col("field_28").alias("taxi_in_minutes_raw"),
    F.col("field_29").alias("air_time_minutes_raw"),

    F.col("field_30").alias("cancellation_code_raw"),

    F.col("field_31").alias("carrier_delay_minutes_raw"),
    F.col("field_32").alias("weather_delay_minutes_raw"),
    F.col("field_33").alias("nas_delay_minutes_raw"),
    F.col("field_34").alias("security_delay_minutes_raw"),
    F.col("field_35").alias("late_aircraft_delay_minutes_raw")
)

display(operations_preview.limit(10))

# COMMAND ----------

disruption_preview = positional_df.select(
    "source_month",

    F.col("field_07").alias("origin"),
    F.col("field_08").alias("destination"),
    F.col("field_09").alias("flight_date_raw"),

    F.col("field_30").alias("cancellation_code_raw"),

    F.col("field_31").alias("carrier_delay_minutes_raw"),
    F.col("field_32").alias("weather_delay_minutes_raw"),
    F.col("field_33").alias("nas_delay_minutes_raw"),
    F.col("field_34").alias("security_delay_minutes_raw"),
    F.col("field_35").alias("late_aircraft_delay_minutes_raw")
)

disrupted_examples = disruption_preview.filter(
    (F.trim(F.col("cancellation_code_raw")) != "") |
    (F.coalesce(F.col("carrier_delay_minutes_raw").cast("int"), F.lit(0)) > 0) |
    (F.coalesce(F.col("weather_delay_minutes_raw").cast("int"), F.lit(0)) > 0) |
    (F.coalesce(F.col("nas_delay_minutes_raw").cast("int"), F.lit(0)) > 0) |
    (F.coalesce(F.col("security_delay_minutes_raw").cast("int"), F.lit(0)) > 0) |
    (F.coalesce(F.col("late_aircraft_delay_minutes_raw").cast("int"), F.lit(0)) > 0)
)

display(disrupted_examples.limit(15))

# COMMAND ----------

irregular_ops_preview = positional_df.select(
    "source_month",

    F.col("field_07").alias("origin"),
    F.col("field_08").alias("destination"),
    F.col("field_09").alias("flight_date_raw"),

    F.col("field_36").alias("first_gate_departure_raw"),
    F.col("field_37").alias("total_gate_return_ground_minutes_raw"),
    F.col("field_38").alias("longest_gate_return_ground_minutes_raw"),

    F.col("field_39").alias("diverted_landings_raw"),

    F.col("field_40").alias("diverted_airport_1_raw"),
    F.col("field_41").alias("diverted_wheels_on_1_raw"),
    F.col("field_42").alias("diverted_ground_time_1_raw"),
    F.col("field_43").alias("diverted_longest_ground_time_1_raw"),
    F.col("field_44").alias("diverted_wheels_off_1_raw"),
    F.col("field_45").alias("diverted_tail_number_1_raw")
)

irregular_examples = irregular_ops_preview.filter(
    (F.trim(F.col("first_gate_departure_raw")) != "") |
    (
        F.coalesce(
            F.col("diverted_landings_raw").cast("int"),
            F.lit(0)
        ) > 0
    )
)

display(irregular_examples.limit(15))

# COMMAND ----------

irregular_examples_clean = irregular_ops_preview.filter(
    (
        F.coalesce(
            F.col("first_gate_departure_raw").cast("int"),
            F.lit(0)
        ) > 0
    )
    |
    (
        F.coalesce(
            F.col("diverted_landings_raw").cast("int"),
            F.lit(0)
        ) > 0
    )
)

display(irregular_examples_clean.limit(20))

# COMMAND ----------

diversion_examples = (
    irregular_ops_preview
    .filter(
        F.coalesce(
            F.col("diverted_landings_raw").cast("int"),
            F.lit(0)
        ) > 0
    )
    .select(
        "source_month",
        "origin",
        "destination",
        "flight_date_raw",
        "diverted_landings_raw",
        "diverted_airport_1_raw",
        "diverted_wheels_on_1_raw",
        "diverted_ground_time_1_raw",
        "diverted_longest_ground_time_1_raw",
        "diverted_wheels_off_1_raw",
        "diverted_tail_number_1_raw"
    )
)

display(diversion_examples.limit(20))

# COMMAND ----------

diversion_code_profile = (
    positional_df
    .select(
        F.trim(F.col("field_39")).alias("raw_code")
    )
    .withColumn(
        "code",
        F.when(
            (F.col("raw_code") == "") | F.col("raw_code").isNull(),
            F.lit(None)
        ).otherwise(
            F.col("raw_code").cast("int")
        )
    )
    .withColumn(
        "meaning",
        F.when(F.col("code").isNull(), "Blank / not reported")
        .when(F.col("code") == 0, "No diverted landing")
        .when(
            F.col("code").between(1, 5),
            "Actual diverted landing(s)"
        )
        .when(
            F.col("code") == 9,
            "Air return ultimately canceled"
        )
        .otherwise("Unexpected code")
    )
    .groupBy(
        "raw_code",
        "code",
        "meaning"
    )
    .count()
    .withColumn(
        "percent_of_records",
        F.round(
            F.col("count") / F.lit(7_736_945) * 100,
            4
        )
    )
    .orderBy("code")
)

display(diversion_code_profile)

# COMMAND ----------

diversion_block_check = (
    positional_df
    .select(
        F.col("field_39").cast("int").alias("diversion_code"),

        F.col("field_40").alias("airport_1"),
        F.col("field_46").alias("airport_2"),
        F.col("field_52").alias("airport_3"),
        F.col("field_58").alias("airport_4"),
        F.col("field_64").alias("airport_5")
    )
    .withColumn(
        "populated_diversion_airports",
        sum(
            F.when(
                (F.col(col_name).isNotNull()) &
                (F.trim(F.col(col_name)) != ""),
                1
            ).otherwise(0)
            for col_name in [
                "airport_1",
                "airport_2",
                "airport_3",
                "airport_4",
                "airport_5"
            ]
        )
    )
    .groupBy(
        "diversion_code",
        "populated_diversion_airports"
    )
    .count()
    .orderBy(
        "diversion_code",
        "populated_diversion_airports"
    )
)

display(diversion_block_check)

# COMMAND ----------

diversion_anomaly = (
    positional_df
    .select(
        "source_month",

        F.col("field_01").alias("marketing_carrier"),
        F.col("field_02").alias("marketing_flight_number"),
        F.col("field_05").alias("operating_carrier"),
        F.col("field_06").alias("operating_flight_number"),

        F.col("field_07").alias("origin"),
        F.col("field_08").alias("destination"),
        F.col("field_09").alias("flight_date_raw"),

        F.col("field_30").alias("cancellation_code_raw"),
        F.col("field_39").alias("diversion_code_raw"),

        F.col("field_40").alias("diverted_airport_1"),
        F.col("field_46").alias("diverted_airport_2"),
        F.col("field_52").alias("diverted_airport_3"),
        F.col("field_58").alias("diverted_airport_4"),
        F.col("field_64").alias("diverted_airport_5"),

        F.col("field_70").alias("form_type"),
        F.col("field_71").alias("duplicate_flag")
    )
    .filter(
        (F.col("diversion_code_raw") == "0") &
        (
            (F.trim(F.col("diverted_airport_1")) != "") |
            (F.trim(F.col("diverted_airport_2")) != "") |
            (F.trim(F.col("diverted_airport_3")) != "") |
            (F.trim(F.col("diverted_airport_4")) != "") |
            (F.trim(F.col("diverted_airport_5")) != "")
        )
    )
)

display(diversion_anomaly)

# COMMAND ----------

display(
    diversion_anomaly.select(
        "origin",
        "destination",
        "flight_date_raw",
        "cancellation_code_raw",
        "diversion_code_raw",
        "diverted_airport_1",
        "form_type",
        "duplicate_flag"
    )
)

# COMMAND ----------

anomaly_detail = (
    diversion_anomaly
    .select(
        "origin",
        "destination",
        "flight_date_raw",
        "cancellation_code_raw",
        "diversion_code_raw",
        "diverted_airport_1",
        F.col("field_41").alias("diverted_wheels_on_1"),
        F.col("field_42").alias("diverted_ground_time_1"),
        F.col("field_43").alias("diverted_longest_ground_time_1"),
        F.col("field_44").alias("diverted_wheels_off_1"),
        F.col("field_45").alias("diverted_tail_number_1"),
        "form_type",
        "duplicate_flag"
    )
    .first()
)

for field_name, value in anomaly_detail.asDict().items():
    print(f"{field_name}: {repr(value)}")

# COMMAND ----------

anomaly_detail = (
    positional_df
    .filter(
        (F.col("field_01") == "DL") &
        (F.col("field_02") == "5859") &
        (F.col("field_07") == "JFK") &
        (F.col("field_08") == "MVY") &
        (F.col("field_09") == "20250906")
    )
    .select(
        F.col("field_07").alias("origin"),
        F.col("field_08").alias("destination"),
        F.col("field_09").alias("flight_date_raw"),
        F.col("field_30").alias("cancellation_code_raw"),
        F.col("field_39").alias("diversion_code_raw"),
        F.col("field_40").alias("diverted_airport_1"),
        F.col("field_41").alias("diverted_wheels_on_1"),
        F.col("field_42").alias("diverted_ground_time_1"),
        F.col("field_43").alias("diverted_longest_ground_time_1"),
        F.col("field_44").alias("diverted_wheels_off_1"),
        F.col("field_45").alias("diverted_tail_number_1"),
        F.col("field_70").alias("form_type"),
        F.col("field_71").alias("duplicate_flag")
    )
    .first()
)

for field_name, value in anomaly_detail.asDict().items():
    print(f"{field_name}: {repr(value)}")

# COMMAND ----------

raw_named_df = positional_df.select(
    "source_year",
    "source_month",
    "source_file",
    "ingestion_timestamp",

    F.col("field_01").alias("marketing_carrier_raw"),
    F.col("field_02").alias("marketing_flight_number_raw"),
    F.col("field_03").alias("scheduled_operating_carrier_raw"),
    F.col("field_04").alias("scheduled_operating_flight_number_raw"),
    F.col("field_05").alias("operating_carrier_raw"),
    F.col("field_06").alias("operating_flight_number_raw"),

    F.col("field_07").alias("origin_raw"),
    F.col("field_08").alias("destination_raw"),
    F.col("field_09").alias("flight_date_raw"),
    F.col("field_10").alias("day_of_week_raw"),

    F.col("field_11").alias("scheduled_departure_oag_raw"),
    F.col("field_12").alias("scheduled_departure_crs_raw"),
    F.col("field_13").alias("actual_gate_departure_raw"),

    F.col("field_14").alias("scheduled_arrival_oag_raw"),
    F.col("field_15").alias("scheduled_arrival_crs_raw"),
    F.col("field_16").alias("actual_gate_arrival_raw"),

    F.col("field_17").alias("departure_schedule_diff_raw"),
    F.col("field_18").alias("arrival_schedule_diff_raw"),

    F.col("field_19").alias("scheduled_elapsed_minutes_raw"),
    F.col("field_20").alias("actual_elapsed_minutes_raw"),
    F.col("field_21").alias("departure_delay_minutes_raw"),
    F.col("field_22").alias("arrival_delay_minutes_raw"),
    F.col("field_23").alias("elapsed_time_diff_raw"),

    F.col("field_24").alias("wheels_off_raw"),
    F.col("field_25").alias("wheels_on_raw"),
    F.col("field_26").alias("tail_number_raw"),

    F.col("field_27").alias("taxi_out_minutes_raw"),
    F.col("field_28").alias("taxi_in_minutes_raw"),
    F.col("field_29").alias("air_time_minutes_raw"),

    F.col("field_30").alias("cancellation_code_raw"),

    F.col("field_31").alias("carrier_delay_minutes_raw"),
    F.col("field_32").alias("weather_delay_minutes_raw"),
    F.col("field_33").alias("nas_delay_minutes_raw"),
    F.col("field_34").alias("security_delay_minutes_raw"),
    F.col("field_35").alias("late_aircraft_delay_minutes_raw"),

    F.col("field_36").alias("first_gate_departure_raw"),
    F.col("field_37").alias("total_gate_return_ground_minutes_raw"),
    F.col("field_38").alias("longest_gate_return_ground_minutes_raw"),

    F.col("field_39").alias("diversion_code_raw"),

    F.col("field_40").alias("diverted_airport_1_raw"),
    F.col("field_41").alias("diverted_wheels_on_1_raw"),
    F.col("field_42").alias("diverted_ground_time_1_raw"),
    F.col("field_43").alias("diverted_longest_ground_time_1_raw"),
    F.col("field_44").alias("diverted_wheels_off_1_raw"),
    F.col("field_45").alias("diverted_tail_number_1_raw"),

    F.col("field_46").alias("diverted_airport_2_raw"),
    F.col("field_47").alias("diverted_wheels_on_2_raw"),
    F.col("field_48").alias("diverted_ground_time_2_raw"),
    F.col("field_49").alias("diverted_longest_ground_time_2_raw"),
    F.col("field_50").alias("diverted_wheels_off_2_raw"),
    F.col("field_51").alias("diverted_tail_number_2_raw"),

    F.col("field_52").alias("diverted_airport_3_raw"),
    F.col("field_53").alias("diverted_wheels_on_3_raw"),
    F.col("field_54").alias("diverted_ground_time_3_raw"),
    F.col("field_55").alias("diverted_longest_ground_time_3_raw"),
    F.col("field_56").alias("diverted_wheels_off_3_raw"),
    F.col("field_57").alias("diverted_tail_number_3_raw"),

    F.col("field_58").alias("diverted_airport_4_raw"),
    F.col("field_59").alias("diverted_wheels_on_4_raw"),
    F.col("field_60").alias("diverted_ground_time_4_raw"),
    F.col("field_61").alias("diverted_longest_ground_time_4_raw"),
    F.col("field_62").alias("diverted_wheels_off_4_raw"),
    F.col("field_63").alias("diverted_tail_number_4_raw"),

    F.col("field_64").alias("diverted_airport_5_raw"),
    F.col("field_65").alias("diverted_wheels_on_5_raw"),
    F.col("field_66").alias("diverted_ground_time_5_raw"),
    F.col("field_67").alias("diverted_longest_ground_time_5_raw"),
    F.col("field_68").alias("diverted_wheels_off_5_raw"),
    F.col("field_69").alias("diverted_tail_number_5_raw"),

    F.col("field_70").alias("form_type_raw"),
    F.col("field_71").alias("duplicate_flag_raw")
)

print("Named raw column count:", len(raw_named_df.columns))

# COMMAND ----------

core_fields = [
    "marketing_carrier_raw",
    "marketing_flight_number_raw",
    "operating_carrier_raw",
    "operating_flight_number_raw",
    "origin_raw",
    "destination_raw",
    "flight_date_raw",
    "day_of_week_raw",
    "form_type_raw",
    "duplicate_flag_raw"
]

missing_expressions = [
    F.sum(
        F.when(
            F.col(column_name).isNull() |
            (F.trim(F.col(column_name)) == ""),
            1
        ).otherwise(0)
    ).alias(column_name)
    for column_name in core_fields
]

missing_counts = (
    raw_named_df
    .agg(*missing_expressions)
    .first()
    .asDict()
)

total_records = 7_736_945

missing_summary = [
    (
        column_name,
        missing_counts[column_name],
        round(
            missing_counts[column_name] / total_records * 100,
            6
        )
    )
    for column_name in core_fields
]

missing_summary_df = spark.createDataFrame(
    missing_summary,
    [
        "field",
        "missing_or_blank_records",
        "missing_percent"
    ]
)

display(
    missing_summary_df.orderBy(
        F.desc("missing_or_blank_records")
    )
)

# COMMAND ----------

date_quality_df = (
    raw_named_df
    .withColumn(
        "flight_date",
        F.to_date(
            F.col("flight_date_raw"),
            "yyyyMMdd"
        )
    )
    .withColumn(
        "day_of_week_int",
        F.col("day_of_week_raw").cast("int")
    )
    .withColumn(
        "derived_bts_day_of_week",
        (
            (F.dayofweek(F.col("flight_date")) + 5) % 7
        ) + 1
    )
)

date_quality_summary = date_quality_df.agg(

    F.count("*").alias("total_records"),

    F.sum(
        F.when(
            F.col("flight_date").isNull(),
            1
        ).otherwise(0)
    ).alias("invalid_flight_dates"),

    F.sum(
        F.when(
            F.col("flight_date").isNotNull() &
            (F.year(F.col("flight_date")) != F.col("source_year")),
            1
        ).otherwise(0)
    ).alias("year_mismatch"),

    F.sum(
        F.when(
            F.col("flight_date").isNotNull() &
            (F.month(F.col("flight_date")) != F.col("source_month")),
            1
        ).otherwise(0)
    ).alias("month_mismatch"),

    F.sum(
        F.when(
            F.col("day_of_week_int").isNull() |
            (~F.col("day_of_week_int").between(1, 7)),
            1
        ).otherwise(0)
    ).alias("invalid_day_of_week"),

    F.sum(
        F.when(
            F.col("flight_date").isNotNull() &
            F.col("day_of_week_int").between(1, 7) &
            (
                F.col("day_of_week_int")
                !=
                F.col("derived_bts_day_of_week")
            ),
            1
        ).otherwise(0)
    ).alias("date_day_of_week_mismatch")
)

display(date_quality_summary)

# COMMAND ----------

airport_quality_summary = (
    raw_named_df
    .agg(
        F.count("*").alias("total_records"),

        F.sum(
            F.when(
                ~F.col("origin_raw").rlike("^[A-Z]{3}$"),
                1
            ).otherwise(0)
        ).alias("invalid_origin_format"),

        F.sum(
            F.when(
                ~F.col("destination_raw").rlike("^[A-Z]{3}$"),
                1
            ).otherwise(0)
        ).alias("invalid_destination_format"),

        F.countDistinct("origin_raw").alias("distinct_origins"),

        F.countDistinct("destination_raw").alias(
            "distinct_destinations"
        ),

        F.sum(
            F.when(
                F.col("origin_raw") == F.col("destination_raw"),
                1
            ).otherwise(0)
        ).alias("same_origin_destination")
    )
)

display(airport_quality_summary)

# COMMAND ----------

same_airport_count = (
    raw_named_df
    .filter(
        F.col("origin_raw") == F.col("destination_raw")
    )
    .count()
)

print("Same origin/destination records:", same_airport_count)

# COMMAND ----------

carrier_flight_quality = (
    raw_named_df
    .agg(
        F.count("*").alias("total_records"),

        F.countDistinct("marketing_carrier_raw")
            .alias("distinct_marketing_carriers"),

        F.countDistinct("operating_carrier_raw")
            .alias("distinct_operating_carriers"),

        F.min(F.length("marketing_carrier_raw"))
            .alias("min_marketing_carrier_length"),

        F.max(F.length("marketing_carrier_raw"))
            .alias("max_marketing_carrier_length"),

        F.min(F.length("operating_carrier_raw"))
            .alias("min_operating_carrier_length"),

        F.max(F.length("operating_carrier_raw"))
            .alias("max_operating_carrier_length"),

        F.sum(
            F.when(
                ~F.col("marketing_carrier_raw").rlike("^[A-Z0-9]+$"),
                1
            ).otherwise(0)
        ).alias("invalid_marketing_carrier_characters"),

        F.sum(
            F.when(
                ~F.col("operating_carrier_raw").rlike("^[A-Z0-9]+$"),
                1
            ).otherwise(0)
        ).alias("invalid_operating_carrier_characters"),

        F.sum(
            F.when(
                ~F.col("marketing_flight_number_raw").rlike("^[0-9]+$"),
                1
            ).otherwise(0)
        ).alias("non_numeric_marketing_flight_numbers"),

        F.sum(
            F.when(
                ~F.col("operating_flight_number_raw").rlike("^[0-9]+$"),
                1
            ).otherwise(0)
        ).alias("non_numeric_operating_flight_numbers")
    )
)

display(carrier_flight_quality)

# COMMAND ----------

display(
    carrier_flight_quality.select(
        "min_marketing_carrier_length",
        "max_marketing_carrier_length",
        "min_operating_carrier_length",
        "max_operating_carrier_length",
        "invalid_marketing_carrier_characters",
        "invalid_operating_carrier_characters",
        "non_numeric_marketing_flight_numbers",
        "non_numeric_operating_flight_numbers"
    )
)

# COMMAND ----------

display(
    carrier_flight_quality.select(
        "invalid_marketing_carrier_characters",
        "invalid_operating_carrier_characters",
        "non_numeric_marketing_flight_numbers",
        "non_numeric_operating_flight_numbers"
    )
)

# COMMAND ----------

result = carrier_flight_quality.select(
    "non_numeric_operating_flight_numbers"
).first()

print(
    "Non-numeric operating flight numbers:",
    result["non_numeric_operating_flight_numbers"]
)

# COMMAND ----------

form_duplicate_profile = (
    raw_named_df
    .select(
        F.trim(F.col("form_type_raw")).alias("form_type"),
        F.trim(F.col("duplicate_flag_raw")).alias("duplicate_flag")
    )
    .groupBy(
        "form_type",
        "duplicate_flag"
    )
    .count()
    .withColumn(
        "percent_of_records",
        F.round(
            F.col("count") / F.lit(7_736_945) * 100,
            6
        )
    )
    .orderBy(
        F.desc("count")
    )
)

display(form_duplicate_profile)

# COMMAND ----------

duplicate_form1 = (
    raw_named_df
    .filter(
        (F.col("form_type_raw") == "FORM-1") &
        (F.col("duplicate_flag_raw") == "Y")
    )
    .select(
        F.col("marketing_carrier_raw").alias("marketing_carrier"),
        F.col("marketing_flight_number_raw").alias("marketing_flight"),
        F.col("operating_carrier_raw").alias("original_operator"),
        F.col("operating_flight_number_raw").alias("original_operator_flight"),
        F.col("origin_raw").alias("origin"),
        F.col("destination_raw").alias("destination"),
        F.col("flight_date_raw").alias("flight_date")
    )
)

form3a = (
    raw_named_df
    .filter(
        (F.col("form_type_raw") == "FORM-3A") &
        (F.col("duplicate_flag_raw") == "N")
    )
    .select(
        F.col("marketing_carrier_raw").alias("marketing_carrier"),
        F.col("marketing_flight_number_raw").alias("marketing_flight"),
        F.col("scheduled_operating_carrier_raw").alias("scheduled_operator"),
        F.col("scheduled_operating_flight_number_raw").alias(
            "scheduled_operator_flight"
        ),
        F.col("operating_carrier_raw").alias("replacement_operator"),
        F.col("operating_flight_number_raw").alias(
            "replacement_operator_flight"
        ),
        F.col("origin_raw").alias("origin"),
        F.col("destination_raw").alias("destination"),
        F.col("flight_date_raw").alias("flight_date")
    )
)

duplicate_match_check = (
    duplicate_form1.alias("d")
    .join(
        form3a.alias("f"),
        (
            (F.col("d.marketing_carrier") == F.col("f.marketing_carrier")) &
            (F.col("d.marketing_flight") == F.col("f.marketing_flight")) &
            (F.col("d.origin") == F.col("f.origin")) &
            (F.col("d.destination") == F.col("f.destination")) &
            (F.col("d.flight_date") == F.col("f.flight_date")) &
            (F.col("d.original_operator") == F.col("f.scheduled_operator")) &
            (
                F.col("d.original_operator_flight")
                == F.col("f.scheduled_operator_flight")
            )
        ),
        "left"
    )
)

duplicate_validation_summary = (
    duplicate_match_check
    .agg(
        F.count("*").alias("duplicate_form1_records"),

        F.sum(
            F.when(
                F.col("f.marketing_carrier").isNotNull(),
                1
            ).otherwise(0)
        ).alias("matched_to_form3a"),

        F.sum(
            F.when(
                F.col("f.marketing_carrier").isNull(),
                1
            ).otherwise(0)
        ).alias("unmatched_duplicate_records")
    )
)

display(duplicate_validation_summary)

# COMMAND ----------

# Re-create the exact unmatched FORM-1 / Y population
exact_condition = (
    (F.col("d.marketing_carrier") == F.col("f.marketing_carrier")) &
    (F.col("d.marketing_flight") == F.col("f.marketing_flight")) &
    (F.col("d.origin") == F.col("f.origin")) &
    (F.col("d.destination") == F.col("f.destination")) &
    (F.col("d.flight_date") == F.col("f.flight_date")) &
    (F.col("d.original_operator") == F.col("f.scheduled_operator")) &
    (
        F.col("d.original_operator_flight")
        == F.col("f.scheduled_operator_flight")
    )
)

unmatched_duplicates = (
    duplicate_form1.alias("d")
    .join(
        form3a.alias("f"),
        exact_condition,
        "left_anti"
    )
)

# Test progressively less restrictive matching rules
same_marketed_flight_date_route = (
    unmatched_duplicates.alias("d")
    .join(
        form3a.alias("f"),
        (
            (F.col("d.marketing_carrier") == F.col("f.marketing_carrier")) &
            (F.col("d.marketing_flight") == F.col("f.marketing_flight")) &
            (F.col("d.origin") == F.col("f.origin")) &
            (F.col("d.destination") == F.col("f.destination")) &
            (F.col("d.flight_date") == F.col("f.flight_date"))
        ),
        "left_semi"
    )
    .count()
)

same_marketing_date_route = (
    unmatched_duplicates.alias("d")
    .join(
        form3a.alias("f"),
        (
            (F.col("d.marketing_carrier") == F.col("f.marketing_carrier")) &
            (F.col("d.origin") == F.col("f.origin")) &
            (F.col("d.destination") == F.col("f.destination")) &
            (F.col("d.flight_date") == F.col("f.flight_date"))
        ),
        "left_semi"
    )
    .count()
)

same_date_route = (
    unmatched_duplicates.alias("d")
    .join(
        form3a.alias("f"),
        (
            (F.col("d.origin") == F.col("f.origin")) &
            (F.col("d.destination") == F.col("f.destination")) &
            (F.col("d.flight_date") == F.col("f.flight_date"))
        ),
        "left_semi"
    )
    .count()
)

print("Unmatched exact records:", unmatched_duplicates.count())
print(
    "Match same marketing carrier + flight + date + route:",
    same_marketed_flight_date_route
)
print(
    "Match same marketing carrier + date + route:",
    same_marketing_date_route
)
print(
    "Match same date + route:",
    same_date_route
)

# COMMAND ----------

match_without_marketing_flight = (
    unmatched_duplicates.alias("d")
    .join(
        form3a.alias("f"),
        (
            (F.col("d.marketing_carrier") == F.col("f.marketing_carrier")) &
            (F.col("d.origin") == F.col("f.origin")) &
            (F.col("d.destination") == F.col("f.destination")) &
            (F.col("d.flight_date") == F.col("f.flight_date")) &
            (F.col("d.original_operator") == F.col("f.scheduled_operator")) &
            (
                F.col("d.original_operator_flight")
                == F.col("f.scheduled_operator_flight")
            )
        ),
        "left_semi"
    )
    .count()
)

print(
    "Unmatched duplicates matched after ignoring only "
    "marketing flight number:",
    match_without_marketing_flight
)

# COMMAND ----------

relaxed_matches = (
    unmatched_duplicates.alias("d")
    .join(
        form3a.alias("f"),
        (
            (F.col("d.marketing_carrier") == F.col("f.marketing_carrier")) &
            (F.col("d.origin") == F.col("f.origin")) &
            (F.col("d.destination") == F.col("f.destination")) &
            (F.col("d.flight_date") == F.col("f.flight_date")) &
            (F.col("d.original_operator") == F.col("f.scheduled_operator")) &
            (
                F.col("d.original_operator_flight")
                == F.col("f.scheduled_operator_flight")
            )
        ),
        "inner"
    )
    .select(
        F.col("d.marketing_carrier").alias("marketing_carrier"),
        F.col("d.marketing_flight").alias("original_marketing_flight"),
        F.col("d.original_operator").alias("original_operator"),
        F.col("d.original_operator_flight").alias("original_operator_flight"),
        F.col("d.origin").alias("origin"),
        F.col("d.destination").alias("destination"),
        F.col("d.flight_date").alias("flight_date"),

        F.col("f.marketing_flight").alias("form3a_marketing_flight"),
        F.col("f.replacement_operator").alias("replacement_operator"),
        F.col("f.replacement_operator_flight").alias(
            "replacement_operator_flight"
        )
    )
)

match_uniqueness = (
    relaxed_matches
    .groupBy(
        "marketing_carrier",
        "original_marketing_flight",
        "original_operator",
        "original_operator_flight",
        "origin",
        "destination",
        "flight_date"
    )
    .agg(
        F.count("*").alias("candidate_form3a_matches")
    )
    .groupBy("candidate_form3a_matches")
    .count()
    .orderBy("candidate_form3a_matches")
)

display(match_uniqueness)

# COMMAND ----------

cancellation_profile = (
    raw_named_df
    .select(
        F.trim(
            F.col("cancellation_code_raw")
        ).alias("cancellation_code")
    )
    .withColumn(
        "meaning",
        F.when(
            (F.col("cancellation_code") == "") |
            F.col("cancellation_code").isNull(),
            "Blank / no cancellation code reported"
        )
        .when(F.col("cancellation_code") == "A", "Carrier")
        .when(F.col("cancellation_code") == "B", "Weather")
        .when(F.col("cancellation_code") == "C", "NAS")
        .when(F.col("cancellation_code") == "D", "Security")
        .otherwise("Unexpected code")
    )
    .groupBy(
        "cancellation_code",
        "meaning"
    )
    .count()
    .withColumn(
        "percent_of_records",
        F.round(
            F.col("count") / F.lit(7_736_945) * 100,
            4
        )
    )
    .orderBy(F.desc("count"))
)

display(cancellation_profile)

# COMMAND ----------

cancelled_operations_profile = (
    raw_named_df
    .withColumn(
        "is_cancelled",
        F.trim(F.col("cancellation_code_raw")).isin("A", "B", "C", "D")
    )
    .groupBy("is_cancelled")
    .agg(
        F.count("*").alias("records"),

        F.sum(
            F.when(
                F.col("actual_gate_departure_raw").cast("int") > 0,
                1
            ).otherwise(0)
        ).alias("has_actual_gate_departure"),

        F.sum(
            F.when(
                F.col("wheels_off_raw").cast("int") > 0,
                1
            ).otherwise(0)
        ).alias("has_wheels_off"),

        F.sum(
            F.when(
                F.col("wheels_on_raw").cast("int") > 0,
                1
            ).otherwise(0)
        ).alias("has_wheels_on"),

        F.sum(
            F.when(
                F.col("actual_gate_arrival_raw").cast("int") > 0,
                1
            ).otherwise(0)
        ).alias("has_actual_gate_arrival"),

        F.sum(
            F.when(
                F.col("first_gate_departure_raw").cast("int") > 0,
                1
            ).otherwise(0)
        ).alias("has_gate_return_activity"),

        F.sum(
            F.when(
                F.col("diversion_code_raw").cast("int") == 9,
                1
            ).otherwise(0)
        ).alias("air_return_cancelled_code_9")
    )
    .orderBy(F.desc("is_cancelled"))
)

display(cancelled_operations_profile)

# COMMAND ----------

display(
    cancelled_operations_profile.select(
        "is_cancelled",
        "records",
        "has_gate_return_activity",
        "air_return_cancelled_code_9"
    )
)

# COMMAND ----------

code9_profile = (
    raw_named_df
    .filter(
        F.col("diversion_code_raw").cast("int") == 9
    )
    .select(
        F.trim(F.col("cancellation_code_raw")).alias("cancellation_code"),
        F.col("form_type_raw").alias("form_type"),
        F.col("duplicate_flag_raw").alias("duplicate_flag")
    )
    .groupBy(
        "cancellation_code",
        "form_type",
        "duplicate_flag"
    )
    .count()
    .orderBy(F.desc("count"))
)

display(code9_profile)

# COMMAND ----------

delay_cause_profile = (
    raw_named_df
    .select(
        F.col("arrival_delay_minutes_raw")
            .cast("int")
            .alias("arrival_delay"),

        F.col("carrier_delay_minutes_raw")
            .cast("int")
            .alias("carrier_delay"),

        F.col("weather_delay_minutes_raw")
            .cast("int")
            .alias("weather_delay"),

        F.col("nas_delay_minutes_raw")
            .cast("int")
            .alias("nas_delay"),

        F.col("security_delay_minutes_raw")
            .cast("int")
            .alias("security_delay"),

        F.col("late_aircraft_delay_minutes_raw")
            .cast("int")
            .alias("late_aircraft_delay")
    )
    .agg(
        F.count("*").alias("total_records"),

        F.sum(
            F.when(F.col("arrival_delay") >= 15, 1).otherwise(0)
        ).alias("arrival_delay_15_plus"),

        F.sum(
            F.when(F.col("carrier_delay") > 0, 1).otherwise(0)
        ).alias("positive_carrier_delay"),

        F.sum(
            F.when(F.col("weather_delay") > 0, 1).otherwise(0)
        ).alias("positive_weather_delay"),

        F.sum(
            F.when(F.col("nas_delay") > 0, 1).otherwise(0)
        ).alias("positive_nas_delay"),

        F.sum(
            F.when(F.col("security_delay") > 0, 1).otherwise(0)
        ).alias("positive_security_delay"),

        F.sum(
            F.when(F.col("late_aircraft_delay") > 0, 1).otherwise(0)
        ).alias("positive_late_aircraft_delay")
    )
)

display(delay_cause_profile)

# COMMAND ----------

display(
    delay_cause_profile.select(
        "arrival_delay_15_plus",
        "positive_carrier_delay",
        "positive_weather_delay",
        "positive_nas_delay",
        "positive_security_delay",
        "positive_late_aircraft_delay"
    )
)

# COMMAND ----------

result = delay_cause_profile.first()

for field_name in [
    "arrival_delay_15_plus",
    "positive_carrier_delay",
    "positive_weather_delay",
    "positive_nas_delay",
    "positive_security_delay",
    "positive_late_aircraft_delay"
]:
    print(f"{field_name}: {result[field_name]:,}")

# COMMAND ----------

late_flight_reconciliation = (
    raw_named_df
    .select(
        F.col("arrival_delay_minutes_raw")
            .cast("int")
            .alias("arrival_delay"),

        F.coalesce(
            F.col("carrier_delay_minutes_raw").cast("int"),
            F.lit(0)
        ).alias("carrier_delay"),

        F.coalesce(
            F.col("weather_delay_minutes_raw").cast("int"),
            F.lit(0)
        ).alias("weather_delay"),

        F.coalesce(
            F.col("nas_delay_minutes_raw").cast("int"),
            F.lit(0)
        ).alias("nas_delay"),

        F.coalesce(
            F.col("security_delay_minutes_raw").cast("int"),
            F.lit(0)
        ).alias("security_delay"),

        F.coalesce(
            F.col("late_aircraft_delay_minutes_raw").cast("int"),
            F.lit(0)
        ).alias("late_aircraft_delay"),

        F.trim(
            F.col("cancellation_code_raw")
        ).alias("cancellation_code"),

        F.col("diversion_code_raw")
            .cast("int")
            .alias("diversion_code"),

        F.col("duplicate_flag_raw")
            .alias("duplicate_flag")
    )
    .filter(
        (F.col("arrival_delay") >= 15) &
        (
            F.col("cancellation_code").isNull() |
            (F.col("cancellation_code") == "")
        ) &
        (F.col("diversion_code") == 0) &
        (F.col("duplicate_flag") == "N")
    )
    .withColumn(
        "total_reported_cause_minutes",
        F.col("carrier_delay")
        + F.col("weather_delay")
        + F.col("nas_delay")
        + F.col("security_delay")
        + F.col("late_aircraft_delay")
    )
    .withColumn(
        "difference",
        F.col("total_reported_cause_minutes")
        - F.col("arrival_delay")
    )
)

delay_reconciliation_summary = (
    late_flight_reconciliation
    .agg(
        F.count("*").alias("eligible_late_flights"),

        F.sum(
            F.when(F.col("difference") == 0, 1).otherwise(0)
        ).alias("exact_matches"),

        F.sum(
            F.when(F.col("difference") < 0, 1).otherwise(0)
        ).alias("cause_minutes_less_than_arrival_delay"),

        F.sum(
            F.when(F.col("difference") > 0, 1).otherwise(0)
        ).alias("cause_minutes_greater_than_arrival_delay"),

        F.sum(
            F.when(
                F.col("total_reported_cause_minutes") == 0,
                1
            ).otherwise(0)
        ).alias("zero_reported_cause_minutes"),

        F.max(
            F.abs(F.col("difference"))
        ).alias("maximum_absolute_difference")
    )
)

display(delay_reconciliation_summary)

# COMMAND ----------

result = delay_reconciliation_summary.first()

print(
    "Zero reported cause minutes:",
    result["zero_reported_cause_minutes"]
)

print(
    "Maximum absolute difference:",
    result["maximum_absolute_difference"]
)

# COMMAND ----------

elapsed_time_check = (
    raw_named_df
    .select(
        F.col("actual_elapsed_minutes_raw")
            .cast("int")
            .alias("actual_elapsed"),

        F.col("taxi_out_minutes_raw")
            .cast("int")
            .alias("taxi_out"),

        F.col("air_time_minutes_raw")
            .cast("int")
            .alias("air_time"),

        F.col("taxi_in_minutes_raw")
            .cast("int")
            .alias("taxi_in"),

        F.trim(
            F.col("cancellation_code_raw")
        ).alias("cancellation_code"),

        F.col("diversion_code_raw")
            .cast("int")
            .alias("diversion_code"),

        F.col("duplicate_flag_raw")
            .alias("duplicate_flag")
    )
    .filter(
        (
            F.col("cancellation_code").isNull() |
            (F.col("cancellation_code") == "")
        ) &
        (F.col("diversion_code") == 0) &
        (F.col("duplicate_flag") == "N")
    )
    .withColumn(
        "component_sum",
        F.col("taxi_out") +
        F.col("air_time") +
        F.col("taxi_in")
    )
    .withColumn(
        "difference",
        F.col("actual_elapsed") -
        F.col("component_sum")
    )
)

elapsed_time_summary = (
    elapsed_time_check
    .agg(
        F.count("*").alias("eligible_completed_flights"),

        F.sum(
            F.when(
                F.col("actual_elapsed").isNull() |
                F.col("taxi_out").isNull() |
                F.col("air_time").isNull() |
                F.col("taxi_in").isNull(),
                1
            ).otherwise(0)
        ).alias("records_with_missing_components"),

        F.sum(
            F.when(
                F.col("difference") == 0,
                1
            ).otherwise(0)
        ).alias("exact_matches"),

        F.sum(
            F.when(
                F.col("difference") != 0,
                1
            ).otherwise(0)
        ).alias("mismatches"),

        F.max(
            F.abs(F.col("difference"))
        ).alias("maximum_absolute_difference")
    )
)

display(elapsed_time_summary)

# COMMAND ----------

elapsed_time_anomaly = (
    raw_named_df
    .select(
        "source_month",

        F.col("marketing_carrier_raw").alias("marketing_carrier"),
        F.col("marketing_flight_number_raw").alias("marketing_flight_number"),

        F.col("operating_carrier_raw").alias("operating_carrier"),
        F.col("operating_flight_number_raw").alias("operating_flight_number"),

        F.col("origin_raw").alias("origin"),
        F.col("destination_raw").alias("destination"),
        F.col("flight_date_raw").alias("flight_date"),

        F.col("actual_elapsed_minutes_raw")
            .cast("int").alias("actual_elapsed"),

        F.col("taxi_out_minutes_raw")
            .cast("int").alias("taxi_out"),

        F.col("air_time_minutes_raw")
            .cast("int").alias("air_time"),

        F.col("taxi_in_minutes_raw")
            .cast("int").alias("taxi_in"),

        F.trim(
            F.col("cancellation_code_raw")
        ).alias("cancellation_code"),

        F.col("diversion_code_raw")
            .cast("int").alias("diversion_code"),

        F.col("form_type_raw").alias("form_type"),
        F.col("duplicate_flag_raw").alias("duplicate_flag")
    )
    .filter(
        (
            F.col("cancellation_code").isNull() |
            (F.col("cancellation_code") == "")
        ) &
        (F.col("diversion_code") == 0) &
        (F.col("duplicate_flag") == "N")
    )
    .withColumn(
        "component_sum",
        F.col("taxi_out") +
        F.col("air_time") +
        F.col("taxi_in")
    )
    .withColumn(
        "difference",
        F.col("actual_elapsed") -
        F.col("component_sum")
    )
    .filter(
        F.col("difference") != 0
    )
)

display(elapsed_time_anomaly)

# COMMAND ----------

elapsed_time_check_refined = (
    raw_named_df
    .select(
        F.col("actual_elapsed_minutes_raw")
            .cast("int").alias("actual_elapsed"),

        F.col("taxi_out_minutes_raw")
            .cast("int").alias("taxi_out"),

        F.col("air_time_minutes_raw")
            .cast("int").alias("air_time"),

        F.col("taxi_in_minutes_raw")
            .cast("int").alias("taxi_in"),

        F.trim(
            F.col("cancellation_code_raw")
        ).alias("cancellation_code"),

        F.col("diversion_code_raw")
            .cast("int").alias("diversion_code"),

        F.col("duplicate_flag_raw")
            .alias("duplicate_flag"),

        F.col("diverted_airport_1_raw").alias("diverted_airport_1"),
        F.col("diverted_airport_2_raw").alias("diverted_airport_2"),
        F.col("diverted_airport_3_raw").alias("diverted_airport_3"),
        F.col("diverted_airport_4_raw").alias("diverted_airport_4"),
        F.col("diverted_airport_5_raw").alias("diverted_airport_5")
    )
    .withColumn(
        "has_diversion_event_data",
        (
            (F.trim(F.col("diverted_airport_1")) != "") |
            (F.trim(F.col("diverted_airport_2")) != "") |
            (F.trim(F.col("diverted_airport_3")) != "") |
            (F.trim(F.col("diverted_airport_4")) != "") |
            (F.trim(F.col("diverted_airport_5")) != "")
        )
    )
    .filter(
        (
            F.col("cancellation_code").isNull() |
            (F.col("cancellation_code") == "")
        ) &
        (F.col("diversion_code") == 0) &
        (~F.col("has_diversion_event_data")) &
        (F.col("duplicate_flag") == "N")
    )
    .withColumn(
        "component_sum",
        F.col("taxi_out") +
        F.col("air_time") +
        F.col("taxi_in")
    )
    .withColumn(
        "difference",
        F.col("actual_elapsed") -
        F.col("component_sum")
    )
)

elapsed_time_summary_refined = (
    elapsed_time_check_refined
    .agg(
        F.count("*").alias("eligible_completed_flights"),

        F.sum(
            F.when(F.col("difference") == 0, 1).otherwise(0)
        ).alias("exact_matches"),

        F.sum(
            F.when(F.col("difference") != 0, 1).otherwise(0)
        ).alias("mismatches"),

        F.max(
            F.abs(F.col("difference"))
        ).alias("maximum_absolute_difference")
    )
)

display(elapsed_time_summary_refined)

# COMMAND ----------

duration_profile = (
    elapsed_time_check_refined
    .agg(
        F.count("*").alias("records"),

        F.min("taxi_out").alias("min_taxi_out"),
        F.max("taxi_out").alias("max_taxi_out"),

        F.min("taxi_in").alias("min_taxi_in"),
        F.max("taxi_in").alias("max_taxi_in"),

        F.min("air_time").alias("min_air_time"),
        F.max("air_time").alias("max_air_time"),

        F.min("actual_elapsed").alias("min_actual_elapsed"),
        F.max("actual_elapsed").alias("max_actual_elapsed"),

        F.sum(
            F.when(F.col("taxi_out") < 0, 1).otherwise(0)
        ).alias("negative_taxi_out"),

        F.sum(
            F.when(F.col("taxi_in") < 0, 1).otherwise(0)
        ).alias("negative_taxi_in"),

        F.sum(
            F.when(F.col("air_time") < 0, 1).otherwise(0)
        ).alias("negative_air_time"),

        F.sum(
            F.when(F.col("actual_elapsed") < 0, 1).otherwise(0)
        ).alias("negative_actual_elapsed")
    )
)

display(duration_profile)

# COMMAND ----------

result = duration_profile.first()

for field_name in [
    "min_actual_elapsed",
    "max_actual_elapsed",
    "negative_taxi_out",
    "negative_taxi_in",
    "negative_air_time",
    "negative_actual_elapsed"
]:
    print(f"{field_name}: {result[field_name]:,}")

# COMMAND ----------

time_fields = [
    "scheduled_departure_oag_raw",
    "scheduled_departure_crs_raw",
    "actual_gate_departure_raw",
    "scheduled_arrival_oag_raw",
    "scheduled_arrival_crs_raw",
    "actual_gate_arrival_raw",
    "wheels_off_raw",
    "wheels_on_raw"
]

time_quality_rows = []

for field_name in time_fields:

    temp = (
        raw_named_df
        .select(
            F.trim(F.col(field_name)).alias("raw_time")
        )
        .filter(
            F.col("raw_time").isNotNull() &
            (F.col("raw_time") != "")
        )
        .withColumn(
            "time_value",
            F.col("raw_time").cast("int")
        )
    )

    summary = temp.agg(
        F.count("*").alias("populated_records"),

        F.min("time_value").alias("minimum_value"),
        F.max("time_value").alias("maximum_value"),

        F.sum(
            F.when(
                F.col("time_value").isNull(),
                1
            ).otherwise(0)
        ).alias("non_numeric_values"),

        F.sum(
            F.when(
                (F.col("time_value") < 0) |
                (F.col("time_value") > 2400),
                1
            ).otherwise(0)
        ).alias("outside_0000_2400"),

        F.sum(
            F.when(
                (F.col("time_value") != 2400) &
                ((F.col("time_value") % 100) >= 60),
                1
            ).otherwise(0)
        ).alias("invalid_minute_component"),

        F.sum(
            F.when(
                F.col("time_value") == 2400,
                1
            ).otherwise(0)
        ).alias("value_2400_count")
    ).first()

    time_quality_rows.append(
        (
            field_name,
            summary["populated_records"],
            summary["minimum_value"],
            summary["maximum_value"],
            summary["non_numeric_values"],
            summary["outside_0000_2400"],
            summary["invalid_minute_component"],
            summary["value_2400_count"]
        )
    )

time_quality_df = spark.createDataFrame(
    time_quality_rows,
    [
        "field",
        "populated_records",
        "minimum_value",
        "maximum_value",
        "non_numeric_values",
        "outside_0000_2400",
        "invalid_minute_component",
        "value_2400_count"
    ]
)

display(time_quality_df)

# COMMAND ----------

result_rows = (
    time_quality_df
    .select(
        "field",
        "outside_0000_2400",
        "invalid_minute_component",
        "value_2400_count"
    )
    .collect()
)

for row in result_rows:
    print(
        f"{row['field']}: "
        f"outside range={row['outside_0000_2400']:,}, "
        f"invalid minutes={row['invalid_minute_component']:,}, "
        f"2400 count={row['value_2400_count']:,}"
    )

# COMMAND ----------

zero_time_profile = (
    raw_named_df
    .agg(
        *[
            F.sum(
                F.when(
                    F.col(field_name).cast("int") == 0,
                    1
                ).otherwise(0)
            ).alias(field_name)
            for field_name in [
                "scheduled_departure_oag_raw",
                "scheduled_departure_crs_raw",
                "actual_gate_departure_raw",
                "scheduled_arrival_oag_raw",
                "scheduled_arrival_crs_raw",
                "actual_gate_arrival_raw",
                "wheels_off_raw",
                "wheels_on_raw"
            ]
        ]
    )
    .first()
)

for field_name, value in zero_time_profile.asDict().items():
    print(f"{field_name}: {value:,}")

# COMMAND ----------

oag_zero_records = (
    raw_named_df
    .filter(
        (F.col("scheduled_departure_oag_raw").cast("int") == 0) |
        (F.col("scheduled_arrival_oag_raw").cast("int") == 0)
    )
    .select(
        "source_month",

        F.col("marketing_carrier_raw").alias("marketing_carrier"),
        F.col("marketing_flight_number_raw").alias("marketing_flight"),

        F.col("operating_carrier_raw").alias("operating_carrier"),

        F.col("origin_raw").alias("origin"),
        F.col("destination_raw").alias("destination"),
        F.col("flight_date_raw").alias("flight_date"),

        F.col("scheduled_departure_oag_raw")
            .alias("oag_departure"),

        F.col("scheduled_departure_crs_raw")
            .alias("crs_departure"),

        F.col("scheduled_arrival_oag_raw")
            .alias("oag_arrival"),

        F.col("scheduled_arrival_crs_raw")
            .alias("crs_arrival"),

        F.col("cancellation_code_raw")
            .alias("cancellation_code"),

        F.col("diversion_code_raw")
            .alias("diversion_code"),

        F.col("form_type_raw")
            .alias("form_type"),

        F.col("duplicate_flag_raw")
            .alias("duplicate_flag")
    )
    .orderBy(
        "flight_date",
        "marketing_carrier",
        "marketing_flight"
    )
)

print("OAG-zero records:", oag_zero_records.count())

display(oag_zero_records)

# COMMAND ----------

display(
    oag_zero_records.select(
        "flight_date",
        "marketing_carrier",
        "marketing_flight",
        "operating_carrier",
        "origin",
        "destination",
        "oag_departure",
        "crs_departure",
        "oag_arrival",
        "crs_arrival",
        "cancellation_code",
        "diversion_code",
        "form_type",
        "duplicate_flag"
    )
)

# COMMAND ----------

rows = oag_zero_records.select(
    "flight_date",
    "marketing_carrier",
    "marketing_flight",
    "operating_carrier",
    "origin",
    "destination",
    "oag_departure",
    "crs_departure",
    "oag_arrival",
    "crs_arrival"
).collect()

for row in rows:
    print(
        f"{row['flight_date']} | "
        f"{row['marketing_carrier']} {row['marketing_flight']} | "
        f"operator={row['operating_carrier']} | "
        f"{row['origin']}->{row['destination']} | "
        f"OAG={row['oag_departure']}->{row['oag_arrival']} | "
        f"CRS={row['crs_departure']}->{row['crs_arrival']}"
    )