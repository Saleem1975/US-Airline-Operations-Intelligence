# Databricks notebook source
from pyspark.sql import functions as F

SILVER_TABLE = "workspace.airline_silver.flights"

silver_df = spark.table(SILVER_TABLE)

print(f"Silver records: {silver_df.count():,}")
print("Silver columns:", len(silver_df.columns))


# ---------------------------------------------------------
# Unique airport codes appearing anywhere in the flight data
# ---------------------------------------------------------

origin_airports = (
    silver_df
    .select(F.col("origin").alias("airport_code"))
)

destination_airports = (
    silver_df
    .select(F.col("destination").alias("airport_code"))
)

airports_used = (
    origin_airports
    .union(destination_airports)
    .distinct()
    .orderBy("airport_code")
)


# ---------------------------------------------------------
# Unique carrier codes
# ---------------------------------------------------------

marketing_carriers = (
    silver_df
    .select(
        F.col("marketing_carrier").alias("carrier_code")
    )
)

operating_carriers = (
    silver_df
    .select(
        F.col("operating_carrier").alias("carrier_code")
    )
)

carriers_used = (
    marketing_carriers
    .union(operating_carriers)
    .distinct()
    .orderBy("carrier_code")
)


print()
print("Distinct airports used:", airports_used.count())
print("Distinct carriers used:", carriers_used.count())

# COMMAND ----------

REFERENCE_DIRECTORY = (
    "/Volumes/workspace/default/my_files/"
    "us_airline_operations_intelligence/reference/bts_support"
)

CARRIER_FILE = f"{REFERENCE_DIRECTORY}/T_CARRIER_DECODE.csv"
AIRPORT_FILE = f"{REFERENCE_DIRECTORY}/T_MASTER_CORD.csv"


carrier_raw_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .option("mode", "FAILFAST")
    .csv(CARRIER_FILE)
)

airport_raw_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .option("mode", "FAILFAST")
    .csv(AIRPORT_FILE)
)


print("=== CARRIER REFERENCE ===")
print("Rows:", carrier_raw_df.count())
print("Columns:", len(carrier_raw_df.columns))
print("Column names:")
print(carrier_raw_df.columns)

print()

print("=== AIRPORT REFERENCE ===")
print("Rows:", airport_raw_df.count())
print("Columns:", len(airport_raw_df.columns))
print("Column names:")
print(airport_raw_df.columns)

# COMMAND ----------

print("=== CARRIER SAMPLE ===")

display(
    carrier_raw_df.select(
        "AIRLINE_ID",
        "CARRIER",
        "CARRIER_NAME",
        "UNIQUE_CARRIER",
        "UNIQUE_CARRIER_NAME",
        "START_DATE_SOURCE",
        "THRU_DATE_SOURCE"
    ).limit(20)
)


print("=== AIRPORT SAMPLE ===")

display(
    airport_raw_df.select(
        "AIRPORT_ID",
        "AIRPORT",
        "DISPLAY_AIRPORT_NAME",
        "DISPLAY_AIRPORT_CITY_NAME_FULL",
        "AIRPORT_STATE_CODE",
        "AIRPORT_COUNTRY_NAME",
        "LATITUDE",
        "LONGITUDE",
        "AIRPORT_START_DATE",
        "AIRPORT_THRU_DATE",
        "AIRPORT_IS_CLOSED",
        "AIRPORT_IS_LATEST"
    ).limit(20)
)
print("=== DATE FIELD EXAMPLES ===")

print("Carrier start dates:")
carrier_raw_df.select(
    "START_DATE_SOURCE"
).distinct().show(10, truncate=False)

print("Carrier thru dates:")
carrier_raw_df.select(
    "THRU_DATE_SOURCE"
).distinct().show(10, truncate=False)

print("Airport start dates:")
airport_raw_df.select(
    "AIRPORT_START_DATE"
).distinct().show(10, truncate=False)

print("Airport thru dates:")
airport_raw_df.select(
    "AIRPORT_THRU_DATE"
).distinct().show(10, truncate=False)
print("=== CARRIER DATE CHECK ===")

carrier_raw_df.select(
    "CARRIER",
    "CARRIER_NAME",
    "START_DATE_SOURCE",
    "THRU_DATE_SOURCE"
).orderBy(
    F.desc("START_DATE_SOURCE")
).show(15, truncate=False)


print("=== AIRPORT DATE CHECK ===")

airport_raw_df.select(
    "AIRPORT",
    "DISPLAY_AIRPORT_NAME",
    "AIRPORT_START_DATE",
    "AIRPORT_THRU_DATE",
    "AIRPORT_IS_CLOSED",
    "AIRPORT_IS_LATEST"
).orderBy(
    F.desc("AIRPORT_START_DATE")
).show(15, truncate=False)
print("Carrier blank/null THRU dates:")

carrier_raw_df.agg(
    F.sum(
        F.when(
            F.col("THRU_DATE_SOURCE").isNull() |
            (F.trim(F.col("THRU_DATE_SOURCE")) == ""),
            1
        ).otherwise(0)
    ).alias("open_ended_carriers")
).show()


print("Airport blank/null THRU dates:")

airport_raw_df.agg(
    F.sum(
        F.when(
            F.col("AIRPORT_THRU_DATE").isNull() |
            (F.trim(F.col("AIRPORT_THRU_DATE")) == ""),
            1
        ).otherwise(0)
    ).alias("open_ended_airports")
).show()

# COMMAND ----------

# Parse BTS reference dates correctly

carrier_dates_df = (
    carrier_raw_df
    .withColumn(
        "start_date",
        F.to_date(
            F.to_timestamp(
                "START_DATE_SOURCE",
                "M/d/yyyy h:mm:ss a"
            )
        )
    )
    .withColumn(
        "thru_date",
        F.to_date(
            F.to_timestamp(
                "THRU_DATE_SOURCE",
                "M/d/yyyy h:mm:ss a"
            )
        )
    )
)

airport_dates_df = (
    airport_raw_df
    .withColumn(
        "start_date",
        F.to_date(
            F.to_timestamp(
                "AIRPORT_START_DATE",
                "M/d/yyyy h:mm:ss a"
            )
        )
    )
    .withColumn(
        "thru_date",
        F.to_date(
            F.to_timestamp(
                "AIRPORT_THRU_DATE",
                "M/d/yyyy h:mm:ss a"
            )
        )
    )
)


print("=== CARRIER DATE PROFILE ===")

carrier_dates_df.agg(
    F.min("start_date").alias("earliest_start"),
    F.max("start_date").alias("latest_start"),
    F.sum(
        F.when(F.col("thru_date").isNull(), 1).otherwise(0)
    ).alias("open_ended_records"),
    F.sum(
        F.when(
            F.col("START_DATE_SOURCE").isNotNull() &
            F.col("start_date").isNull(),
            1
        ).otherwise(0)
    ).alias("failed_start_date_parses"),
    F.sum(
        F.when(
            F.col("THRU_DATE_SOURCE").isNotNull() &
            (F.trim(F.col("THRU_DATE_SOURCE")) != "") &
            F.col("thru_date").isNull(),
            1
        ).otherwise(0)
    ).alias("failed_thru_date_parses")
).show(truncate=False)


print("=== AIRPORT DATE PROFILE ===")

airport_dates_df.agg(
    F.min("start_date").alias("earliest_start"),
    F.max("start_date").alias("latest_start"),
    F.sum(
        F.when(F.col("thru_date").isNull(), 1).otherwise(0)
    ).alias("open_ended_records"),
    F.sum(
        F.when(
            F.col("AIRPORT_START_DATE").isNotNull() &
            F.col("start_date").isNull(),
            1
        ).otherwise(0)
    ).alias("failed_start_date_parses"),
    F.sum(
        F.when(
            F.col("AIRPORT_THRU_DATE").isNotNull() &
            (F.trim(F.col("AIRPORT_THRU_DATE")) != "") &
            F.col("thru_date").isNull(),
            1
        ).otherwise(0)
    ).alias("failed_thru_date_parses")
).show(truncate=False)

# COMMAND ----------

REFERENCE_START = F.to_date(F.lit("2025-01-01"))
REFERENCE_END   = F.to_date(F.lit("2025-12-31"))


# ---------------------------------------------------------
# Carrier records whose effective period overlaps 2025
# ---------------------------------------------------------

carrier_2025_candidates = (
    carrier_dates_df
    .filter(
        (F.col("start_date") <= REFERENCE_END) &
        (
            F.col("thru_date").isNull() |
            (F.col("thru_date") >= REFERENCE_START)
        )
    )
    .select(
        F.trim(F.col("CARRIER")).alias("carrier_code"),
        F.trim(F.col("CARRIER_NAME")).alias("carrier_name"),
        F.trim(F.col("UNIQUE_CARRIER")).alias("unique_carrier"),
        F.trim(F.col("UNIQUE_CARRIER_NAME")).alias("unique_carrier_name"),
        "start_date",
        "thru_date"
    )
)


# ---------------------------------------------------------
# Airport records whose effective period overlaps 2025
# ---------------------------------------------------------

airport_2025_candidates = (
    airport_dates_df
    .filter(
        (F.col("start_date") <= REFERENCE_END) &
        (
            F.col("thru_date").isNull() |
            (F.col("thru_date") >= REFERENCE_START)
        )
    )
    .select(
        F.trim(F.col("AIRPORT")).alias("airport_code"),
        F.trim(F.col("DISPLAY_AIRPORT_NAME")).alias("airport_name"),
        F.trim(
            F.col("DISPLAY_AIRPORT_CITY_NAME_FULL")
        ).alias("city_name"),
        F.trim(F.col("AIRPORT_STATE_CODE")).alias("state_code"),
        F.trim(F.col("AIRPORT_COUNTRY_NAME")).alias("country_name"),
        F.col("LATITUDE").cast("double").alias("latitude"),
        F.col("LONGITUDE").cast("double").alias("longitude"),
        "start_date",
        "thru_date",
        F.col("AIRPORT_IS_CLOSED").alias("airport_is_closed"),
        F.col("AIRPORT_IS_LATEST").alias("airport_is_latest")
    )
)


# ---------------------------------------------------------
# Coverage against codes actually used in our Silver data
# ---------------------------------------------------------

carrier_coverage = (
    carriers_used.alias("u")
    .join(
        carrier_2025_candidates
            .select("carrier_code")
            .distinct()
            .alias("r"),
        "carrier_code",
        "left"
    )
)

airport_coverage = (
    airports_used.alias("u")
    .join(
        airport_2025_candidates
            .select("airport_code")
            .distinct()
            .alias("r"),
        "airport_code",
        "left"
    )
)


matched_carriers = (
    carriers_used
    .join(
        carrier_2025_candidates
            .select("carrier_code")
            .distinct(),
        "carrier_code",
        "inner"
    )
    .count()
)

unmatched_carriers_df = (
    carriers_used
    .join(
        carrier_2025_candidates
            .select("carrier_code")
            .distinct(),
        "carrier_code",
        "left_anti"
    )
)

matched_airports = (
    airports_used
    .join(
        airport_2025_candidates
            .select("airport_code")
            .distinct(),
        "airport_code",
        "inner"
    )
    .count()
)

unmatched_airports_df = (
    airports_used
    .join(
        airport_2025_candidates
            .select("airport_code")
            .distinct(),
        "airport_code",
        "left_anti"
    )
)


print("=== 2025 REFERENCE COVERAGE ===")
print(
    "Carrier candidates:",
    carrier_2025_candidates.count()
)
print(
    "Distinct carrier codes in 2025 reference:",
    carrier_2025_candidates
        .select("carrier_code")
        .distinct()
        .count()
)

print()
print(
    f"Carriers matched: {matched_carriers} / {carriers_used.count()}"
)
print(
    "Unmatched carriers:",
    unmatched_carriers_df.count()
)

print()
print(
    "Airport candidates:",
    airport_2025_candidates.count()
)
print(
    "Distinct airport codes in 2025 reference:",
    airport_2025_candidates
        .select("airport_code")
        .distinct()
        .count()
)

print()
print(
    f"Airports matched: {matched_airports} / {airports_used.count()}"
)
print(
    "Unmatched airports:",
    unmatched_airports_df.count()
)


print("\n=== UNMATCHED CARRIERS ===")
unmatched_carriers_df.show(50, truncate=False)

print("\n=== UNMATCHED AIRPORTS ===")
unmatched_airports_df.show(100, truncate=False)

# COMMAND ----------

print("=== 6L USAGE IN 2025 FLIGHT DATA ===")

silver_df.filter(
    (F.col("marketing_carrier") == "6L") |
    (F.col("operating_carrier") == "6L")
).agg(
    F.count("*").alias("records"),

    F.sum(
        F.when(
            F.col("marketing_carrier") == "6L",
            1
        ).otherwise(0)
    ).alias("as_marketing_carrier"),

    F.sum(
        F.when(
            F.col("operating_carrier") == "6L",
            1
        ).otherwise(0)
    ).alias("as_operating_carrier"),

    F.min("flight_date").alias("first_flight_date"),
    F.max("flight_date").alias("last_flight_date"),

    F.countDistinct("origin").alias("origins"),
    F.countDistinct("destination").alias("destinations")
).show(truncate=False)


print("=== SAMPLE 6L FLIGHTS ===")

silver_df.filter(
    (F.col("marketing_carrier") == "6L") |
    (F.col("operating_carrier") == "6L")
).select(
    "flight_date",
    "marketing_carrier",
    "marketing_flight_number",
    "scheduled_operating_carrier",
    "scheduled_operating_flight_number",
    "operating_carrier",
    "operating_flight_number",
    "origin",
    "destination",
    "form_type",
    "duplicate_flag"
).orderBy(
    "flight_date"
).show(20, truncate=False)


print("=== 6L IN FULL BTS CARRIER REFERENCE ===")

carrier_dates_df.filter(
    F.trim(F.col("CARRIER")) == "6L"
).select(
    "AIRLINE_ID",
    "CARRIER",
    "CARRIER_NAME",
    "UNIQUE_CARRIER",
    "UNIQUE_CARRIER_NAME",
    "start_date",
    "thru_date"
).orderBy(
    "start_date"
).show(50, truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Check how many 2025-valid reference rows exist
# for each code actually used in our flight dataset
# ---------------------------------------------------------

used_carrier_reference_counts = (
    carriers_used
    .join(
        carrier_2025_candidates,
        "carrier_code",
        "left"
    )
    .groupBy("carrier_code")
    .agg(
        F.count("carrier_name").alias("reference_rows")
    )
    .orderBy(
        F.desc("reference_rows"),
        "carrier_code"
    )
)


used_airport_reference_counts = (
    airports_used
    .join(
        airport_2025_candidates,
        "airport_code",
        "left"
    )
    .groupBy("airport_code")
    .agg(
        F.count("airport_name").alias("reference_rows")
    )
    .orderBy(
        F.desc("reference_rows"),
        "airport_code"
    )
)


print("=== CARRIER REFERENCE MULTIPLICITY ===")

used_carrier_reference_counts.groupBy(
    "reference_rows"
).count().orderBy(
    "reference_rows"
).show(truncate=False)


print("Carrier codes with more than one 2025 reference row:")

used_carrier_reference_counts.filter(
    F.col("reference_rows") > 1
).show(50, truncate=False)


print("=== AIRPORT REFERENCE MULTIPLICITY ===")

used_airport_reference_counts.groupBy(
    "reference_rows"
).count().orderBy(
    "reference_rows"
).show(truncate=False)


print("Airport codes with more than one 2025 reference row:")

used_airport_reference_counts.filter(
    F.col("reference_rows") > 1
).show(100, truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Build distinct carrier/date combinations actually used
# in the 2025 flight data
# ---------------------------------------------------------

marketing_usage = (
    silver_df
    .select(
        F.col("flight_date"),
        F.col("marketing_carrier").alias("carrier_code")
    )
)

scheduled_operating_usage = (
    silver_df
    .select(
        F.col("flight_date"),
        F.col("scheduled_operating_carrier").alias("carrier_code")
    )
)

operating_usage = (
    silver_df
    .select(
        F.col("flight_date"),
        F.col("operating_carrier").alias("carrier_code")
    )
)

carrier_date_usage = (
    marketing_usage
    .union(scheduled_operating_usage)
    .union(operating_usage)
    .filter(F.col("carrier_code").isNotNull())
    .distinct()
)


# ---------------------------------------------------------
# Join each carrier/date combination to the reference row
# that was valid on that exact date
# ---------------------------------------------------------

carrier_date_match_counts = (
    carrier_date_usage.alias("u")
    .join(
        carrier_2025_candidates.alias("r"),
        (
            (F.col("u.carrier_code") == F.col("r.carrier_code")) &
            (F.col("u.flight_date") >= F.col("r.start_date")) &
            (
                F.col("r.thru_date").isNull() |
                (F.col("u.flight_date") <= F.col("r.thru_date"))
            )
        ),
        "left"
    )
    .groupBy(
        F.col("u.carrier_code").alias("carrier_code"),
        F.col("u.flight_date").alias("flight_date")
    )
    .agg(
        F.count(F.col("r.carrier_name")).alias("reference_matches")
    )
)


print("=== DATE-AWARE CARRIER MATCH DISTRIBUTION ===")

carrier_date_match_counts.groupBy(
    "reference_matches"
).count().orderBy(
    "reference_matches"
).show(truncate=False)


print("=== CARRIER/DATE COMBINATIONS WITH NO MATCH ===")

carrier_date_match_counts.filter(
    F.col("reference_matches") == 0
).orderBy(
    "carrier_code",
    "flight_date"
).show(50, truncate=False)


print("=== CARRIER/DATE COMBINATIONS WITH MULTIPLE MATCHES ===")

carrier_date_match_counts.filter(
    F.col("reference_matches") > 1
).orderBy(
    F.desc("reference_matches"),
    "carrier_code",
    "flight_date"
).show(100, truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Diagnose whether multiple reference matches actually
# represent different carrier identities
# ---------------------------------------------------------

carrier_date_matches = (
    carrier_date_usage.alias("u")
    .join(
        carrier_2025_candidates.alias("r"),
        (
            (F.col("u.carrier_code") == F.col("r.carrier_code")) &
            (F.col("u.flight_date") >= F.col("r.start_date")) &
            (
                F.col("r.thru_date").isNull() |
                (F.col("u.flight_date") <= F.col("r.thru_date"))
            )
        ),
        "left"
    )
)


carrier_identity_diagnostic = (
    carrier_date_matches
    .groupBy(
        F.col("u.carrier_code").alias("carrier_code"),
        F.col("u.flight_date").alias("flight_date")
    )
    .agg(
        F.count(F.col("r.carrier_name")).alias("reference_rows"),
        F.countDistinct(F.col("r.carrier_name"))
            .alias("distinct_carrier_names"),
        F.countDistinct(F.col("r.unique_carrier"))
            .alias("distinct_unique_carriers"),
        F.countDistinct(F.col("r.unique_carrier_name"))
            .alias("distinct_unique_carrier_names")
    )
)


print("=== MULTIPLE-MATCH IDENTITY PROFILE ===")

carrier_identity_diagnostic.filter(
    F.col("reference_rows") > 1
).agg(
    F.count("*").alias("multiple_match_combinations"),

    F.sum(
        F.when(
            (F.col("distinct_carrier_names") == 1) &
            (F.col("distinct_unique_carriers") == 1) &
            (F.col("distinct_unique_carrier_names") == 1),
            1
        ).otherwise(0)
    ).alias("same_identity_despite_multiple_rows"),

    F.sum(
        F.when(
            (F.col("distinct_carrier_names") > 1) |
            (F.col("distinct_unique_carriers") > 1) |
            (F.col("distinct_unique_carrier_names") > 1),
            1
        ).otherwise(0)
    ).alias("genuinely_ambiguous_combinations")
).show(truncate=False)


print("=== SAMPLE GENUINELY AMBIGUOUS CASES ===")

carrier_identity_diagnostic.filter(
    (F.col("reference_rows") > 1) &
    (
        (F.col("distinct_carrier_names") > 1) |
        (F.col("distinct_unique_carriers") > 1) |
        (F.col("distinct_unique_carrier_names") > 1)
    )
).orderBy(
    "carrier_code",
    "flight_date"
).show(30, truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Check whether each carrier code used in our flights
# has one stable carrier identity throughout 2025
# ---------------------------------------------------------

used_carrier_identity_profile = (
    carriers_used
    .join(
        carrier_2025_candidates,
        "carrier_code",
        "left"
    )
    .groupBy("carrier_code")
    .agg(
        F.countDistinct("carrier_name")
            .alias("distinct_carrier_names"),

        F.countDistinct("unique_carrier")
            .alias("distinct_unique_carriers"),

        F.countDistinct("unique_carrier_name")
            .alias("distinct_unique_carrier_names")
    )
    .orderBy("carrier_code")
)


print("=== 2025 CARRIER IDENTITY STABILITY ===")

used_carrier_identity_profile.groupBy(
    "distinct_carrier_names",
    "distinct_unique_carriers",
    "distinct_unique_carrier_names"
).count().orderBy(
    "distinct_carrier_names",
    "distinct_unique_carriers",
    "distinct_unique_carrier_names"
).show(truncate=False)


print("=== CARRIERS WITH CHANGING / AMBIGUOUS IDENTITY ===")

used_carrier_identity_profile.filter(
    (F.col("distinct_carrier_names") > 1) |
    (F.col("distinct_unique_carriers") > 1) |
    (F.col("distinct_unique_carrier_names") > 1)
).show(50, truncate=False)


print("=== UNMATCHED CARRIERS ===")

used_carrier_identity_profile.filter(
    F.col("distinct_carrier_names") == 0
).show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Create one stable 2025 identity row per matched carrier
# ---------------------------------------------------------

carrier_identity_2025 = (
    carrier_2025_candidates
    .select(
        "carrier_code",
        "carrier_name",
        "unique_carrier",
        "unique_carrier_name"
    )
    .distinct()
)


# ---------------------------------------------------------
# Build dimension using ALL carrier codes found in Silver
# This intentionally retains unmatched carrier 6L
# ---------------------------------------------------------

dim_carrier_df = (
    carriers_used.alias("u")
    .join(
        carrier_identity_2025.alias("r"),
        "carrier_code",
        "left"
    )
    .select(
        "carrier_code",
        "carrier_name",
        "unique_carrier",
        "unique_carrier_name",

        F.col("carrier_name").isNotNull()
            .alias("reference_match_flag")
    )
    .orderBy("carrier_code")
)


print("=== CARRIER DIMENSION ===")

print("Rows:", dim_carrier_df.count())
print(
    "Distinct carrier codes:",
    dim_carrier_df.select("carrier_code").distinct().count()
)

print(
    "Matched carriers:",
    dim_carrier_df.filter(
        F.col("reference_match_flag") == True
    ).count()
)

print(
    "Unmatched carriers:",
    dim_carrier_df.filter(
        F.col("reference_match_flag") == False
    ).count()
)


display(dim_carrier_df)

# COMMAND ----------

# ---------------------------------------------------------
# Check whether each airport used in our 2025 flights
# has one stable reference identity throughout the year
# ---------------------------------------------------------

used_airport_identity_profile = (
    airports_used
    .join(
        airport_2025_candidates,
        "airport_code",
        "left"
    )
    .groupBy("airport_code")
    .agg(
        F.count("start_date").alias("reference_rows"),

        F.countDistinct(
            F.struct(
                "airport_name",
                "city_name",
                "state_code",
                "country_name",
                "latitude",
                "longitude"
            )
        ).alias("distinct_reference_identities"),

        F.countDistinct("airport_name")
            .alias("distinct_airport_names"),

        F.countDistinct("city_name")
            .alias("distinct_city_names"),

        F.countDistinct("state_code")
            .alias("distinct_state_codes"),

        F.countDistinct("latitude")
            .alias("distinct_latitudes"),

        F.countDistinct("longitude")
            .alias("distinct_longitudes")
    )
    .orderBy("airport_code")
)


print("=== AIRPORT DIMENSION READINESS ===")

used_airport_identity_profile.agg(
    F.count("*").alias("airports_used"),

    F.sum(
        F.when(
            F.col("reference_rows") == 0,
            1
        ).otherwise(0)
    ).alias("unmatched_airports"),

    F.sum(
        F.when(
            F.col("reference_rows") == 1,
            1
        ).otherwise(0)
    ).alias("single_reference_row"),

    F.sum(
        F.when(
            F.col("reference_rows") > 1,
            1
        ).otherwise(0)
    ).alias("multiple_reference_rows"),

    F.sum(
        F.when(
            (F.col("reference_rows") > 0) &
            (F.col("distinct_reference_identities") == 1),
            1
        ).otherwise(0)
    ).alias("stable_identity_airports"),

    F.sum(
        F.when(
            F.col("distinct_reference_identities") > 1,
            1
        ).otherwise(0)
    ).alias("changing_or_ambiguous_airports")
).show(truncate=False)


print("=== AIRPORTS WITH MULTIPLE 2025 REFERENCE ROWS ===")

used_airport_identity_profile.filter(
    F.col("reference_rows") > 1
).orderBy(
    F.desc("reference_rows"),
    "airport_code"
).show(50, truncate=False)


print("=== AIRPORTS WITH CHANGING / AMBIGUOUS IDENTITY ===")

used_airport_identity_profile.filter(
    F.col("distinct_reference_identities") > 1
).orderBy(
    "airport_code"
).show(50, truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Build distinct airport/date combinations actually used
# by flights during 2025
# ---------------------------------------------------------

origin_airport_usage = (
    silver_df
    .select(
        "flight_date",
        F.col("origin").alias("airport_code")
    )
)

destination_airport_usage = (
    silver_df
    .select(
        "flight_date",
        F.col("destination").alias("airport_code")
    )
)

airport_date_usage = (
    origin_airport_usage
    .union(destination_airport_usage)
    .filter(F.col("airport_code").isNotNull())
    .distinct()
)


# ---------------------------------------------------------
# Match each airport/date combination to the BTS airport
# reference record valid on that exact flight date
# ---------------------------------------------------------

airport_date_match_counts = (
    airport_date_usage.alias("u")
    .join(
        airport_2025_candidates.alias("r"),
        (
            (F.col("u.airport_code") == F.col("r.airport_code")) &
            (F.col("u.flight_date") >= F.col("r.start_date")) &
            (
                F.col("r.thru_date").isNull() |
                (F.col("u.flight_date") <= F.col("r.thru_date"))
            )
        ),
        "left"
    )
    .groupBy(
        F.col("u.airport_code").alias("airport_code"),
        F.col("u.flight_date").alias("flight_date")
    )
    .agg(
        F.count(F.col("r.airport_name"))
            .alias("reference_matches")
    )
)


print("=== DATE-AWARE AIRPORT MATCH DISTRIBUTION ===")

airport_date_match_counts.groupBy(
    "reference_matches"
).count().orderBy(
    "reference_matches"
).show(truncate=False)


print("=== AIRPORT/DATE COMBINATIONS WITH NO MATCH ===")

airport_date_match_counts.filter(
    F.col("reference_matches") == 0
).orderBy(
    "airport_code",
    "flight_date"
).show(50, truncate=False)


print("=== AIRPORT/DATE COMBINATIONS WITH MULTIPLE MATCHES ===")

airport_date_match_counts.filter(
    F.col("reference_matches") > 1
).orderBy(
    F.desc("reference_matches"),
    "airport_code",
    "flight_date"
).show(50, truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Build historical airport dimension for airports actually
# used in our 2025 flight data
# ---------------------------------------------------------

dim_airport_history_df = (
    airports_used
    .join(
        airport_2025_candidates,
        "airport_code",
        "inner"
    )
    .select(
        "airport_code",
        "airport_name",
        "city_name",
        "state_code",
        "country_name",
        "latitude",
        "longitude",

        F.col("start_date").alias("effective_start_date"),
        F.col("thru_date").alias("effective_end_date"),

        "airport_is_closed",
        "airport_is_latest"
    )
    .distinct()
    .orderBy(
        "airport_code",
        "effective_start_date"
    )
)


print("=== HISTORICAL AIRPORT DIMENSION ===")

print(
    "Dimension rows:",
    dim_airport_history_df.count()
)

print(
    "Distinct airport codes:",
    dim_airport_history_df
        .select("airport_code")
        .distinct()
        .count()
)

print(
    "Airports with one 2025 version:",
    dim_airport_history_df
        .groupBy("airport_code")
        .count()
        .filter(F.col("count") == 1)
        .count()
)

print(
    "Airports with multiple 2025 versions:",
    dim_airport_history_df
        .groupBy("airport_code")
        .count()
        .filter(F.col("count") > 1)
        .count()
)

print(
    "Maximum versions for one airport:",
    dim_airport_history_df
        .groupBy("airport_code")
        .count()
        .agg(F.max("count"))
        .first()[0]
)


display(
    dim_airport_history_df
    .filter(
        F.col("airport_code").isin(
            "ATL", "AZA", "BTV", "DEN", "JFK"
        )
    )
)

# COMMAND ----------

# ---------------------------------------------------------
# Finalize airport dimension data types
# ---------------------------------------------------------

dim_airport_history_df = (
    dim_airport_history_df
    .withColumn(
        "airport_is_closed",
        F.when(F.col("airport_is_closed") == "1", True)
         .when(F.col("airport_is_closed") == "0", False)
         .otherwise(F.lit(None).cast("boolean"))
    )
    .withColumn(
        "airport_is_latest",
        F.when(F.col("airport_is_latest") == "1", True)
         .when(F.col("airport_is_latest") == "0", False)
         .otherwise(F.lit(None).cast("boolean"))
    )
)


print("=== FINAL AIRPORT DIMENSION TYPES ===")

for column_name, data_type in dim_airport_history_df.dtypes:
    print(f"{column_name:28} {data_type}")


print("\n=== AIRPORT STATUS FLAG VALIDATION ===")

dim_airport_history_df.agg(
    F.count("*").alias("rows"),

    F.sum(
        F.when(F.col("airport_is_closed").isNull(), 1).otherwise(0)
    ).alias("null_closed_flags"),

    F.sum(
        F.when(F.col("airport_is_latest").isNull(), 1).otherwise(0)
    ).alias("null_latest_flags"),

    F.sum(
        F.when(F.col("airport_is_closed") == True, 1).otherwise(0)
    ).alias("closed_records"),

    F.sum(
        F.when(F.col("airport_is_latest") == True, 1).otherwise(0)
    ).alias("latest_records")
).show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Persist reference dimensions as Delta tables
# ---------------------------------------------------------

CARRIER_DIM_TABLE = "workspace.airline_silver.dim_carrier"
AIRPORT_DIM_TABLE = "workspace.airline_silver.dim_airport_history"


# Carrier dimension
(
    dim_carrier_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CARRIER_DIM_TABLE)
)


# Historical airport dimension
(
    dim_airport_history_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(AIRPORT_DIM_TABLE)
)


print("Reference dimensions created successfully.")
print()
print("Carrier dimension:")
print(CARRIER_DIM_TABLE)

print()
print("Airport dimension:")
print(AIRPORT_DIM_TABLE)

# COMMAND ----------

CARRIER_DIM_TABLE = "workspace.airline_silver.dim_carrier"
AIRPORT_DIM_TABLE = "workspace.airline_silver.dim_airport_history"

carrier_dim_table_df = spark.table(CARRIER_DIM_TABLE)
airport_dim_table_df = spark.table(AIRPORT_DIM_TABLE)


print("=== PERSISTED CARRIER DIMENSION ===")

print("Rows:", carrier_dim_table_df.count())

print(
    "Distinct carrier codes:",
    carrier_dim_table_df
        .select("carrier_code")
        .distinct()
        .count()
)

print(
    "Unmatched references:",
    carrier_dim_table_df
        .filter(F.col("reference_match_flag") == False)
        .count()
)


print("\n=== PERSISTED AIRPORT DIMENSION ===")

print("Rows:", airport_dim_table_df.count())

print(
    "Distinct airport codes:",
    airport_dim_table_df
        .select("airport_code")
        .distinct()
        .count()
)

print(
    "Maximum versions per airport:",
    airport_dim_table_df
        .groupBy("airport_code")
        .count()
        .agg(F.max("count"))
        .first()[0]
)

# COMMAND ----------

from pyspark.sql import functions as F


# ---------------------------------------------------------
# Reload persisted Silver and reference dimensions
# ---------------------------------------------------------

flights = spark.table(
    "workspace.airline_silver.flights"
)

carrier_dim = spark.table(
    "workspace.airline_silver.dim_carrier"
)

airport_dim = spark.table(
    "workspace.airline_silver.dim_airport_history"
)


# ---------------------------------------------------------
# Alias and broadcast small reference dimensions
# ---------------------------------------------------------

f = flights.alias("f")

mc = F.broadcast(carrier_dim).alias("mc")
sc = F.broadcast(carrier_dim).alias("sc")
oc = F.broadcast(carrier_dim).alias("oc")

oa = F.broadcast(airport_dim).alias("oa")
da = F.broadcast(airport_dim).alias("da")


# ---------------------------------------------------------
# Enrich flights
# ---------------------------------------------------------

enriched_df = (
    f

    # Marketing carrier
    .join(
        mc,
        F.col("f.marketing_carrier") ==
        F.col("mc.carrier_code"),
        "left"
    )

    # Scheduled operating carrier
    .join(
        sc,
        F.col("f.scheduled_operating_carrier") ==
        F.col("sc.carrier_code"),
        "left"
    )

    # Actual operating carrier
    .join(
        oc,
        F.col("f.operating_carrier") ==
        F.col("oc.carrier_code"),
        "left"
    )

    # Origin airport — date-aware
    .join(
        oa,
        (
            (F.col("f.origin") ==
             F.col("oa.airport_code"))
            &
            (F.col("f.flight_date") >=
             F.col("oa.effective_start_date"))
            &
            (
                F.col("oa.effective_end_date").isNull()
                |
                (
                    F.col("f.flight_date") <=
                    F.col("oa.effective_end_date")
                )
            )
        ),
        "left"
    )

    # Destination airport — date-aware
    .join(
        da,
        (
            (F.col("f.destination") ==
             F.col("da.airport_code"))
            &
            (F.col("f.flight_date") >=
             F.col("da.effective_start_date"))
            &
            (
                F.col("da.effective_end_date").isNull()
                |
                (
                    F.col("f.flight_date") <=
                    F.col("da.effective_end_date")
                )
            )
        ),
        "left"
    )

    # -----------------------------------------------------
    # Original Silver fields + enrichment attributes
    # -----------------------------------------------------
    .select(
        "f.*",

        F.col("mc.carrier_name")
        .alias("marketing_carrier_name"),

        F.col("mc.reference_match_flag")
        .alias("marketing_carrier_reference_match_flag"),

        F.col("sc.carrier_name")
        .alias("scheduled_operating_carrier_name"),

        F.col("oc.carrier_name")
        .alias("operating_carrier_name"),

        F.col("oa.airport_name")
        .alias("origin_airport_name"),

        F.col("oa.city_name")
        .alias("origin_city_name"),

        F.col("oa.state_code")
        .alias("origin_state_code"),

        F.col("oa.country_name")
        .alias("origin_country_name"),

        F.col("oa.latitude")
        .alias("origin_latitude"),

        F.col("oa.longitude")
        .alias("origin_longitude"),

        F.col("da.airport_name")
        .alias("destination_airport_name"),

        F.col("da.city_name")
        .alias("destination_city_name"),

        F.col("da.state_code")
        .alias("destination_state_code"),

        F.col("da.country_name")
        .alias("destination_country_name"),

        F.col("da.latitude")
        .alias("destination_latitude"),

        F.col("da.longitude")
        .alias("destination_longitude")
    )
)


# ---------------------------------------------------------
# Basic row-preservation validation
# ---------------------------------------------------------

print("=== ENRICHED FLIGHT DATAFRAME ===")

print(
    "Original Silver rows:",
    flights.count()
)

print(
    "Enriched rows:",
    enriched_df.count()
)

print(
    "Original Silver columns:",
    len(flights.columns)
)

print(
    "Enriched columns:",
    len(enriched_df.columns)
)

# COMMAND ----------

# ---------------------------------------------------------
# Validate enrichment completeness
# ---------------------------------------------------------

enrichment_validation = enriched_df.agg(

    F.count("*").alias("total_records"),

    F.sum(
        F.when(
            F.col("marketing_carrier_name").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_marketing_carrier_name"),

    F.sum(
        F.when(
            F.col("scheduled_operating_carrier_name").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_scheduled_operator_name"),

    F.sum(
        F.when(
            F.col("operating_carrier_name").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_actual_operator_name"),

    F.sum(
        F.when(
            F.col("origin_airport_name").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_origin_airport"),

    F.sum(
        F.when(
            F.col("destination_airport_name").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_destination_airport"),

    F.sum(
        F.when(
            F.col("origin_latitude").isNull() |
            F.col("origin_longitude").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_origin_coordinates"),

    F.sum(
        F.when(
            F.col("destination_latitude").isNull() |
            F.col("destination_longitude").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_destination_coordinates")
)

print("=== ENRICHMENT COMPLETENESS ===")
enrichment_validation.show(truncate=False)


print("=== UNMATCHED CARRIER CODES BY ROLE ===")

print("Marketing:")
enriched_df.filter(
    F.col("marketing_carrier_name").isNull()
).groupBy(
    "marketing_carrier"
).count().orderBy(
    F.desc("count")
).show(20, truncate=False)


print("Scheduled operating:")
enriched_df.filter(
    F.col("scheduled_operating_carrier_name").isNull()
).groupBy(
    "scheduled_operating_carrier"
).count().orderBy(
    F.desc("count")
).show(20, truncate=False)


print("Actual operating:")
enriched_df.filter(
    F.col("operating_carrier_name").isNull()
).groupBy(
    "operating_carrier"
).count().orderBy(
    F.desc("count")
).show(20, truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Conditional reference validation
# Only treat a missing name as a problem when a source
# carrier code actually exists.
# ---------------------------------------------------------

conditional_validation = enriched_df.agg(

    # Marketing carrier
    F.sum(
        F.when(
            F.col("marketing_carrier").isNotNull(),
            1
        ).otherwise(0)
    ).alias("marketing_codes_present"),

    F.sum(
        F.when(
            F.col("marketing_carrier").isNotNull() &
            F.col("marketing_carrier_name").isNull(),
            1
        ).otherwise(0)
    ).alias("unmatched_marketing_codes"),

    # Scheduled operating carrier
    F.sum(
        F.when(
            F.col("scheduled_operating_carrier").isNotNull(),
            1
        ).otherwise(0)
    ).alias("scheduled_operator_codes_present"),

    F.sum(
        F.when(
            F.col("scheduled_operating_carrier").isNotNull() &
            F.col("scheduled_operating_carrier_name").isNull(),
            1
        ).otherwise(0)
    ).alias("unmatched_scheduled_operator_codes"),

    # Actual operating carrier
    F.sum(
        F.when(
            F.col("operating_carrier").isNotNull(),
            1
        ).otherwise(0)
    ).alias("actual_operator_codes_present"),

    F.sum(
        F.when(
            F.col("operating_carrier").isNotNull() &
            F.col("operating_carrier_name").isNull(),
            1
        ).otherwise(0)
    ).alias("unmatched_actual_operator_codes"),

    # Airports
    F.sum(
        F.when(
            F.col("origin_airport_name").isNull(),
            1
        ).otherwise(0)
    ).alias("unmatched_origin_airports"),

    F.sum(
        F.when(
            F.col("destination_airport_name").isNull(),
            1
        ).otherwise(0)
    ).alias("unmatched_destination_airports"),

    F.sum(
        F.when(
            F.col("origin_latitude").isNull() |
            F.col("origin_longitude").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_origin_coordinates"),

    F.sum(
        F.when(
            F.col("destination_latitude").isNull() |
            F.col("destination_longitude").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_destination_coordinates")
)

print("=== CONDITIONAL ENRICHMENT VALIDATION ===")
conditional_validation.show(truncate=False)


print("=== SCHEDULED OPERATOR BY FORM TYPE ===")

enriched_df.groupBy(
    "form_type"
).agg(
    F.count("*").alias("records"),

    F.sum(
        F.when(
            F.col("scheduled_operating_carrier").isNotNull(),
            1
        ).otherwise(0)
    ).alias("scheduled_operator_populated"),

    F.sum(
        F.when(
            F.col("scheduled_operating_carrier").isNotNull() &
            F.col("scheduled_operating_carrier_name").isNull(),
            1
        ).otherwise(0)
    ).alias("scheduled_operator_unmatched")
).orderBy("form_type").show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Persist enriched Silver flight table
# ---------------------------------------------------------

ENRICHED_TABLE = "workspace.airline_silver.flights_enriched"

(
    enriched_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(ENRICHED_TABLE)
)

print("Enriched Silver table created successfully.")
print("Table:", ENRICHED_TABLE)

# COMMAND ----------

# ---------------------------------------------------------
# Validate persisted enriched Silver table
# ---------------------------------------------------------

ENRICHED_TABLE = "workspace.airline_silver.flights_enriched"

enriched_table_df = spark.table(ENRICHED_TABLE)


print("=== PERSISTED ENRICHED SILVER TABLE ===")

print(
    "Rows:",
    enriched_table_df.count()
)

print(
    "Columns:",
    len(enriched_table_df.columns)
)

print(
    "Distinct months:",
    enriched_table_df
        .select("source_month")
        .distinct()
        .count()
)

print(
    "Flight-date range:",
    enriched_table_df
        .agg(
            F.min("flight_date").alias("min_date"),
            F.max("flight_date").alias("max_date")
        )
        .first()
)


print("\n=== REFERENCE ENRICHMENT CHECK ===")

enriched_table_df.agg(

    F.sum(
        F.when(
            F.col("marketing_carrier").isNotNull() &
            F.col("marketing_carrier_name").isNull(),
            1
        ).otherwise(0)
    ).alias("unmatched_marketing_carriers"),

    F.sum(
        F.when(
            F.col("scheduled_operating_carrier").isNotNull() &
            F.col("scheduled_operating_carrier_name").isNull(),
            1
        ).otherwise(0)
    ).alias("unmatched_scheduled_operators"),

    F.sum(
        F.when(
            F.col("operating_carrier").isNotNull() &
            F.col("operating_carrier_name").isNull(),
            1
        ).otherwise(0)
    ).alias("unmatched_actual_operators"),

    F.sum(
        F.when(
            F.col("origin_airport_name").isNull(),
            1
        ).otherwise(0)
    ).alias("unmatched_origins"),

    F.sum(
        F.when(
            F.col("destination_airport_name").isNull(),
            1
        ).otherwise(0)
    ).alias("unmatched_destinations")

).show(truncate=False)