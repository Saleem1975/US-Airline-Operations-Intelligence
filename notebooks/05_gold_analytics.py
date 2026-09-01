# Databricks notebook source
from pyspark.sql import functions as F


# ---------------------------------------------------------
# Phase 8 — Gold Analytics
# Load enriched Silver source
# ---------------------------------------------------------

ENRICHED_TABLE = "workspace.airline_silver.flights_enriched"
GOLD_SCHEMA = "workspace.airline_gold"

flights_enriched = spark.table(ENRICHED_TABLE)


# ---------------------------------------------------------
# Canonical analytical population
#
# Exclude the 175 superseded duplicate records.
# Keep cancellations, diversions, delays, etc.,
# because they are genuine operational outcomes.
# ---------------------------------------------------------

analytics_flights = (
    flights_enriched
    .filter(
        F.col("canonical_operation_flag") == True
    )
)


print("=== GOLD ANALYTICS BASE POPULATION ===")

print(
    "Enriched Silver records:",
    flights_enriched.count()
)

print(
    "Canonical analytics records:",
    analytics_flights.count()
)

print(
    "Superseded records excluded:",
    flights_enriched.count() - analytics_flights.count()
)

print(
    "Months represented:",
    analytics_flights
        .select("source_month")
        .distinct()
        .count()
)

print(
    "Flight-date range:",
    analytics_flights
        .agg(
            F.min("flight_date").alias("min_date"),
            F.max("flight_date").alias("max_date")
        )
        .first()
)

# COMMAND ----------

# ---------------------------------------------------------
# Gold Table 1 — Monthly Operating Carrier KPIs
# ---------------------------------------------------------

gold_monthly_carrier_kpi = (
    analytics_flights
    .groupBy(
        "source_year",
        "source_month",
        "operating_carrier",
        "operating_carrier_name"
    )
    .agg(

        # Total canonical flight operations
        F.count("*")
            .alias("total_operations"),

        # Operational cancellations
        F.sum(
            F.when(
                F.col("operational_cancellation_flag") == True,
                1
            ).otherwise(0)
        ).alias("cancelled_operations"),

        # Flights with a usable arrival-delay measurement
        F.sum(
            F.when(
                F.col("arrival_delay_minutes").isNotNull(),
                1
            ).otherwise(0)
        ).alias("arrival_delay_observations"),

        # Arrivals 15+ minutes late
        F.sum(
            F.when(
                F.col("arrival_delay_15_plus_flag") == True,
                1
            ).otherwise(0)
        ).alias("late_arrivals_15_plus"),

        # Average delay measures
        F.round(
            F.avg("departure_delay_minutes"),
            2
        ).alias("avg_departure_delay_minutes"),

        F.round(
            F.avg("arrival_delay_minutes"),
            2
        ).alias("avg_arrival_delay_minutes"),

        # Operational disruption indicator
        F.sum(
            F.when(
                F.col("has_diversion_event_data") == True,
                1
            ).otherwise(0)
        ).alias("diversion_event_operations")
    )

    # -----------------------------------------------------
    # Derived KPI rates
    # -----------------------------------------------------

    .withColumn(
        "cancellation_rate_pct",
        F.round(
            100.0 *
            F.col("cancelled_operations") /
            F.col("total_operations"),
            2
        )
    )

    .withColumn(
        "late_arrival_rate_pct",
        F.round(
            100.0 *
            F.col("late_arrivals_15_plus") /
            F.col("arrival_delay_observations"),
            2
        )
    )

    .withColumn(
        "diversion_event_rate_pct",
        F.round(
            100.0 *
            F.col("diversion_event_operations") /
            F.col("total_operations"),
            3
        )
    )

    .orderBy(
        "source_year",
        "source_month",
        "operating_carrier"
    )
)


print("=== MONTHLY CARRIER KPI TABLE ===")

print(
    "Rows:",
    gold_monthly_carrier_kpi.count()
)

print(
    "Distinct operating carriers:",
    gold_monthly_carrier_kpi
        .select("operating_carrier")
        .distinct()
        .count()
)

print(
    "Months represented:",
    gold_monthly_carrier_kpi
        .select("source_month")
        .distinct()
        .count()
)


display(
    gold_monthly_carrier_kpi.orderBy(
        F.desc("total_operations")
    )
)

# COMMAND ----------

# ---------------------------------------------------------
# Validate Gold carrier KPI totals against analytics source
# ---------------------------------------------------------

source_totals = analytics_flights.agg(

    F.count("*")
        .alias("source_total_operations"),

    F.sum(
        F.when(
            F.col("operational_cancellation_flag") == True,
            1
        ).otherwise(0)
    ).alias("source_cancelled_operations"),

    F.sum(
        F.when(
            F.col("arrival_delay_minutes").isNotNull(),
            1
        ).otherwise(0)
    ).alias("source_arrival_delay_observations"),

    F.sum(
        F.when(
            F.col("arrival_delay_15_plus_flag") == True,
            1
        ).otherwise(0)
    ).alias("source_late_arrivals_15_plus"),

    F.sum(
        F.when(
            F.col("has_diversion_event_data") == True,
            1
        ).otherwise(0)
    ).alias("source_diversion_events")
).first()


gold_totals = gold_monthly_carrier_kpi.agg(

    F.sum("total_operations")
        .alias("gold_total_operations"),

    F.sum("cancelled_operations")
        .alias("gold_cancelled_operations"),

    F.sum("arrival_delay_observations")
        .alias("gold_arrival_delay_observations"),

    F.sum("late_arrivals_15_plus")
        .alias("gold_late_arrivals_15_plus"),

    F.sum("diversion_event_operations")
        .alias("gold_diversion_events")
).first()


print("=== GOLD KPI RECONCILIATION ===")

print(
    "Total operations:",
    source_totals["source_total_operations"],
    "| Gold:",
    gold_totals["gold_total_operations"]
)

print(
    "Cancelled operations:",
    source_totals["source_cancelled_operations"],
    "| Gold:",
    gold_totals["gold_cancelled_operations"]
)

print(
    "Arrival-delay observations:",
    source_totals["source_arrival_delay_observations"],
    "| Gold:",
    gold_totals["gold_arrival_delay_observations"]
)

print(
    "Late arrivals 15+:",
    source_totals["source_late_arrivals_15_plus"],
    "| Gold:",
    gold_totals["gold_late_arrivals_15_plus"]
)

print(
    "Diversion-event operations:",
    source_totals["source_diversion_events"],
    "| Gold:",
    gold_totals["gold_diversion_events"]
)


print("\n=== KPI RATE VALIDATION ===")

gold_monthly_carrier_kpi.agg(

    F.min("cancellation_rate_pct")
        .alias("min_cancellation_rate"),

    F.max("cancellation_rate_pct")
        .alias("max_cancellation_rate"),

    F.min("late_arrival_rate_pct")
        .alias("min_late_arrival_rate"),

    F.max("late_arrival_rate_pct")
        .alias("max_late_arrival_rate"),

    F.min("diversion_event_rate_pct")
        .alias("min_diversion_rate"),

    F.max("diversion_event_rate_pct")
        .alias("max_diversion_rate"),

    F.sum(
        F.when(
            F.col("late_arrival_rate_pct").isNull(),
            1
        ).otherwise(0)
    ).alias("null_late_arrival_rates")

).show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Validate the correct arrival-performance denominator
# ---------------------------------------------------------

arrival_population_check = analytics_flights.agg(

    F.count("*")
        .alias("total_canonical_operations"),

    F.sum(
        F.when(
            F.col("operational_cancellation_flag") == False,
            1
        ).otherwise(0)
    ).alias("non_cancelled_operations"),

    F.sum(
        F.when(
            (F.col("operational_cancellation_flag") == False) &
            F.col("actual_gate_arrival").isNotNull(),
            1
        ).otherwise(0)
    ).alias("completed_arrival_operations"),

    F.sum(
        F.when(
            (F.col("operational_cancellation_flag") == False) &
            F.col("actual_gate_arrival").isNotNull() &
            (F.col("arrival_delay_minutes") >= 15),
            1
        ).otherwise(0)
    ).alias("completed_late_arrivals_15_plus"),

    F.sum(
        F.when(
            (F.col("operational_cancellation_flag") == False) &
            F.col("actual_gate_arrival").isNull(),
            1
        ).otherwise(0)
    ).alias("non_cancelled_without_gate_arrival")
)

print("=== ARRIVAL PERFORMANCE POPULATION CHECK ===")
arrival_population_check.show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Gold Table 1 — Corrected Monthly Operating Carrier KPIs
# ---------------------------------------------------------

gold_monthly_carrier_kpi = (
    analytics_flights
    .groupBy(
        "source_year",
        "source_month",
        "operating_carrier",
        "operating_carrier_name"
    )
    .agg(

        # -------------------------------------------------
        # Overall operations
        # -------------------------------------------------

        F.count("*")
            .alias("total_operations"),

        F.sum(
            F.when(
                F.col("operational_cancellation_flag") == True,
                1
            ).otherwise(0)
        ).alias("cancelled_operations"),

        # -------------------------------------------------
        # Completed-arrival analytical population
        # -------------------------------------------------

        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_arrival").isNotNull(),
                1
            ).otherwise(0)
        ).alias("completed_arrivals"),

        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_arrival").isNotNull() &
                (F.col("arrival_delay_minutes") >= 15),
                1
            ).otherwise(0)
        ).alias("late_arrivals_15_plus"),

        # -------------------------------------------------
        # Average departure delay
        # Only non-cancelled flights with actual departure
        # -------------------------------------------------

        F.round(
            F.avg(
                F.when(
                    (F.col("operational_cancellation_flag") == False) &
                    F.col("actual_gate_departure").isNotNull(),
                    F.col("departure_delay_minutes")
                )
            ),
            2
        ).alias("avg_departure_delay_minutes"),

        # -------------------------------------------------
        # Average arrival delay
        # Only completed arrivals
        # -------------------------------------------------

        F.round(
            F.avg(
                F.when(
                    (F.col("operational_cancellation_flag") == False) &
                    F.col("actual_gate_arrival").isNotNull(),
                    F.col("arrival_delay_minutes")
                )
            ),
            2
        ).alias("avg_arrival_delay_minutes"),

        # -------------------------------------------------
        # Diversion events
        # -------------------------------------------------

        F.sum(
            F.when(
                F.col("has_diversion_event_data") == True,
                1
            ).otherwise(0)
        ).alias("diversion_event_operations")
    )

    # -----------------------------------------------------
    # KPI rates
    # -----------------------------------------------------

    .withColumn(
        "cancellation_rate_pct",
        F.round(
            100.0 *
            F.col("cancelled_operations") /
            F.col("total_operations"),
            2
        )
    )

    .withColumn(
        "late_arrival_rate_pct",
        F.round(
            100.0 *
            F.col("late_arrivals_15_plus") /
            F.col("completed_arrivals"),
            2
        )
    )

    .withColumn(
        "diversion_event_rate_pct",
        F.round(
            100.0 *
            F.col("diversion_event_operations") /
            F.col("total_operations"),
            3
        )
    )

    .orderBy(
        "source_year",
        "source_month",
        "operating_carrier"
    )
)


print("=== CORRECTED MONTHLY CARRIER KPI ===")

print("Rows:", gold_monthly_carrier_kpi.count())

print(
    "Distinct operating carriers:",
    gold_monthly_carrier_kpi
        .select("operating_carrier")
        .distinct()
        .count()
)

print(
    "Months represented:",
    gold_monthly_carrier_kpi
        .select("source_month")
        .distinct()
        .count()
)

# COMMAND ----------

# ---------------------------------------------------------
# Final reconciliation of corrected monthly carrier KPIs
# ---------------------------------------------------------

source_corrected_totals = analytics_flights.agg(

    F.count("*").alias("total_operations"),

    F.sum(
        F.when(
            F.col("operational_cancellation_flag") == True,
            1
        ).otherwise(0)
    ).alias("cancelled_operations"),

    F.sum(
        F.when(
            (F.col("operational_cancellation_flag") == False) &
            F.col("actual_gate_arrival").isNotNull(),
            1
        ).otherwise(0)
    ).alias("completed_arrivals"),

    F.sum(
        F.when(
            (F.col("operational_cancellation_flag") == False) &
            F.col("actual_gate_arrival").isNotNull() &
            (F.col("arrival_delay_minutes") >= 15),
            1
        ).otherwise(0)
    ).alias("late_arrivals_15_plus"),

    F.sum(
        F.when(
            F.col("has_diversion_event_data") == True,
            1
        ).otherwise(0)
    ).alias("diversion_event_operations")

).first()


gold_corrected_totals = gold_monthly_carrier_kpi.agg(

    F.sum("total_operations")
        .alias("total_operations"),

    F.sum("cancelled_operations")
        .alias("cancelled_operations"),

    F.sum("completed_arrivals")
        .alias("completed_arrivals"),

    F.sum("late_arrivals_15_plus")
        .alias("late_arrivals_15_plus"),

    F.sum("diversion_event_operations")
        .alias("diversion_event_operations")

).first()


print("=== FINAL CORRECTED GOLD RECONCILIATION ===")

for metric in [
    "total_operations",
    "cancelled_operations",
    "completed_arrivals",
    "late_arrivals_15_plus",
    "diversion_event_operations"
]:

    source_value = source_corrected_totals[metric]
    gold_value = gold_corrected_totals[metric]

    print(
        f"{metric}: "
        f"Source = {source_value:,} | "
        f"Gold = {gold_value:,} | "
        f"Match = {source_value == gold_value}"
    )


print("\n=== FINAL KPI RATE CHECK ===")

gold_monthly_carrier_kpi.agg(

    F.min("cancellation_rate_pct")
        .alias("min_cancellation_rate"),

    F.max("cancellation_rate_pct")
        .alias("max_cancellation_rate"),

    F.min("late_arrival_rate_pct")
        .alias("min_late_arrival_rate"),

    F.max("late_arrival_rate_pct")
        .alias("max_late_arrival_rate"),

    F.min("diversion_event_rate_pct")
        .alias("min_diversion_rate"),

    F.max("diversion_event_rate_pct")
        .alias("max_diversion_rate")

).show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Persist Gold monthly carrier KPI table
# ---------------------------------------------------------

GOLD_SCHEMA = "workspace.airline_gold"

MONTHLY_CARRIER_KPI_TABLE = (
    "workspace.airline_gold.monthly_carrier_kpi"
)


# Create Gold schema if it does not already exist
spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS {GOLD_SCHEMA}"
)


# Persist as Delta
(
    gold_monthly_carrier_kpi.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(MONTHLY_CARRIER_KPI_TABLE)
)


print("Gold monthly carrier KPI table created successfully.")
print("Table:", MONTHLY_CARRIER_KPI_TABLE)

# COMMAND ----------

# ---------------------------------------------------------
# Read back and validate persisted Gold carrier KPI table
# ---------------------------------------------------------

MONTHLY_CARRIER_KPI_TABLE = (
    "workspace.airline_gold.monthly_carrier_kpi"
)

gold_carrier_table_df = spark.table(
    MONTHLY_CARRIER_KPI_TABLE
)


print("=== PERSISTED GOLD CARRIER KPI TABLE ===")

print(
    "Rows:",
    gold_carrier_table_df.count()
)

print(
    "Columns:",
    len(gold_carrier_table_df.columns)
)

print(
    "Distinct carriers:",
    gold_carrier_table_df
        .select("operating_carrier")
        .distinct()
        .count()
)

print(
    "Months represented:",
    gold_carrier_table_df
        .select("source_month")
        .distinct()
        .count()
)


print("\n=== PERSISTED KPI TOTALS ===")

gold_carrier_table_df.agg(

    F.sum("total_operations")
        .alias("total_operations"),

    F.sum("cancelled_operations")
        .alias("cancelled_operations"),

    F.sum("completed_arrivals")
        .alias("completed_arrivals"),

    F.sum("late_arrivals_15_plus")
        .alias("late_arrivals_15_plus"),

    F.sum("diversion_event_operations")
        .alias("diversion_event_operations")

).show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Gold Table 2A — Monthly Airport Departure KPIs
# ---------------------------------------------------------

gold_airport_departure_kpi = (
    analytics_flights
    .groupBy(
        "source_year",
        "source_month",
        F.col("origin").alias("airport_code")
    )
    .agg(

        # Use the most recent airport metadata
        # applicable within that month
        F.max_by(
            "origin_airport_name",
            "flight_date"
        ).alias("airport_name"),

        F.max_by(
            "origin_city_name",
            "flight_date"
        ).alias("city_name"),

        F.max_by(
            "origin_state_code",
            "flight_date"
        ).alias("state_code"),

        # ---------------------------------------------
        # Operations
        # ---------------------------------------------

        F.count("*")
            .alias("total_departure_operations"),

        F.sum(
            F.when(
                F.col("operational_cancellation_flag") == True,
                1
            ).otherwise(0)
        ).alias("cancelled_departures"),

        # Non-cancelled operations with actual departure
        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_departure").isNotNull(),
                1
            ).otherwise(0)
        ).alias("completed_departures"),

        # Departure delay 15+
        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_departure").isNotNull() &
                (F.col("departure_delay_minutes") >= 15),
                1
            ).otherwise(0)
        ).alias("late_departures_15_plus"),

        # Average departure delay
        F.round(
            F.avg(
                F.when(
                    (F.col("operational_cancellation_flag") == False) &
                    F.col("actual_gate_departure").isNotNull(),
                    F.col("departure_delay_minutes")
                )
            ),
            2
        ).alias("avg_departure_delay_minutes"),

        # Diversion events
        F.sum(
            F.when(
                F.col("has_diversion_event_data") == True,
                1
            ).otherwise(0)
        ).alias("diversion_event_departures")
    )

    # ---------------------------------------------
    # KPI rates
    # ---------------------------------------------

    .withColumn(
        "cancellation_rate_pct",
        F.round(
            100.0 *
            F.col("cancelled_departures") /
            F.col("total_departure_operations"),
            2
        )
    )

    .withColumn(
        "late_departure_rate_pct",
        F.round(
            100.0 *
            F.col("late_departures_15_plus") /
            F.col("completed_departures"),
            2
        )
    )

    .withColumn(
        "diversion_event_rate_pct",
        F.round(
            100.0 *
            F.col("diversion_event_departures") /
            F.col("total_departure_operations"),
            3
        )
    )
)


print("=== MONTHLY AIRPORT DEPARTURE KPI ===")

print(
    "Rows:",
    gold_airport_departure_kpi.count()
)

print(
    "Distinct airports:",
    gold_airport_departure_kpi
        .select("airport_code")
        .distinct()
        .count()
)

print(
    "Months represented:",
    gold_airport_departure_kpi
        .select("source_month")
        .distinct()
        .count()
)


display(
    gold_airport_departure_kpi
    .orderBy(
        F.desc("total_departure_operations")
    )
)

# COMMAND ----------

# ---------------------------------------------------------
# Reconcile monthly airport departure KPIs to source
# ---------------------------------------------------------

source_departure_totals = analytics_flights.agg(

    F.count("*")
        .alias("total_departure_operations"),

    F.sum(
        F.when(
            F.col("operational_cancellation_flag") == True,
            1
        ).otherwise(0)
    ).alias("cancelled_departures"),

    F.sum(
        F.when(
            (F.col("operational_cancellation_flag") == False) &
            F.col("actual_gate_departure").isNotNull(),
            1
        ).otherwise(0)
    ).alias("completed_departures"),

    F.sum(
        F.when(
            (F.col("operational_cancellation_flag") == False) &
            F.col("actual_gate_departure").isNotNull() &
            (F.col("departure_delay_minutes") >= 15),
            1
        ).otherwise(0)
    ).alias("late_departures_15_plus"),

    F.sum(
        F.when(
            F.col("has_diversion_event_data") == True,
            1
        ).otherwise(0)
    ).alias("diversion_event_departures")

).first()


gold_departure_totals = gold_airport_departure_kpi.agg(

    F.sum("total_departure_operations")
        .alias("total_departure_operations"),

    F.sum("cancelled_departures")
        .alias("cancelled_departures"),

    F.sum("completed_departures")
        .alias("completed_departures"),

    F.sum("late_departures_15_plus")
        .alias("late_departures_15_plus"),

    F.sum("diversion_event_departures")
        .alias("diversion_event_departures")

).first()


print("=== AIRPORT DEPARTURE KPI RECONCILIATION ===")

for metric in [
    "total_departure_operations",
    "cancelled_departures",
    "completed_departures",
    "late_departures_15_plus",
    "diversion_event_departures"
]:

    source_value = source_departure_totals[metric]
    gold_value = gold_departure_totals[metric]

    print(
        f"{metric}: "
        f"Source = {source_value:,} | "
        f"Gold = {gold_value:,} | "
        f"Match = {source_value == gold_value}"
    )

# COMMAND ----------

# ---------------------------------------------------------
# Gold Table 2B — Monthly Airport Arrival KPIs
# ---------------------------------------------------------

gold_airport_arrival_kpi = (
    analytics_flights
    .groupBy(
        "source_year",
        "source_month",
        F.col("destination").alias("airport_code")
    )
    .agg(

        # -------------------------------------------------
        # Descriptive airport metadata
        # -------------------------------------------------

        F.max_by(
            "destination_airport_name",
            "flight_date"
        ).alias("airport_name"),

        F.max_by(
            "destination_city_name",
            "flight_date"
        ).alias("city_name"),

        F.max_by(
            "destination_state_code",
            "flight_date"
        ).alias("state_code"),

        # -------------------------------------------------
        # Scheduled inbound operations
        # -------------------------------------------------

        F.count("*")
            .alias("scheduled_arrival_operations"),

        # -------------------------------------------------
        # Completed arrivals
        # -------------------------------------------------

        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_arrival").isNotNull(),
                1
            ).otherwise(0)
        ).alias("completed_arrivals"),

        # -------------------------------------------------
        # Late arrivals 15+
        # -------------------------------------------------

        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_arrival").isNotNull() &
                (F.col("arrival_delay_minutes") >= 15),
                1
            ).otherwise(0)
        ).alias("late_arrivals_15_plus"),

        # -------------------------------------------------
        # Average arrival delay
        # -------------------------------------------------

        F.round(
            F.avg(
                F.when(
                    (F.col("operational_cancellation_flag") == False) &
                    F.col("actual_gate_arrival").isNotNull(),
                    F.col("arrival_delay_minutes")
                )
            ),
            2
        ).alias("avg_arrival_delay_minutes"),

        # -------------------------------------------------
        # Flights destined here with diversion activity
        # -------------------------------------------------

        F.sum(
            F.when(
                F.col("has_diversion_event_data") == True,
                1
            ).otherwise(0)
        ).alias("diversion_event_arrivals")
    )

    # -----------------------------------------------------
    # Derived KPIs
    # -----------------------------------------------------

    .withColumn(
        "arrival_completion_rate_pct",
        F.round(
            100.0 *
            F.col("completed_arrivals") /
            F.col("scheduled_arrival_operations"),
            2
        )
    )

    .withColumn(
        "late_arrival_rate_pct",
        F.round(
            100.0 *
            F.col("late_arrivals_15_plus") /
            F.col("completed_arrivals"),
            2
        )
    )

    .withColumn(
        "diversion_event_rate_pct",
        F.round(
            100.0 *
            F.col("diversion_event_arrivals") /
            F.col("scheduled_arrival_operations"),
            3
        )
    )
)


print("=== MONTHLY AIRPORT ARRIVAL KPI ===")

print(
    "Rows:",
    gold_airport_arrival_kpi.count()
)

print(
    "Distinct airports:",
    gold_airport_arrival_kpi
        .select("airport_code")
        .distinct()
        .count()
)

print(
    "Months represented:",
    gold_airport_arrival_kpi
        .select("source_month")
        .distinct()
        .count()
)


display(
    gold_airport_arrival_kpi
    .orderBy(
        F.desc("scheduled_arrival_operations")
    )
)

# COMMAND ----------

# ---------------------------------------------------------
# Reconcile monthly airport arrival KPIs to source
# ---------------------------------------------------------

source_arrival_totals = analytics_flights.agg(

    F.count("*")
        .alias("scheduled_arrival_operations"),

    F.sum(
        F.when(
            (F.col("operational_cancellation_flag") == False) &
            F.col("actual_gate_arrival").isNotNull(),
            1
        ).otherwise(0)
    ).alias("completed_arrivals"),

    F.sum(
        F.when(
            (F.col("operational_cancellation_flag") == False) &
            F.col("actual_gate_arrival").isNotNull() &
            (F.col("arrival_delay_minutes") >= 15),
            1
        ).otherwise(0)
    ).alias("late_arrivals_15_plus"),

    F.sum(
        F.when(
            F.col("has_diversion_event_data") == True,
            1
        ).otherwise(0)
    ).alias("diversion_event_arrivals")

).first()


gold_arrival_totals = gold_airport_arrival_kpi.agg(

    F.sum("scheduled_arrival_operations")
        .alias("scheduled_arrival_operations"),

    F.sum("completed_arrivals")
        .alias("completed_arrivals"),

    F.sum("late_arrivals_15_plus")
        .alias("late_arrivals_15_plus"),

    F.sum("diversion_event_arrivals")
        .alias("diversion_event_arrivals")

).first()


print("=== AIRPORT ARRIVAL KPI RECONCILIATION ===")

for metric in [
    "scheduled_arrival_operations",
    "completed_arrivals",
    "late_arrivals_15_plus",
    "diversion_event_arrivals"
]:

    source_value = source_arrival_totals[metric]
    gold_value = gold_arrival_totals[metric]

    print(
        f"{metric}: "
        f"Source = {source_value:,} | "
        f"Gold = {gold_value:,} | "
        f"Match = {source_value == gold_value}"
    )


print("\n=== AIRPORT ARRIVAL RATE CHECK ===")

gold_airport_arrival_kpi.agg(

    F.min("arrival_completion_rate_pct")
        .alias("min_completion_rate"),

    F.max("arrival_completion_rate_pct")
        .alias("max_completion_rate"),

    F.min("late_arrival_rate_pct")
        .alias("min_late_arrival_rate"),

    F.max("late_arrival_rate_pct")
        .alias("max_late_arrival_rate"),

    F.min("diversion_event_rate_pct")
        .alias("min_diversion_rate"),

    F.max("diversion_event_rate_pct")
        .alias("max_diversion_rate")

).show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Persist monthly airport Gold KPI tables
# ---------------------------------------------------------

AIRPORT_DEPARTURE_KPI_TABLE = (
    "workspace.airline_gold.monthly_airport_departure_kpi"
)

AIRPORT_ARRIVAL_KPI_TABLE = (
    "workspace.airline_gold.monthly_airport_arrival_kpi"
)


# Departure KPI table
(
    gold_airport_departure_kpi.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(AIRPORT_DEPARTURE_KPI_TABLE)
)


# Arrival KPI table
(
    gold_airport_arrival_kpi.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(AIRPORT_ARRIVAL_KPI_TABLE)
)


print("Airport Gold KPI tables created successfully.")

print()
print("Departure table:")
print(AIRPORT_DEPARTURE_KPI_TABLE)

print()
print("Arrival table:")
print(AIRPORT_ARRIVAL_KPI_TABLE)

# COMMAND ----------

# ---------------------------------------------------------
# Read back and validate persisted airport Gold tables
# ---------------------------------------------------------

departure_gold = spark.table(
    "workspace.airline_gold.monthly_airport_departure_kpi"
)

arrival_gold = spark.table(
    "workspace.airline_gold.monthly_airport_arrival_kpi"
)


print("=== PERSISTED AIRPORT DEPARTURE KPI ===")

print("Rows:", departure_gold.count())
print("Columns:", len(departure_gold.columns))

print(
    "Distinct airports:",
    departure_gold
        .select("airport_code")
        .distinct()
        .count()
)

print(
    "Months:",
    departure_gold
        .select("source_month")
        .distinct()
        .count()
)


print("\n=== PERSISTED AIRPORT ARRIVAL KPI ===")

print("Rows:", arrival_gold.count())
print("Columns:", len(arrival_gold.columns))

print(
    "Distinct airports:",
    arrival_gold
        .select("airport_code")
        .distinct()
        .count()
)

print(
    "Months:",
    arrival_gold
        .select("source_month")
        .distinct()
        .count()
)

# COMMAND ----------

# ---------------------------------------------------------
# Gold Table 4 — Monthly Route Performance KPIs
# Grain: year × month × origin × destination
# ---------------------------------------------------------

gold_monthly_route_kpi = (
    analytics_flights
    .groupBy(
        "source_year",
        "source_month",
        "origin",
        "destination"
    )
    .agg(

        # -------------------------------------------------
        # Descriptive route metadata
        # -------------------------------------------------

        F.max_by(
            "origin_airport_name",
            "flight_date"
        ).alias("origin_airport_name"),

        F.max_by(
            "origin_city_name",
            "flight_date"
        ).alias("origin_city_name"),

        F.max_by(
            "destination_airport_name",
            "flight_date"
        ).alias("destination_airport_name"),

        F.max_by(
            "destination_city_name",
            "flight_date"
        ).alias("destination_city_name"),

        # -------------------------------------------------
        # Route volume
        # -------------------------------------------------

        F.count("*")
            .alias("total_operations"),

        # -------------------------------------------------
        # Cancellations
        # -------------------------------------------------

        F.sum(
            F.when(
                F.col("operational_cancellation_flag") == True,
                1
            ).otherwise(0)
        ).alias("cancelled_operations"),

        # -------------------------------------------------
        # Completed arrivals
        # -------------------------------------------------

        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_arrival").isNotNull(),
                1
            ).otherwise(0)
        ).alias("completed_arrivals"),

        # -------------------------------------------------
        # Late arrivals 15+
        # -------------------------------------------------

        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_arrival").isNotNull() &
                (F.col("arrival_delay_minutes") >= 15),
                1
            ).otherwise(0)
        ).alias("late_arrivals_15_plus"),

        # -------------------------------------------------
        # Average delays
        # -------------------------------------------------

        F.round(
            F.avg(
                F.when(
                    (F.col("operational_cancellation_flag") == False) &
                    F.col("actual_gate_departure").isNotNull(),
                    F.col("departure_delay_minutes")
                )
            ),
            2
        ).alias("avg_departure_delay_minutes"),

        F.round(
            F.avg(
                F.when(
                    (F.col("operational_cancellation_flag") == False) &
                    F.col("actual_gate_arrival").isNotNull(),
                    F.col("arrival_delay_minutes")
                )
            ),
            2
        ).alias("avg_arrival_delay_minutes"),

        # -------------------------------------------------
        # Diversion activity
        # -------------------------------------------------

        F.sum(
            F.when(
                F.col("has_diversion_event_data") == True,
                1
            ).otherwise(0)
        ).alias("diversion_event_operations")
    )

    # -----------------------------------------------------
    # Route identifier
    # -----------------------------------------------------

    .withColumn(
        "route",
        F.concat_ws(" → ", "origin", "destination")
    )

    # -----------------------------------------------------
    # KPI rates
    # -----------------------------------------------------

    .withColumn(
        "cancellation_rate_pct",
        F.round(
            100.0 *
            F.col("cancelled_operations") /
            F.col("total_operations"),
            2
        )
    )

    .withColumn(
        "completion_rate_pct",
        F.round(
            100.0 *
            F.col("completed_arrivals") /
            F.col("total_operations"),
            2
        )
    )

    .withColumn(
        "late_arrival_rate_pct",
        F.round(
            100.0 *
            F.col("late_arrivals_15_plus") /
            F.col("completed_arrivals"),
            2
        )
    )

    .withColumn(
        "diversion_event_rate_pct",
        F.round(
            100.0 *
            F.col("diversion_event_operations") /
            F.col("total_operations"),
            3
        )
    )
)


print("=== MONTHLY ROUTE KPI ===")

print(
    "Rows:",
    gold_monthly_route_kpi.count()
)

print(
    "Distinct directional routes:",
    gold_monthly_route_kpi
        .select("origin", "destination")
        .distinct()
        .count()
)

print(
    "Months represented:",
    gold_monthly_route_kpi
        .select("source_month")
        .distinct()
        .count()
)


display(
    gold_monthly_route_kpi
    .orderBy(
        F.desc("total_operations")
    )
)

# COMMAND ----------

# ---------------------------------------------------------
# Fix route KPI division-by-zero
#
# A late-arrival rate is undefined when there are
# zero completed arrivals, so return NULL rather than 0%.
# ---------------------------------------------------------

gold_monthly_route_kpi = (
    gold_monthly_route_kpi

    .withColumn(
        "late_arrival_rate_pct",
        F.when(
            F.col("completed_arrivals") > 0,
            F.round(
                100.0 *
                F.col("late_arrivals_15_plus") /
                F.col("completed_arrivals"),
                2
            )
        ).otherwise(
            F.lit(None).cast("double")
        )
    )
)


print("=== ROUTE KPI ZERO-DENOMINATOR CHECK ===")

gold_monthly_route_kpi.agg(

    F.sum(
        F.when(
            F.col("completed_arrivals") == 0,
            1
        ).otherwise(0)
    ).alias("route_months_with_zero_completed_arrivals"),

    F.sum(
        F.when(
            F.col("late_arrival_rate_pct").isNull(),
            1
        ).otherwise(0)
    ).alias("null_late_arrival_rates")

).show(truncate=False)


print("=== MONTHLY ROUTE KPI ===")

print("Rows:", gold_monthly_route_kpi.count())

print(
    "Distinct directional routes:",
    gold_monthly_route_kpi
        .select("origin", "destination")
        .distinct()
        .count()
)

print(
    "Months represented:",
    gold_monthly_route_kpi
        .select("source_month")
        .distinct()
        .count()
)


display(
    gold_monthly_route_kpi
    .orderBy(
        F.desc("total_operations")
    )
)

# COMMAND ----------

# ---------------------------------------------------------
# Reconcile monthly route KPIs to canonical analytics source
# ---------------------------------------------------------

source_route_totals = analytics_flights.agg(

    F.count("*")
        .alias("total_operations"),

    F.sum(
        F.when(
            F.col("operational_cancellation_flag") == True,
            1
        ).otherwise(0)
    ).alias("cancelled_operations"),

    F.sum(
        F.when(
            (F.col("operational_cancellation_flag") == False) &
            F.col("actual_gate_arrival").isNotNull(),
            1
        ).otherwise(0)
    ).alias("completed_arrivals"),

    F.sum(
        F.when(
            (F.col("operational_cancellation_flag") == False) &
            F.col("actual_gate_arrival").isNotNull() &
            (F.col("arrival_delay_minutes") >= 15),
            1
        ).otherwise(0)
    ).alias("late_arrivals_15_plus"),

    F.sum(
        F.when(
            F.col("has_diversion_event_data") == True,
            1
        ).otherwise(0)
    ).alias("diversion_event_operations")

).first()


gold_route_totals = gold_monthly_route_kpi.agg(

    F.sum("total_operations")
        .alias("total_operations"),

    F.sum("cancelled_operations")
        .alias("cancelled_operations"),

    F.sum("completed_arrivals")
        .alias("completed_arrivals"),

    F.sum("late_arrivals_15_plus")
        .alias("late_arrivals_15_plus"),

    F.sum("diversion_event_operations")
        .alias("diversion_event_operations")

).first()


print("=== ROUTE KPI RECONCILIATION ===")

for metric in [
    "total_operations",
    "cancelled_operations",
    "completed_arrivals",
    "late_arrivals_15_plus",
    "diversion_event_operations"
]:

    source_value = source_route_totals[metric]
    gold_value = gold_route_totals[metric]

    print(
        f"{metric}: "
        f"Source = {source_value:,} | "
        f"Gold = {gold_value:,} | "
        f"Match = {source_value == gold_value}"
    )


print("\n=== ROUTE KPI RATE CHECK ===")

gold_monthly_route_kpi.agg(

    F.min("cancellation_rate_pct")
        .alias("min_cancellation_rate"),

    F.max("cancellation_rate_pct")
        .alias("max_cancellation_rate"),

    F.min("completion_rate_pct")
        .alias("min_completion_rate"),

    F.max("completion_rate_pct")
        .alias("max_completion_rate"),

    F.min("late_arrival_rate_pct")
        .alias("min_late_arrival_rate"),

    F.max("late_arrival_rate_pct")
        .alias("max_late_arrival_rate"),

    F.min("diversion_event_rate_pct")
        .alias("min_diversion_rate"),

    F.max("diversion_event_rate_pct")
        .alias("max_diversion_rate"),

    F.sum(
        F.when(
            F.col("late_arrival_rate_pct").isNull(),
            1
        ).otherwise(0)
    ).alias("null_late_arrival_rates")

).show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Persist monthly route KPI Gold table
# ---------------------------------------------------------

MONTHLY_ROUTE_KPI_TABLE = (
    "workspace.airline_gold.monthly_route_kpi"
)


(
    gold_monthly_route_kpi.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(MONTHLY_ROUTE_KPI_TABLE)
)


print("Gold monthly route KPI table created successfully.")
print("Table:", MONTHLY_ROUTE_KPI_TABLE)

# COMMAND ----------

# ---------------------------------------------------------
# Validate persisted monthly route KPI Gold table
# ---------------------------------------------------------

MONTHLY_ROUTE_KPI_TABLE = (
    "workspace.airline_gold.monthly_route_kpi"
)

route_gold = spark.table(
    MONTHLY_ROUTE_KPI_TABLE
)


print("=== PERSISTED MONTHLY ROUTE KPI ===")

print(
    "Rows:",
    route_gold.count()
)

print(
    "Columns:",
    len(route_gold.columns)
)

print(
    "Distinct directional routes:",
    route_gold
        .select("origin", "destination")
        .distinct()
        .count()
)

print(
    "Months:",
    route_gold
        .select("source_month")
        .distinct()
        .count()
)


print("\n=== PERSISTED ROUTE KPI TOTALS ===")

route_gold.agg(

    F.sum("total_operations")
        .alias("total_operations"),

    F.sum("cancelled_operations")
        .alias("cancelled_operations"),

    F.sum("completed_arrivals")
        .alias("completed_arrivals"),

    F.sum("late_arrivals_15_plus")
        .alias("late_arrivals_15_plus"),

    F.sum("diversion_event_operations")
        .alias("diversion_event_operations"),

    F.sum(
        F.when(
            F.col("late_arrival_rate_pct").isNull(),
            1
        ).otherwise(0)
    ).alias("null_late_arrival_rates")

).show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Gold Table 5 — Delay Cause Analytics
# Step 1: Establish eligible late-flight population
# ---------------------------------------------------------

delay_cause_flights = (
    analytics_flights
    .filter(
        (F.col("operational_cancellation_flag") == False) &
        (F.col("actual_gate_arrival").isNotNull()) &
        (F.col("arrival_delay_minutes") >= 15) &
        (F.col("has_diversion_event_data") == False)
    )
)


# ---------------------------------------------------------
# Reconcile the five BTS delay causes to arrival delay
# ---------------------------------------------------------

delay_cause_validation = (
    delay_cause_flights
    .withColumn(
        "reported_cause_minutes",
        F.coalesce(F.col("carrier_delay_minutes"), F.lit(0)) +
        F.coalesce(F.col("weather_delay_minutes"), F.lit(0)) +
        F.coalesce(F.col("nas_delay_minutes"), F.lit(0)) +
        F.coalesce(F.col("security_delay_minutes"), F.lit(0)) +
        F.coalesce(F.col("late_aircraft_delay_minutes"), F.lit(0))
    )
)


print("=== DELAY CAUSE ANALYTICAL POPULATION ===")

delay_cause_validation.agg(

    F.count("*")
        .alias("eligible_late_flights"),

    F.sum(
        F.when(
            F.col("reported_cause_minutes") ==
            F.col("arrival_delay_minutes"),
            1
        ).otherwise(0)
    ).alias("exact_reconciliations"),

    F.sum(
        F.when(
            F.col("reported_cause_minutes") !=
            F.col("arrival_delay_minutes"),
            1
        ).otherwise(0)
    ).alias("reconciliation_mismatches"),

    F.sum(
        F.when(
            F.col("reported_cause_minutes") == 0,
            1
        ).otherwise(0)
    ).alias("zero_reported_cause_minutes"),

    F.max(
        F.abs(
            F.col("reported_cause_minutes") -
            F.col("arrival_delay_minutes")
        )
    ).alias("max_absolute_difference")

).show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Gold Table 5 — Monthly Carrier Delay-Cause KPIs
# Grain: year × month × actual operating carrier
# Population: ordinary completed flights arriving 15+ min late
# ---------------------------------------------------------

gold_monthly_delay_cause_kpi = (
    delay_cause_validation
    .groupBy(
        "source_year",
        "source_month",
        "operating_carrier",
        "operating_carrier_name"
    )
    .agg(

        # ---------------------------------------------
        # Eligible late-flight volume
        # ---------------------------------------------

        F.count("*")
            .alias("eligible_late_flights"),

        F.sum("arrival_delay_minutes")
            .alias("total_arrival_delay_minutes"),

        # ---------------------------------------------
        # Delay minutes by cause
        # ---------------------------------------------

        F.sum("carrier_delay_minutes")
            .alias("carrier_delay_minutes"),

        F.sum("weather_delay_minutes")
            .alias("weather_delay_minutes"),

        F.sum("nas_delay_minutes")
            .alias("nas_delay_minutes"),

        F.sum("security_delay_minutes")
            .alias("security_delay_minutes"),

        F.sum("late_aircraft_delay_minutes")
            .alias("late_aircraft_delay_minutes"),

        # ---------------------------------------------
        # Number of flights affected by each cause
        # Causes may overlap on the same flight
        # ---------------------------------------------

        F.sum(
            F.when(
                F.col("carrier_delay_minutes") > 0,
                1
            ).otherwise(0)
        ).alias("flights_with_carrier_delay"),

        F.sum(
            F.when(
                F.col("weather_delay_minutes") > 0,
                1
            ).otherwise(0)
        ).alias("flights_with_weather_delay"),

        F.sum(
            F.when(
                F.col("nas_delay_minutes") > 0,
                1
            ).otherwise(0)
        ).alias("flights_with_nas_delay"),

        F.sum(
            F.when(
                F.col("security_delay_minutes") > 0,
                1
            ).otherwise(0)
        ).alias("flights_with_security_delay"),

        F.sum(
            F.when(
                F.col("late_aircraft_delay_minutes") > 0,
                1
            ).otherwise(0)
        ).alias("flights_with_late_aircraft_delay")
    )

    # -----------------------------------------------------
    # Share of total delay minutes attributable to each cause
    # -----------------------------------------------------

    .withColumn(
        "carrier_delay_share_pct",
        F.round(
            100.0 *
            F.col("carrier_delay_minutes") /
            F.col("total_arrival_delay_minutes"),
            2
        )
    )

    .withColumn(
        "weather_delay_share_pct",
        F.round(
            100.0 *
            F.col("weather_delay_minutes") /
            F.col("total_arrival_delay_minutes"),
            2
        )
    )

    .withColumn(
        "nas_delay_share_pct",
        F.round(
            100.0 *
            F.col("nas_delay_minutes") /
            F.col("total_arrival_delay_minutes"),
            2
        )
    )

    .withColumn(
        "security_delay_share_pct",
        F.round(
            100.0 *
            F.col("security_delay_minutes") /
            F.col("total_arrival_delay_minutes"),
            2
        )
    )

    .withColumn(
        "late_aircraft_delay_share_pct",
        F.round(
            100.0 *
            F.col("late_aircraft_delay_minutes") /
            F.col("total_arrival_delay_minutes"),
            2
        )
    )
)


print("=== MONTHLY CARRIER DELAY-CAUSE KPI ===")

print(
    "Rows:",
    gold_monthly_delay_cause_kpi.count()
)

print(
    "Distinct operating carriers:",
    gold_monthly_delay_cause_kpi
        .select("operating_carrier")
        .distinct()
        .count()
)

print(
    "Months represented:",
    gold_monthly_delay_cause_kpi
        .select("source_month")
        .distinct()
        .count()
)


display(
    gold_monthly_delay_cause_kpi
    .orderBy(
        F.desc("total_arrival_delay_minutes")
    )
)

# COMMAND ----------

# ---------------------------------------------------------
# Reconcile Gold delay-cause KPIs to validated source
# ---------------------------------------------------------

source_delay_totals = delay_cause_validation.agg(

    F.count("*")
        .alias("eligible_late_flights"),

    F.sum("arrival_delay_minutes")
        .alias("total_arrival_delay_minutes"),

    F.sum("carrier_delay_minutes")
        .alias("carrier_delay_minutes"),

    F.sum("weather_delay_minutes")
        .alias("weather_delay_minutes"),

    F.sum("nas_delay_minutes")
        .alias("nas_delay_minutes"),

    F.sum("security_delay_minutes")
        .alias("security_delay_minutes"),

    F.sum("late_aircraft_delay_minutes")
        .alias("late_aircraft_delay_minutes")

).first()


gold_delay_totals = gold_monthly_delay_cause_kpi.agg(

    F.sum("eligible_late_flights")
        .alias("eligible_late_flights"),

    F.sum("total_arrival_delay_minutes")
        .alias("total_arrival_delay_minutes"),

    F.sum("carrier_delay_minutes")
        .alias("carrier_delay_minutes"),

    F.sum("weather_delay_minutes")
        .alias("weather_delay_minutes"),

    F.sum("nas_delay_minutes")
        .alias("nas_delay_minutes"),

    F.sum("security_delay_minutes")
        .alias("security_delay_minutes"),

    F.sum("late_aircraft_delay_minutes")
        .alias("late_aircraft_delay_minutes")

).first()


print("=== DELAY-CAUSE GOLD RECONCILIATION ===")

for metric in [
    "eligible_late_flights",
    "total_arrival_delay_minutes",
    "carrier_delay_minutes",
    "weather_delay_minutes",
    "nas_delay_minutes",
    "security_delay_minutes",
    "late_aircraft_delay_minutes"
]:

    source_value = source_delay_totals[metric]
    gold_value = gold_delay_totals[metric]

    print(
        f"{metric}: "
        f"Source = {source_value:,} | "
        f"Gold = {gold_value:,} | "
        f"Match = {source_value == gold_value}"
    )


# ---------------------------------------------------------
# Verify that the five cause totals equal arrival delay
# ---------------------------------------------------------

cause_total = (
    gold_delay_totals["carrier_delay_minutes"] +
    gold_delay_totals["weather_delay_minutes"] +
    gold_delay_totals["nas_delay_minutes"] +
    gold_delay_totals["security_delay_minutes"] +
    gold_delay_totals["late_aircraft_delay_minutes"]
)

print("\n=== CAUSE-MINUTE RECONCILIATION ===")

print(
    "Total arrival delay minutes:",
    f"{gold_delay_totals['total_arrival_delay_minutes']:,}"
)

print(
    "Sum of five delay causes:",
    f"{cause_total:,}"
)

print(
    "Exact match:",
    cause_total ==
    gold_delay_totals["total_arrival_delay_minutes"]
)

# COMMAND ----------

# ---------------------------------------------------------
# Overall 2025 delay-cause composition
# ---------------------------------------------------------

overall_delay_cause_summary = (
    delay_cause_validation
    .agg(

        F.sum("arrival_delay_minutes")
            .alias("total_delay_minutes"),

        F.sum("carrier_delay_minutes")
            .alias("carrier_delay_minutes"),

        F.sum("weather_delay_minutes")
            .alias("weather_delay_minutes"),

        F.sum("nas_delay_minutes")
            .alias("nas_delay_minutes"),

        F.sum("security_delay_minutes")
            .alias("security_delay_minutes"),

        F.sum("late_aircraft_delay_minutes")
            .alias("late_aircraft_delay_minutes")
    )

    .withColumn(
        "carrier_delay_share_pct",
        F.round(
            100.0 *
            F.col("carrier_delay_minutes") /
            F.col("total_delay_minutes"),
            2
        )
    )

    .withColumn(
        "weather_delay_share_pct",
        F.round(
            100.0 *
            F.col("weather_delay_minutes") /
            F.col("total_delay_minutes"),
            2
        )
    )

    .withColumn(
        "nas_delay_share_pct",
        F.round(
            100.0 *
            F.col("nas_delay_minutes") /
            F.col("total_delay_minutes"),
            2
        )
    )

    .withColumn(
        "security_delay_share_pct",
        F.round(
            100.0 *
            F.col("security_delay_minutes") /
            F.col("total_delay_minutes"),
            2
        )
    )

    .withColumn(
        "late_aircraft_delay_share_pct",
        F.round(
            100.0 *
            F.col("late_aircraft_delay_minutes") /
            F.col("total_delay_minutes"),
            2
        )
    )
)


print("=== 2025 OVERALL DELAY-CAUSE COMPOSITION ===")

display(overall_delay_cause_summary)

# COMMAND ----------

# ---------------------------------------------------------
# Persist monthly carrier delay-cause Gold table
# ---------------------------------------------------------

MONTHLY_DELAY_CAUSE_KPI_TABLE = (
    "workspace.airline_gold.monthly_delay_cause_kpi"
)


(
    gold_monthly_delay_cause_kpi.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(MONTHLY_DELAY_CAUSE_KPI_TABLE)
)


print("Gold monthly delay-cause KPI table created successfully.")
print("Table:", MONTHLY_DELAY_CAUSE_KPI_TABLE)

# COMMAND ----------

# ---------------------------------------------------------
# Validate persisted monthly delay-cause KPI Gold table
# ---------------------------------------------------------

MONTHLY_DELAY_CAUSE_KPI_TABLE = (
    "workspace.airline_gold.monthly_delay_cause_kpi"
)

delay_cause_gold = spark.table(
    MONTHLY_DELAY_CAUSE_KPI_TABLE
)


print("=== PERSISTED MONTHLY DELAY-CAUSE KPI ===")

print(
    "Rows:",
    delay_cause_gold.count()
)

print(
    "Columns:",
    len(delay_cause_gold.columns)
)

print(
    "Distinct carriers:",
    delay_cause_gold
        .select("operating_carrier")
        .distinct()
        .count()
)

print(
    "Months:",
    delay_cause_gold
        .select("source_month")
        .distinct()
        .count()
)


print("\n=== PERSISTED DELAY-CAUSE TOTALS ===")

persisted_delay_totals = delay_cause_gold.agg(

    F.sum("eligible_late_flights")
        .alias("eligible_late_flights"),

    F.sum("total_arrival_delay_minutes")
        .alias("total_arrival_delay_minutes"),

    F.sum("carrier_delay_minutes")
        .alias("carrier_delay_minutes"),

    F.sum("weather_delay_minutes")
        .alias("weather_delay_minutes"),

    F.sum("nas_delay_minutes")
        .alias("nas_delay_minutes"),

    F.sum("security_delay_minutes")
        .alias("security_delay_minutes"),

    F.sum("late_aircraft_delay_minutes")
        .alias("late_aircraft_delay_minutes")

).first()


for metric in [
    "eligible_late_flights",
    "total_arrival_delay_minutes",
    "carrier_delay_minutes",
    "weather_delay_minutes",
    "nas_delay_minutes",
    "security_delay_minutes",
    "late_aircraft_delay_minutes"
]:

    print(
        f"{metric}: "
        f"{persisted_delay_totals[metric]:,}"
    )