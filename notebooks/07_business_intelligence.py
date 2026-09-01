# Databricks notebook source
from pyspark.sql import functions as F


# ---------------------------------------------------------
# Phase 10 — Business Intelligence
# Executive-level 2025 KPI summary
# ---------------------------------------------------------

flights = (
    spark.table(
        "workspace.airline_silver.flights_enriched"
    )
    .filter(
        F.col("canonical_operation_flag") == True
    )
)


executive_kpi_2025 = (
    flights
    .agg(

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
        ).alias("diversion_event_operations"),

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

        F.countDistinct("operating_carrier")
            .alias("operating_carriers"),

        F.countDistinct("origin")
            .alias("airports"),

        F.countDistinct(
            F.struct("origin", "destination")
        ).alias("directional_routes")
    )

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
)


print("=== 2025 EXECUTIVE AIRLINE OPERATIONS KPI ===")

display(executive_kpi_2025)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 2
# Compact executive KPI summary
# ---------------------------------------------------------

executive_row = executive_kpi_2025.first()


print("=== 2025 EXECUTIVE AIRLINE OPERATIONS SUMMARY ===")

print(
    "Total operations:",
    f"{executive_row['total_operations']:,}"
)

print(
    "Cancelled operations:",
    f"{executive_row['cancelled_operations']:,}"
)

print(
    "Cancellation rate:",
    f"{executive_row['cancellation_rate_pct']:.2f}%"
)

print(
    "Completed arrivals:",
    f"{executive_row['completed_arrivals']:,}"
)

print(
    "Late arrivals 15+:",
    f"{executive_row['late_arrivals_15_plus']:,}"
)

print(
    "Late-arrival rate:",
    f"{executive_row['late_arrival_rate_pct']:.2f}%"
)

print(
    "Average departure delay:",
    f"{executive_row['avg_departure_delay_minutes']:.2f} minutes"
)

print(
    "Average arrival delay:",
    f"{executive_row['avg_arrival_delay_minutes']:.2f} minutes"
)

print(
    "Diversion-event operations:",
    f"{executive_row['diversion_event_operations']:,}"
)

print(
    "Diversion-event rate:",
    f"{executive_row['diversion_event_rate_pct']:.3f}%"
)

print(
    "Operating carriers:",
    executive_row["operating_carriers"]
)

print(
    "Airports:",
    executive_row["airports"]
)

print(
    "Directional routes:",
    f"{executive_row['directional_routes']:,}"
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 3
# Monthly network performance trend
# ---------------------------------------------------------

monthly_network_kpi = (
    flights
    .groupBy(
        "source_year",
        "source_month"
    )
    .agg(

        # ---------------------------------------------
        # Network volume
        # ---------------------------------------------

        F.count("*")
            .alias("total_operations"),

        # ---------------------------------------------
        # Cancellations
        # ---------------------------------------------

        F.sum(
            F.when(
                F.col("operational_cancellation_flag") == True,
                1
            ).otherwise(0)
        ).alias("cancelled_operations"),

        # ---------------------------------------------
        # Completed arrivals
        # ---------------------------------------------

        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_arrival").isNotNull(),
                1
            ).otherwise(0)
        ).alias("completed_arrivals"),

        # ---------------------------------------------
        # Late arrivals 15+
        # ---------------------------------------------

        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_arrival").isNotNull() &
                (F.col("arrival_delay_minutes") >= 15),
                1
            ).otherwise(0)
        ).alias("late_arrivals_15_plus"),

        # ---------------------------------------------
        # Average departure delay
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Average arrival delay
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Diversions
        # ---------------------------------------------

        F.sum(
            F.when(
                F.col("has_diversion_event_data") == True,
                1
            ).otherwise(0)
        ).alias("diversion_event_operations")
    )

    # ---------------------------------------------
    # KPI rates
    # ---------------------------------------------

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
        "on_time_arrival_rate_pct",
        F.round(
            100.0 -
            F.col("late_arrival_rate_pct"),
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

    # Month label useful later for dashboards
    .withColumn(
        "month_start",
        F.make_date(
            F.col("source_year"),
            F.col("source_month"),
            F.lit(1)
        )
    )

    .withColumn(
        "month_name",
        F.date_format(
            F.col("month_start"),
            "MMM"
        )
    )

    .orderBy(
        "source_year",
        "source_month"
    )
)


print("=== 2025 MONTHLY NETWORK PERFORMANCE ===")

print(
    "Rows:",
    monthly_network_kpi.count()
)

display(
    monthly_network_kpi.select(
        "source_month",
        "month_name",
        "total_operations",
        "cancellation_rate_pct",
        "late_arrival_rate_pct",
        "on_time_arrival_rate_pct",
        "avg_departure_delay_minutes",
        "avg_arrival_delay_minutes",
        "diversion_event_rate_pct"
    )
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 4
# Identify best / worst operating months
# ---------------------------------------------------------

print("=== MONTHLY PERFORMANCE EXTREMES ===")


# ---------------------------------------------------------
# Cancellation rate
# ---------------------------------------------------------

best_cancel = (
    monthly_network_kpi
    .orderBy(
        F.asc("cancellation_rate_pct")
    )
    .first()
)

worst_cancel = (
    monthly_network_kpi
    .orderBy(
        F.desc("cancellation_rate_pct")
    )
    .first()
)


print("\nCancellation performance")

print(
    f"Best month:  "
    f"{best_cancel['month_name']} "
    f"({best_cancel['cancellation_rate_pct']:.2f}%)"
)

print(
    f"Worst month: "
    f"{worst_cancel['month_name']} "
    f"({worst_cancel['cancellation_rate_pct']:.2f}%)"
)


# ---------------------------------------------------------
# Late-arrival rate
# ---------------------------------------------------------

best_late = (
    monthly_network_kpi
    .orderBy(
        F.asc("late_arrival_rate_pct")
    )
    .first()
)

worst_late = (
    monthly_network_kpi
    .orderBy(
        F.desc("late_arrival_rate_pct")
    )
    .first()
)


print("\nLate-arrival performance")

print(
    f"Best month:  "
    f"{best_late['month_name']} "
    f"({best_late['late_arrival_rate_pct']:.2f}%)"
)

print(
    f"Worst month: "
    f"{worst_late['month_name']} "
    f"({worst_late['late_arrival_rate_pct']:.2f}%)"
)


# ---------------------------------------------------------
# Average departure delay
# ---------------------------------------------------------

best_dep_delay = (
    monthly_network_kpi
    .orderBy(
        F.asc("avg_departure_delay_minutes")
    )
    .first()
)

worst_dep_delay = (
    monthly_network_kpi
    .orderBy(
        F.desc("avg_departure_delay_minutes")
    )
    .first()
)


print("\nAverage departure delay")

print(
    f"Best month:  "
    f"{best_dep_delay['month_name']} "
    f"({best_dep_delay['avg_departure_delay_minutes']:.2f} min)"
)

print(
    f"Worst month: "
    f"{worst_dep_delay['month_name']} "
    f"({worst_dep_delay['avg_departure_delay_minutes']:.2f} min)"
)


# ---------------------------------------------------------
# Average arrival delay
# ---------------------------------------------------------

best_arr_delay = (
    monthly_network_kpi
    .orderBy(
        F.asc("avg_arrival_delay_minutes")
    )
    .first()
)

worst_arr_delay = (
    monthly_network_kpi
    .orderBy(
        F.desc("avg_arrival_delay_minutes")
    )
    .first()
)


print("\nAverage arrival delay")

print(
    f"Best month:  "
    f"{best_arr_delay['month_name']} "
    f"({best_arr_delay['avg_arrival_delay_minutes']:.2f} min)"
)

print(
    f"Worst month: "
    f"{worst_arr_delay['month_name']} "
    f"({worst_arr_delay['avg_arrival_delay_minutes']:.2f} min)"
)


# ---------------------------------------------------------
# Diversion rate
# ---------------------------------------------------------

best_diversion = (
    monthly_network_kpi
    .orderBy(
        F.asc("diversion_event_rate_pct")
    )
    .first()
)

worst_diversion = (
    monthly_network_kpi
    .orderBy(
        F.desc("diversion_event_rate_pct")
    )
    .first()
)


print("\nDiversion performance")

print(
    f"Lowest month:  "
    f"{best_diversion['month_name']} "
    f"({best_diversion['diversion_event_rate_pct']:.3f}%)"
)

print(
    f"Highest month: "
    f"{worst_diversion['month_name']} "
    f"({worst_diversion['diversion_event_rate_pct']:.3f}%)"
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 5
# Persist monthly network performance for BI/dashboard use
# ---------------------------------------------------------

MONTHLY_NETWORK_BI_TABLE = (
    "workspace.airline_gold.monthly_network_performance"
)


(
    monthly_network_kpi.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(MONTHLY_NETWORK_BI_TABLE)
)


# ---------------------------------------------------------
# Read-back validation
# ---------------------------------------------------------

monthly_network_gold = spark.table(
    MONTHLY_NETWORK_BI_TABLE
)


print("=== PERSISTED MONTHLY NETWORK PERFORMANCE ===")

print(
    "Rows:",
    monthly_network_gold.count()
)

print(
    "Months:",
    monthly_network_gold
        .select("source_month")
        .distinct()
        .count()
)

print(
    "Month range:",
    monthly_network_gold
        .agg(
            F.min("source_month").alias("min_month"),
            F.max("source_month").alias("max_month")
        )
        .first()
)


print("\nTable:")
print(MONTHLY_NETWORK_BI_TABLE)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 6
# 2025 annual operating-carrier performance
# ---------------------------------------------------------

annual_carrier_kpi = (
    flights
    .groupBy(
        "operating_carrier",
        "operating_carrier_name"
    )
    .agg(

        # ---------------------------------------------
        # Volume
        # ---------------------------------------------

        F.count("*")
            .alias("total_operations"),

        # ---------------------------------------------
        # Cancellations
        # ---------------------------------------------

        F.sum(
            F.when(
                F.col("operational_cancellation_flag") == True,
                1
            ).otherwise(0)
        ).alias("cancelled_operations"),

        # ---------------------------------------------
        # Completed departures
        # ---------------------------------------------

        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_departure").isNotNull(),
                1
            ).otherwise(0)
        ).alias("completed_departures"),

        # ---------------------------------------------
        # Completed arrivals
        # ---------------------------------------------

        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_arrival").isNotNull(),
                1
            ).otherwise(0)
        ).alias("completed_arrivals"),

        # ---------------------------------------------
        # Late departures 15+
        # ---------------------------------------------

        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_departure").isNotNull() &
                (F.col("departure_delay_minutes") >= 15),
                1
            ).otherwise(0)
        ).alias("late_departures_15_plus"),

        # ---------------------------------------------
        # Late arrivals 15+
        # ---------------------------------------------

        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_arrival").isNotNull() &
                (F.col("arrival_delay_minutes") >= 15),
                1
            ).otherwise(0)
        ).alias("late_arrivals_15_plus"),

        # ---------------------------------------------
        # Average delays
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Diversions
        # ---------------------------------------------

        F.sum(
            F.when(
                F.col("has_diversion_event_data") == True,
                1
            ).otherwise(0)
        ).alias("diversion_event_operations")
    )

    # ---------------------------------------------
    # Rates
    # ---------------------------------------------

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
        "late_departure_rate_pct",
        F.round(
            100.0 *
            F.col("late_departures_15_plus") /
            F.col("completed_departures"),
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


print("=== 2025 ANNUAL CARRIER PERFORMANCE ===")

print(
    "Rows:",
    annual_carrier_kpi.count()
)

print(
    "Distinct operating carriers:",
    annual_carrier_kpi
        .select("operating_carrier")
        .distinct()
        .count()
)


display(
    annual_carrier_kpi
    .orderBy(
        F.desc("total_operations")
    )
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 7
# Evaluate carrier-volume thresholds for fair benchmarking
# ---------------------------------------------------------

network_operations = (
    annual_carrier_kpi
    .agg(
        F.sum("total_operations")
            .alias("network_operations")
    )
    .first()["network_operations"]
)


volume_thresholds = [
    50000,
    100000,
    200000,
    500000
]


threshold_results = []

for threshold in volume_thresholds:

    eligible = annual_carrier_kpi.filter(
        F.col("total_operations") >= threshold
    )

    carrier_count = eligible.count()

    covered_operations = (
        eligible
        .agg(
            F.sum("total_operations")
                .alias("covered_operations")
        )
        .first()["covered_operations"]
    )

    coverage_pct = (
        100.0 *
        covered_operations /
        network_operations
    )

    threshold_results.append(
        (
            threshold,
            carrier_count,
            covered_operations,
            round(coverage_pct, 2)
        )
    )


threshold_df = spark.createDataFrame(
    threshold_results,
    [
        "minimum_annual_operations",
        "eligible_carriers",
        "covered_operations",
        "network_coverage_pct"
    ]
)


print("=== CARRIER BENCHMARK THRESHOLD OPTIONS ===")

threshold_df.orderBy(
    "minimum_annual_operations"
).show(truncate=False)


print("\n=== CARRIER VOLUME DISTRIBUTION ===")

annual_carrier_kpi.select(
    "operating_carrier",
    "operating_carrier_name",
    "total_operations"
).orderBy(
    F.desc("total_operations")
).show(
    30,
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 8
# Benchmark major operating carriers
#
# Eligibility threshold:
# >= 100,000 annual operations
# Covers 94.49% of 2025 network operations
# ---------------------------------------------------------

from pyspark.sql.window import Window


MAJOR_CARRIER_THRESHOLD = 100000


major_carriers = (
    annual_carrier_kpi
    .filter(
        F.col("total_operations") >= MAJOR_CARRIER_THRESHOLD
    )
)


# ---------------------------------------------------------
# Rank carriers
# Lower value = better for all selected KPIs
# ---------------------------------------------------------

rank_window_cancel = Window.orderBy(
    F.asc("cancellation_rate_pct")
)

rank_window_late = Window.orderBy(
    F.asc("late_arrival_rate_pct")
)

rank_window_arr_delay = Window.orderBy(
    F.asc("avg_arrival_delay_minutes")
)

rank_window_dep_delay = Window.orderBy(
    F.asc("avg_departure_delay_minutes")
)

rank_window_diversion = Window.orderBy(
    F.asc("diversion_event_rate_pct")
)


carrier_benchmark = (
    major_carriers

    .withColumn(
        "cancellation_rank",
        F.rank().over(rank_window_cancel)
    )

    .withColumn(
        "late_arrival_rank",
        F.rank().over(rank_window_late)
    )

    .withColumn(
        "arrival_delay_rank",
        F.rank().over(rank_window_arr_delay)
    )

    .withColumn(
        "departure_delay_rank",
        F.rank().over(rank_window_dep_delay)
    )

    .withColumn(
        "diversion_rank",
        F.rank().over(rank_window_diversion)
    )

    # Simple transparent composite ranking:
    # equal weight across five operational dimensions
    .withColumn(
        "average_operational_rank",
        F.round(
            (
                F.col("cancellation_rank") +
                F.col("late_arrival_rank") +
                F.col("arrival_delay_rank") +
                F.col("departure_delay_rank") +
                F.col("diversion_rank")
            ) / 5.0,
            2
        )
    )
)


print("=== MAJOR CARRIER BENCHMARK ===")

print(
    "Eligible carriers:",
    carrier_benchmark.count()
)

print(
    "Minimum annual operations:",
    f"{MAJOR_CARRIER_THRESHOLD:,}"
)


display(
    carrier_benchmark
    .select(
        "operating_carrier",
        "operating_carrier_name",
        "total_operations",

        "cancellation_rate_pct",
        "cancellation_rank",

        "late_arrival_rate_pct",
        "late_arrival_rank",

        "avg_arrival_delay_minutes",
        "arrival_delay_rank",

        "avg_departure_delay_minutes",
        "departure_delay_rank",

        "diversion_event_rate_pct",
        "diversion_rank",

        "average_operational_rank"
    )
    .orderBy(
        "average_operational_rank",
        F.desc("total_operations")
    )
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 8A
# Compact final major-carrier benchmark
# ---------------------------------------------------------

carrier_ranking_compact = (
    carrier_benchmark
    .select(
        "operating_carrier",
        "operating_carrier_name",
        "total_operations",

        "cancellation_rate_pct",
        "late_arrival_rate_pct",
        "avg_departure_delay_minutes",
        "avg_arrival_delay_minutes",
        "diversion_event_rate_pct",

        "cancellation_rank",
        "late_arrival_rank",
        "departure_delay_rank",
        "arrival_delay_rank",
        "diversion_rank",

        "average_operational_rank"
    )
    .orderBy(
        "average_operational_rank",
        F.desc("total_operations")
    )
)


print("=== FINAL MAJOR-CARRIER BENCHMARK ===")

rows = carrier_ranking_compact.collect()

for position, row in enumerate(rows, start=1):

    print(
        f"{position:2}. "
        f"{row['operating_carrier']:>2} | "
        f"{row['operating_carrier_name']:<32} | "
        f"Ops={row['total_operations']:,} | "
        f"Cancel={row['cancellation_rate_pct']:.2f}% | "
        f"LateArr={row['late_arrival_rate_pct']:.2f}% | "
        f"DepDelay={row['avg_departure_delay_minutes']:.2f} | "
        f"ArrDelay={row['avg_arrival_delay_minutes']:.2f} | "
        f"Div={row['diversion_event_rate_pct']:.3f}% | "
        f"AvgRank={row['average_operational_rank']:.2f}"
    )

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 8B
# Diagnose best and worst major-carrier performance
# ---------------------------------------------------------

ranked_carriers = (
    carrier_benchmark
    .orderBy(
        F.asc("average_operational_rank"),
        F.desc("total_operations")
    )
)


ranked_rows = ranked_carriers.collect()

top_5 = ranked_rows[:5]
bottom_5 = ranked_rows[-5:]


def print_carrier_detail(position, row):

    print(
        f"{position:2}. "
        f"{row['operating_carrier']} - "
        f"{row['operating_carrier_name']}"
    )

    print(
        f"    Operations: "
        f"{row['total_operations']:,}"
    )

    print(
        f"    Cancellation: "
        f"{row['cancellation_rate_pct']:.2f}% "
        f"(rank {row['cancellation_rank']})"
    )

    print(
        f"    Late arrival: "
        f"{row['late_arrival_rate_pct']:.2f}% "
        f"(rank {row['late_arrival_rank']})"
    )

    print(
        f"    Avg departure delay: "
        f"{row['avg_departure_delay_minutes']:.2f} min "
        f"(rank {row['departure_delay_rank']})"
    )

    print(
        f"    Avg arrival delay: "
        f"{row['avg_arrival_delay_minutes']:.2f} min "
        f"(rank {row['arrival_delay_rank']})"
    )

    print(
        f"    Diversion rate: "
        f"{row['diversion_event_rate_pct']:.3f}% "
        f"(rank {row['diversion_rank']})"
    )

    print(
        f"    Composite average rank: "
        f"{row['average_operational_rank']:.2f}"
    )

    print()


print("=== TOP 5 MAJOR CARRIERS ===")

for i, row in enumerate(top_5, start=1):
    print_carrier_detail(i, row)


print("=== BOTTOM 5 MAJOR CARRIERS ===")

start_position = len(ranked_rows) - 4

for i, row in enumerate(
    bottom_5,
    start=start_position
):
    print_carrier_detail(i, row)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 8C
# Compact Top 5 / Bottom 5 carrier diagnostics
# ---------------------------------------------------------

ranked_rows = (
    carrier_benchmark
    .orderBy(
        F.asc("average_operational_rank"),
        F.desc("total_operations")
    )
    .collect()
)


print("=== TOP 5 MAJOR CARRIERS ===")

for position, row in enumerate(ranked_rows[:5], start=1):

    print(
        f"{position}. "
        f"{row['operating_carrier']} | "
        f"{row['operating_carrier_name']} | "
        f"AvgRank={row['average_operational_rank']:.2f} | "
        f"Cancel={row['cancellation_rate_pct']:.2f}% | "
        f"LateArr={row['late_arrival_rate_pct']:.2f}% | "
        f"DepDelay={row['avg_departure_delay_minutes']:.2f} | "
        f"ArrDelay={row['avg_arrival_delay_minutes']:.2f} | "
        f"Div={row['diversion_event_rate_pct']:.3f}%"
    )


print("\n=== BOTTOM 5 MAJOR CARRIERS ===")

for position, row in enumerate(
    ranked_rows[-5:],
    start=len(ranked_rows) - 4
):

    print(
        f"{position}. "
        f"{row['operating_carrier']} | "
        f"{row['operating_carrier_name']} | "
        f"AvgRank={row['average_operational_rank']:.2f} | "
        f"Cancel={row['cancellation_rate_pct']:.2f}% | "
        f"LateArr={row['late_arrival_rate_pct']:.2f}% | "
        f"DepDelay={row['avg_departure_delay_minutes']:.2f} | "
        f"ArrDelay={row['avg_arrival_delay_minutes']:.2f} | "
        f"Div={row['diversion_event_rate_pct']:.3f}%"
    )

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 9
# Persist annual carrier benchmarking table
# ---------------------------------------------------------

ANNUAL_CARRIER_BENCHMARK_TABLE = (
    "workspace.airline_gold.annual_carrier_benchmark_2025"
)


carrier_benchmark_gold = (
    carrier_benchmark

    .withColumn(
        "minimum_operations_threshold",
        F.lit(100000)
    )

    .withColumn(
        "benchmark_network_coverage_pct",
        F.lit(94.49)
    )
)


(
    carrier_benchmark_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(ANNUAL_CARRIER_BENCHMARK_TABLE)
)


# ---------------------------------------------------------
# Read-back validation
# ---------------------------------------------------------

carrier_benchmark_table = spark.table(
    ANNUAL_CARRIER_BENCHMARK_TABLE
)


print("=== PERSISTED ANNUAL CARRIER BENCHMARK ===")

print(
    "Rows:",
    carrier_benchmark_table.count()
)

print(
    "Distinct carriers:",
    carrier_benchmark_table
        .select("operating_carrier")
        .distinct()
        .count()
)

print(
    "Minimum operations threshold:",
    carrier_benchmark_table
        .select("minimum_operations_threshold")
        .first()[0]
)

print(
    "Network coverage:",
    f"{carrier_benchmark_table.select('benchmark_network_coverage_pct').first()[0]:.2f}%"
)


print("\nTable:")
print(ANNUAL_CARRIER_BENCHMARK_TABLE)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 10
# Build 2025 annual airport performance
#
# One row per airport:
# departure-side + arrival-side operating KPIs
# ---------------------------------------------------------


# ---------------------------------------------------------
# Departure-side annual KPIs
# ---------------------------------------------------------

annual_airport_departure = (
    flights
    .groupBy(
        F.col("origin").alias("airport_code"),
        F.col("origin_airport_name").alias("airport_name"),
        F.col("origin_city_name").alias("city_name"),
        F.col("origin_state_code").alias("state_code")
    )
    .agg(

        F.count("*")
            .alias("departure_operations"),

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

        F.sum(
            F.when(
                F.col("has_diversion_event_data") == True,
                1
            ).otherwise(0)
        ).alias("departure_diversion_events")
    )

    .withColumn(
        "cancellation_rate_pct",
        F.round(
            100.0 *
            F.col("cancelled_departures") /
            F.col("departure_operations"),
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
        "departure_diversion_rate_pct",
        F.round(
            100.0 *
            F.col("departure_diversion_events") /
            F.col("departure_operations"),
            3
        )
    )
)


# ---------------------------------------------------------
# Arrival-side annual KPIs
# ---------------------------------------------------------

annual_airport_arrival = (
    flights
    .groupBy(
        F.col("destination").alias("airport_code")
    )
    .agg(

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

        F.round(
            F.avg(
                F.when(
                    (F.col("operational_cancellation_flag") == False) &
                    F.col("actual_gate_arrival").isNotNull(),
                    F.col("arrival_delay_minutes")
                )
            ),
            2
        ).alias("avg_arrival_delay_minutes")
    )

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
)


# ---------------------------------------------------------
# Combine departure and arrival performance
# ---------------------------------------------------------

annual_airport_kpi = (
    annual_airport_departure.alias("d")

    .join(
        annual_airport_arrival.alias("a"),
        on="airport_code",
        how="inner"
    )

    .select(
        "airport_code",
        "airport_name",
        "city_name",
        "state_code",

        "departure_operations",
        "scheduled_arrival_operations",

        "cancelled_departures",
        "cancellation_rate_pct",

        "completed_departures",
        "late_departures_15_plus",
        "late_departure_rate_pct",
        "avg_departure_delay_minutes",

        "completed_arrivals",
        "arrival_completion_rate_pct",
        "late_arrivals_15_plus",
        "late_arrival_rate_pct",
        "avg_arrival_delay_minutes",

        "departure_diversion_events",
        "departure_diversion_rate_pct"
    )
)


print("=== 2025 ANNUAL AIRPORT PERFORMANCE ===")

print(
    "Rows:",
    annual_airport_kpi.count()
)

print(
    "Distinct airports:",
    annual_airport_kpi
        .select("airport_code")
        .distinct()
        .count()
)


display(
    annual_airport_kpi
    .orderBy(
        F.desc("departure_operations")
    )
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 10A
# Correct annual airport KPI grain
#
# Key principle:
# airport_code = unique entity key
# names / city / state = descriptive attributes
# ---------------------------------------------------------


# ---------------------------------------------------------
# Departure-side annual KPIs
# Group ONLY by airport code
# ---------------------------------------------------------

annual_airport_departure = (
    flights
    .groupBy(
        F.col("origin").alias("airport_code")
    )
    .agg(

        # Latest observed airport descriptive identity
        F.max_by(
            F.struct(
                F.col("origin_airport_name").alias("airport_name"),
                F.col("origin_city_name").alias("city_name"),
                F.col("origin_state_code").alias("state_code")
            ),
            F.col("flight_date")
        ).alias("airport_identity"),

        # Volume
        F.count("*")
            .alias("departure_operations"),

        # Cancellations
        F.sum(
            F.when(
                F.col("operational_cancellation_flag") == True,
                1
            ).otherwise(0)
        ).alias("cancelled_departures"),

        # Completed departures
        F.sum(
            F.when(
                (F.col("operational_cancellation_flag") == False) &
                F.col("actual_gate_departure").isNotNull(),
                1
            ).otherwise(0)
        ).alias("completed_departures"),

        # Late departures
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

        # Diversions
        F.sum(
            F.when(
                F.col("has_diversion_event_data") == True,
                1
            ).otherwise(0)
        ).alias("departure_diversion_events")
    )

    .select(
        "airport_code",

        F.col("airport_identity.airport_name")
            .alias("airport_name"),

        F.col("airport_identity.city_name")
            .alias("city_name"),

        F.col("airport_identity.state_code")
            .alias("state_code"),

        "departure_operations",
        "cancelled_departures",
        "completed_departures",
        "late_departures_15_plus",
        "avg_departure_delay_minutes",
        "departure_diversion_events"
    )

    .withColumn(
        "cancellation_rate_pct",
        F.round(
            100.0 *
            F.col("cancelled_departures") /
            F.col("departure_operations"),
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
        "departure_diversion_rate_pct",
        F.round(
            100.0 *
            F.col("departure_diversion_events") /
            F.col("departure_operations"),
            3
        )
    )
)


# ---------------------------------------------------------
# Arrival-side annual KPIs
# Already grouped by airport code only
# ---------------------------------------------------------

annual_airport_arrival = (
    flights
    .groupBy(
        F.col("destination").alias("airport_code")
    )
    .agg(

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

        F.round(
            F.avg(
                F.when(
                    (F.col("operational_cancellation_flag") == False) &
                    F.col("actual_gate_arrival").isNotNull(),
                    F.col("arrival_delay_minutes")
                )
            ),
            2
        ).alias("avg_arrival_delay_minutes")
    )

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
)


# ---------------------------------------------------------
# Final annual airport table
# ---------------------------------------------------------

annual_airport_kpi = (
    annual_airport_departure.alias("d")

    .join(
        annual_airport_arrival.alias("a"),
        on="airport_code",
        how="inner"
    )

    .select(
        "airport_code",
        "airport_name",
        "city_name",
        "state_code",

        "departure_operations",
        "scheduled_arrival_operations",

        "cancelled_departures",
        "cancellation_rate_pct",

        "completed_departures",
        "late_departures_15_plus",
        "late_departure_rate_pct",
        "avg_departure_delay_minutes",

        "completed_arrivals",
        "arrival_completion_rate_pct",
        "late_arrivals_15_plus",
        "late_arrival_rate_pct",
        "avg_arrival_delay_minutes",

        "departure_diversion_events",
        "departure_diversion_rate_pct"
    )
)


print("=== CORRECTED 2025 ANNUAL AIRPORT PERFORMANCE ===")

print(
    "Rows:",
    annual_airport_kpi.count()
)

print(
    "Distinct airports:",
    annual_airport_kpi
        .select("airport_code")
        .distinct()
        .count()
)

print(
    "Exact one-row-per-airport:",
    annual_airport_kpi.count() ==
    annual_airport_kpi.select("airport_code").distinct().count()
)


display(
    annual_airport_kpi
    .orderBy(
        F.desc("departure_operations")
    )
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 11
# Evaluate airport-volume thresholds for fair benchmarking
# ---------------------------------------------------------

network_departure_operations = (
    annual_airport_kpi
    .agg(
        F.sum("departure_operations")
            .alias("network_departures")
    )
    .first()["network_departures"]
)


airport_volume_thresholds = [
    5000,
    10000,
    25000,
    50000,
    100000
]


airport_threshold_results = []


for threshold in airport_volume_thresholds:

    eligible = (
        annual_airport_kpi
        .filter(
            F.col("departure_operations") >= threshold
        )
    )

    airport_count = eligible.count()

    covered_operations = (
        eligible
        .agg(
            F.sum("departure_operations")
                .alias("covered_operations")
        )
        .first()["covered_operations"]
    )

    coverage_pct = (
        100.0 *
        covered_operations /
        network_departure_operations
    )

    airport_threshold_results.append(
        (
            int(threshold),
            int(airport_count),
            int(covered_operations),
            float(round(coverage_pct, 2))
        )
    )


airport_threshold_df = spark.createDataFrame(
    airport_threshold_results,
    [
        "minimum_annual_departures",
        "eligible_airports",
        "covered_departures",
        "network_coverage_pct"
    ]
)


print("=== AIRPORT BENCHMARK THRESHOLD OPTIONS ===")

airport_threshold_df.orderBy(
    "minimum_annual_departures"
).show(truncate=False)


print("\n=== TOP 25 AIRPORTS BY ANNUAL DEPARTURES ===")

annual_airport_kpi.select(
    "airport_code",
    "airport_name",
    "departure_operations"
).orderBy(
    F.desc("departure_operations")
).show(
    25,
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 12
# Benchmark major airports
#
# Eligibility:
# >= 10,000 annual departures
# Network coverage = 91.85%
# ---------------------------------------------------------

from pyspark.sql.window import Window


MAJOR_AIRPORT_THRESHOLD = 10000


major_airports = (
    annual_airport_kpi
    .filter(
        F.col("departure_operations") >=
        MAJOR_AIRPORT_THRESHOLD
    )
)


# ---------------------------------------------------------
# Global ranking windows
# Lower = better for all selected KPIs
# ---------------------------------------------------------

cancel_window = Window.orderBy(
    F.asc("cancellation_rate_pct")
)

late_dep_window = Window.orderBy(
    F.asc("late_departure_rate_pct")
)

avg_dep_delay_window = Window.orderBy(
    F.asc("avg_departure_delay_minutes")
)

late_arr_window = Window.orderBy(
    F.asc("late_arrival_rate_pct")
)

avg_arr_delay_window = Window.orderBy(
    F.asc("avg_arrival_delay_minutes")
)

diversion_window = Window.orderBy(
    F.asc("departure_diversion_rate_pct")
)


# ---------------------------------------------------------
# Create transparent equal-weight benchmark
# ---------------------------------------------------------

airport_benchmark = (
    major_airports

    .withColumn(
        "cancellation_rank",
        F.rank().over(cancel_window)
    )

    .withColumn(
        "late_departure_rank",
        F.rank().over(late_dep_window)
    )

    .withColumn(
        "departure_delay_rank",
        F.rank().over(avg_dep_delay_window)
    )

    .withColumn(
        "late_arrival_rank",
        F.rank().over(late_arr_window)
    )

    .withColumn(
        "arrival_delay_rank",
        F.rank().over(avg_arr_delay_window)
    )

    .withColumn(
        "diversion_rank",
        F.rank().over(diversion_window)
    )

    .withColumn(
        "average_operational_rank",
        F.round(
            (
                F.col("cancellation_rank") +
                F.col("late_departure_rank") +
                F.col("departure_delay_rank") +
                F.col("late_arrival_rank") +
                F.col("arrival_delay_rank") +
                F.col("diversion_rank")
            ) / 6.0,
            2
        )
    )
)


print("=== MAJOR AIRPORT BENCHMARK ===")

print(
    "Eligible airports:",
    airport_benchmark.count()
)

print(
    "Minimum annual departures:",
    f"{MAJOR_AIRPORT_THRESHOLD:,}"
)

print(
    "Network coverage:",
    "91.85%"
)


# ---------------------------------------------------------
# Show top 15 benchmark airports
# ---------------------------------------------------------

airport_benchmark.select(
    "airport_code",
    "airport_name",
    "departure_operations",

    "cancellation_rate_pct",
    "late_departure_rate_pct",
    "avg_departure_delay_minutes",

    "late_arrival_rate_pct",
    "avg_arrival_delay_minutes",

    "departure_diversion_rate_pct",

    "average_operational_rank"
).orderBy(
    F.asc("average_operational_rank"),
    F.desc("departure_operations")
).show(
    15,
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 13
# Compact Top 10 / Bottom 10 airport benchmark
# ---------------------------------------------------------

ranked_airports = (
    airport_benchmark
    .orderBy(
        F.asc("average_operational_rank"),
        F.desc("departure_operations")
    )
)

ranked_airport_rows = ranked_airports.collect()


print("=== TOP 10 MAJOR AIRPORTS ===")

for position, row in enumerate(
    ranked_airport_rows[:10],
    start=1
):

    print(
        f"{position:3}. "
        f"{row['airport_code']} | "
        f"{row['airport_name']} | "
        f"Ops={row['departure_operations']:,} | "
        f"Cancel={row['cancellation_rate_pct']:.2f}% | "
        f"LateDep={row['late_departure_rate_pct']:.2f}% | "
        f"DepDelay={row['avg_departure_delay_minutes']:.2f} | "
        f"LateArr={row['late_arrival_rate_pct']:.2f}% | "
        f"ArrDelay={row['avg_arrival_delay_minutes']:.2f} | "
        f"Div={row['departure_diversion_rate_pct']:.3f}% | "
        f"AvgRank={row['average_operational_rank']:.2f}"
    )


print("\n=== BOTTOM 10 MAJOR AIRPORTS ===")

for position, row in enumerate(
    ranked_airport_rows[-10:],
    start=len(ranked_airport_rows) - 9
):

    print(
        f"{position:3}. "
        f"{row['airport_code']} | "
        f"{row['airport_name']} | "
        f"Ops={row['departure_operations']:,} | "
        f"Cancel={row['cancellation_rate_pct']:.2f}% | "
        f"LateDep={row['late_departure_rate_pct']:.2f}% | "
        f"DepDelay={row['avg_departure_delay_minutes']:.2f} | "
        f"LateArr={row['late_arrival_rate_pct']:.2f}% | "
        f"ArrDelay={row['avg_arrival_delay_minutes']:.2f} | "
        f"Div={row['departure_diversion_rate_pct']:.3f}% | "
        f"AvgRank={row['average_operational_rank']:.2f}"
    )

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 13A
# Show missing #10 and Bottom 10 airports only
# ---------------------------------------------------------

print("=== RANK #10 ===")

row = ranked_airport_rows[9]

print(
    f"10. "
    f"{row['airport_code']} | "
    f"{row['airport_name']} | "
    f"Ops={row['departure_operations']:,} | "
    f"Cancel={row['cancellation_rate_pct']:.2f}% | "
    f"LateDep={row['late_departure_rate_pct']:.2f}% | "
    f"DepDelay={row['avg_departure_delay_minutes']:.2f} | "
    f"LateArr={row['late_arrival_rate_pct']:.2f}% | "
    f"ArrDelay={row['avg_arrival_delay_minutes']:.2f} | "
    f"Div={row['departure_diversion_rate_pct']:.3f}% | "
    f"AvgRank={row['average_operational_rank']:.2f}"
)


print("\n=== BOTTOM 10 MAJOR AIRPORTS ===")

for position, row in enumerate(
    ranked_airport_rows[-10:],
    start=104
):

    print(
        f"{position}. "
        f"{row['airport_code']} | "
        f"{row['airport_name']} | "
        f"Ops={row['departure_operations']:,} | "
        f"Cancel={row['cancellation_rate_pct']:.2f}% | "
        f"LateDep={row['late_departure_rate_pct']:.2f}% | "
        f"DepDelay={row['avg_departure_delay_minutes']:.2f} | "
        f"LateArr={row['late_arrival_rate_pct']:.2f}% | "
        f"ArrDelay={row['avg_arrival_delay_minutes']:.2f} | "
        f"Div={row['departure_diversion_rate_pct']:.3f}% | "
        f"AvgRank={row['average_operational_rank']:.2f}"
    )

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 14
# Cross-analyze annual airport benchmark with
# Phase 9 operational-state clustering
# ---------------------------------------------------------

airport_stability = spark.table(
    "workspace.airline_gold.airport_operational_stability_2025"
)


# ---------------------------------------------------------
# Bottom 10 annual benchmark airports
# ---------------------------------------------------------

bottom_10_airports = (
    airport_benchmark
    .orderBy(
        F.desc("average_operational_rank"),
        F.asc("departure_operations")
    )
    .limit(10)
)


# ---------------------------------------------------------
# Add clustering / stability information
# ---------------------------------------------------------

bottom_10_with_clusters = (
    bottom_10_airports.alias("b")

    .join(
        airport_stability.alias("s"),
        on="airport_code",
        how="left"
    )

    .select(
        "airport_code",

        F.col("b.airport_name")
            .alias("airport_name"),

        "departure_operations",

        "average_operational_rank",

        "cancellation_rate_pct",
        "late_departure_rate_pct",
        "late_arrival_rate_pct",
        "avg_departure_delay_minutes",
        "avg_arrival_delay_minutes",

        "months_delay_stressed",
        "months_stable",
        "months_cancellation_prone",

        "dominant_state",
        "refined_stability_class"
    )

    .orderBy(
        F.desc("average_operational_rank")
    )
)


print("=== BOTTOM 10 AIRPORTS + OPERATIONAL CLUSTER HISTORY ===")

bottom_10_with_clusters.show(
    10,
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 15
# Persist annual airport benchmark + clustering context
# ---------------------------------------------------------

ANNUAL_AIRPORT_BENCHMARK_TABLE = (
    "workspace.airline_gold.annual_airport_benchmark_2025"
)

airport_stability = spark.table(
    "workspace.airline_gold.airport_operational_stability_2025"
)


airport_benchmark_gold = (
    airport_benchmark.alias("b")

    .join(
        airport_stability.alias("s"),
        on="airport_code",
        how="left"
    )

    .select(
        "airport_code",

        F.col("b.airport_name").alias("airport_name"),
        F.col("b.city_name").alias("city_name"),
        F.col("b.state_code").alias("state_code"),

        "departure_operations",
        "scheduled_arrival_operations",

        "cancellation_rate_pct",
        "late_departure_rate_pct",
        "avg_departure_delay_minutes",

        "late_arrival_rate_pct",
        "avg_arrival_delay_minutes",

        "departure_diversion_rate_pct",

        "cancellation_rank",
        "late_departure_rank",
        "departure_delay_rank",
        "late_arrival_rank",
        "arrival_delay_rank",
        "diversion_rank",

        "average_operational_rank",

        # Monthly clustering context
        "months_observed",
        "months_delay_stressed",
        "months_stable",
        "months_cancellation_prone",
        "dominant_state",
        "refined_stability_class"
    )

    # Benchmark methodology metadata
    .withColumn(
        "minimum_departures_threshold",
        F.lit(10000)
    )

    .withColumn(
        "benchmark_network_coverage_pct",
        F.lit(91.85)
    )
)


# ---------------------------------------------------------
# Persist as Delta
# ---------------------------------------------------------

(
    airport_benchmark_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(ANNUAL_AIRPORT_BENCHMARK_TABLE)
)


# ---------------------------------------------------------
# Read-back validation
# ---------------------------------------------------------

airport_benchmark_table = spark.table(
    ANNUAL_AIRPORT_BENCHMARK_TABLE
)


print("=== PERSISTED ANNUAL AIRPORT BENCHMARK ===")

print(
    "Rows:",
    airport_benchmark_table.count()
)

print(
    "Distinct airports:",
    airport_benchmark_table
        .select("airport_code")
        .distinct()
        .count()
)

print(
    "Minimum departure threshold:",
    airport_benchmark_table
        .select("minimum_departures_threshold")
        .first()[0]
)

print(
    "Network coverage:",
    f"{airport_benchmark_table.select('benchmark_network_coverage_pct').first()[0]:.2f}%"
)

print("\nTable:")
print(ANNUAL_AIRPORT_BENCHMARK_TABLE)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 16
# Build 2025 annual directional-route performance
# ---------------------------------------------------------

annual_route_kpi = (
    flights

    .groupBy(
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
        # Volume
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
    # Route label
    # -----------------------------------------------------

    .withColumn(
        "route",
        F.concat_ws(
            " → ",
            "origin",
            "destination"
        )
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


print("=== 2025 ANNUAL ROUTE PERFORMANCE ===")

print(
    "Rows:",
    annual_route_kpi.count()
)

print(
    "Distinct directional routes:",
    annual_route_kpi
        .select("origin", "destination")
        .distinct()
        .count()
)

print(
    "Routes with zero completed arrivals:",
    annual_route_kpi
        .filter(
            F.col("completed_arrivals") == 0
        )
        .count()
)


print("\n=== TOP 20 ROUTES BY ANNUAL VOLUME ===")

annual_route_kpi.select(
    "route",
    "origin_city_name",
    "destination_city_name",
    "total_operations",
    "cancellation_rate_pct",
    "late_arrival_rate_pct",
    "avg_arrival_delay_minutes"
).orderBy(
    F.desc("total_operations")
).show(
    20,
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 17
# Evaluate annual route-volume thresholds
# ---------------------------------------------------------

network_route_operations = (
    annual_route_kpi
    .agg(
        F.sum("total_operations")
            .alias("network_operations")
    )
    .first()["network_operations"]
)


route_volume_thresholds = [
    500,
    1000,
    2000,
    3000,
    5000
]


route_threshold_results = []


for threshold in route_volume_thresholds:

    eligible = (
        annual_route_kpi
        .filter(
            F.col("total_operations") >= threshold
        )
    )

    route_count = eligible.count()

    covered_operations = (
        eligible
        .agg(
            F.sum("total_operations")
                .alias("covered_operations")
        )
        .first()["covered_operations"]
    )

    coverage_pct = (
        100.0 *
        covered_operations /
        network_route_operations
    )

    route_threshold_results.append(
        (
            int(threshold),
            int(route_count),
            int(covered_operations),
            float(round(coverage_pct, 2))
        )
    )


route_threshold_df = spark.createDataFrame(
    route_threshold_results,
    [
        "minimum_annual_operations",
        "eligible_directional_routes",
        "covered_operations",
        "network_coverage_pct"
    ]
)


print("=== ROUTE BENCHMARK THRESHOLD OPTIONS ===")

route_threshold_df.orderBy(
    "minimum_annual_operations"
).show(
    truncate=False
)


print("\n=== ROUTE VOLUME DISTRIBUTION ===")

annual_route_kpi.select(
    F.expr(
        "percentile_approx(total_operations, "
        "array(0.50, 0.75, 0.90, 0.95, 0.99))"
    ).alias("volume_percentiles")
).show(
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 18
# Benchmark major directional routes
#
# Eligibility:
# >= 1,000 annual operations
# Network coverage = 81.68%
# ---------------------------------------------------------

from pyspark.sql.window import Window


MAJOR_ROUTE_THRESHOLD = 1000


major_routes = (
    annual_route_kpi
    .filter(
        F.col("total_operations") >= MAJOR_ROUTE_THRESHOLD
    )
)


# ---------------------------------------------------------
# Global ranking windows
# ---------------------------------------------------------

cancel_window = Window.orderBy(
    F.asc("cancellation_rate_pct")
)

late_arrival_window = Window.orderBy(
    F.asc("late_arrival_rate_pct")
)

departure_delay_window = Window.orderBy(
    F.asc("avg_departure_delay_minutes")
)

arrival_delay_window = Window.orderBy(
    F.asc("avg_arrival_delay_minutes")
)

diversion_window = Window.orderBy(
    F.asc("diversion_event_rate_pct")
)


# ---------------------------------------------------------
# Create equal-weight route benchmark
# ---------------------------------------------------------

route_benchmark = (
    major_routes

    .withColumn(
        "cancellation_rank",
        F.rank().over(cancel_window)
    )

    .withColumn(
        "late_arrival_rank",
        F.rank().over(late_arrival_window)
    )

    .withColumn(
        "departure_delay_rank",
        F.rank().over(departure_delay_window)
    )

    .withColumn(
        "arrival_delay_rank",
        F.rank().over(arrival_delay_window)
    )

    .withColumn(
        "diversion_rank",
        F.rank().over(diversion_window)
    )

    .withColumn(
        "average_operational_rank",
        F.round(
            (
                F.col("cancellation_rank") +
                F.col("late_arrival_rank") +
                F.col("departure_delay_rank") +
                F.col("arrival_delay_rank") +
                F.col("diversion_rank")
            ) / 5.0,
            2
        )
    )
)


print("=== MAJOR ROUTE BENCHMARK ===")

print(
    "Eligible directional routes:",
    route_benchmark.count()
)

print(
    "Minimum annual operations:",
    f"{MAJOR_ROUTE_THRESHOLD:,}"
)

print(
    "Network coverage:",
    "81.68%"
)


print("\n=== TOP 15 MAJOR ROUTES ===")

route_benchmark.select(
    "route",
    "total_operations",
    "cancellation_rate_pct",
    "late_arrival_rate_pct",
    "avg_departure_delay_minutes",
    "avg_arrival_delay_minutes",
    "diversion_event_rate_pct",
    "average_operational_rank"
).orderBy(
    F.asc("average_operational_rank"),
    F.desc("total_operations")
).show(
    15,
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 19
# Bottom 15 major directional routes
# ---------------------------------------------------------

ranked_route_rows = (
    route_benchmark
    .orderBy(
        F.asc("average_operational_rank"),
        F.desc("total_operations")
    )
    .collect()
)


print("=== BOTTOM 15 MAJOR DIRECTIONAL ROUTES ===")

for position, row in enumerate(
    ranked_route_rows[-15:],
    start=len(ranked_route_rows) - 14
):

    print(
        f"{position:4}. "
        f"{row['route']} | "
        f"Ops={row['total_operations']:,} | "
        f"Cancel={row['cancellation_rate_pct']:.2f}% | "
        f"LateArr={row['late_arrival_rate_pct']:.2f}% | "
        f"DepDelay={row['avg_departure_delay_minutes']:.2f} | "
        f"ArrDelay={row['avg_arrival_delay_minutes']:.2f} | "
        f"Div={row['diversion_event_rate_pct']:.3f}% | "
        f"AvgRank={row['average_operational_rank']:.2f}"
    )

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 20
# Diagnose network-node concentration among
# the 15 weakest major directional routes
# ---------------------------------------------------------

bottom_15_routes = (
    route_benchmark
    .orderBy(
        F.desc("average_operational_rank")
    )
    .limit(15)
)


# ---------------------------------------------------------
# Convert every route into two endpoint observations
# ---------------------------------------------------------

bottom_route_endpoints = (
    bottom_15_routes
    .select(
        F.col("origin").alias("airport_code"),
        F.lit("Origin").alias("endpoint_role")
    )

    .unionByName(

        bottom_15_routes
        .select(
            F.col("destination").alias("airport_code"),
            F.lit("Destination").alias("endpoint_role")
        )
    )
)


# ---------------------------------------------------------
# Count appearances by airport
# ---------------------------------------------------------

endpoint_concentration = (
    bottom_route_endpoints
    .groupBy("airport_code")
    .agg(

        F.count("*")
            .alias("bottom_15_route_appearances"),

        F.sum(
            F.when(
                F.col("endpoint_role") == "Origin",
                1
            ).otherwise(0)
        ).alias("origin_appearances"),

        F.sum(
            F.when(
                F.col("endpoint_role") == "Destination",
                1
            ).otherwise(0)
        ).alias("destination_appearances")
    )
)


# ---------------------------------------------------------
# Add annual airport benchmark context
# ---------------------------------------------------------

endpoint_diagnostic = (
    endpoint_concentration.alias("e")

    .join(
        airport_benchmark.select(
            "airport_code",
            "airport_name",
            F.col("average_operational_rank")
                .alias("airport_average_rank"),
            "cancellation_rate_pct",
            "late_arrival_rate_pct"
        ).alias("a"),
        on="airport_code",
        how="left"
    )

    .orderBy(
        F.desc("bottom_15_route_appearances"),
        F.desc("airport_average_rank")
    )
)


print(
    "=== AIRPORT CONCENTRATION WITHIN "
    "BOTTOM 15 MAJOR ROUTES ==="
)

endpoint_diagnostic.show(
    20,
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 21
# Persist 2025 annual major-route benchmark
# ---------------------------------------------------------

ANNUAL_ROUTE_BENCHMARK_TABLE = (
    "workspace.airline_gold.annual_route_benchmark_2025"
)


route_benchmark_gold = (
    route_benchmark

    # -----------------------------------------------------
    # Benchmark methodology metadata
    # -----------------------------------------------------

    .withColumn(
        "minimum_operations_threshold",
        F.lit(1000)
    )

    .withColumn(
        "benchmark_network_coverage_pct",
        F.lit(81.68)
    )
)


# ---------------------------------------------------------
# Persist as Delta
# ---------------------------------------------------------

(
    route_benchmark_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(ANNUAL_ROUTE_BENCHMARK_TABLE)
)


# ---------------------------------------------------------
# Read-back validation
# ---------------------------------------------------------

route_benchmark_table = spark.table(
    ANNUAL_ROUTE_BENCHMARK_TABLE
)


print("=== PERSISTED ANNUAL ROUTE BENCHMARK ===")

print(
    "Rows:",
    route_benchmark_table.count()
)

print(
    "Distinct directional routes:",
    route_benchmark_table
        .select("origin", "destination")
        .distinct()
        .count()
)

print(
    "Minimum operations threshold:",
    route_benchmark_table
        .select("minimum_operations_threshold")
        .first()[0]
)

print(
    "Network coverage:",
    f"{route_benchmark_table.select('benchmark_network_coverage_pct').first()[0]:.2f}%"
)

print("\nTable:")
print(ANNUAL_ROUTE_BENCHMARK_TABLE)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 22
# Inspect existing Gold delay-cause dataset
# ---------------------------------------------------------

delay_cause_gold = spark.table(
    "workspace.airline_gold.monthly_delay_cause_kpi"
)


print("=== MONTHLY DELAY-CAUSE GOLD TABLE ===")

print(
    "Rows:",
    delay_cause_gold.count()
)

print(
    "Columns:",
    len(delay_cause_gold.columns)
)


print("\n=== COLUMN NAMES ===")

for col_name in delay_cause_gold.columns:
    print(col_name)


print("\n=== SAMPLE ROWS ===")

delay_cause_gold.show(
    5,
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 22A
# Print complete delay-cause schema with column numbers
# ---------------------------------------------------------

print("=== COMPLETE DELAY-CAUSE COLUMN LIST ===")

for i, col_name in enumerate(
    delay_cause_gold.columns,
    start=1
):
    print(f"{i:2}. {col_name}")

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 23
# Build annual 2025 network delay-cause decomposition
#
# Important:
# Annual cause shares are calculated from summed minutes,
# NOT by averaging monthly percentage columns.
# ---------------------------------------------------------

annual_delay_totals = (
    delay_cause_gold
    .agg(

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
            .alias("late_aircraft_delay_minutes"),

        F.sum("flights_with_carrier_delay")
            .alias("flights_with_carrier_delay"),

        F.sum("flights_with_weather_delay")
            .alias("flights_with_weather_delay"),

        F.sum("flights_with_nas_delay")
            .alias("flights_with_nas_delay"),

        F.sum("flights_with_security_delay")
            .alias("flights_with_security_delay"),

        F.sum("flights_with_late_aircraft_delay")
            .alias("flights_with_late_aircraft_delay")
    )

    .withColumn(
        "total_cause_minutes",
        F.col("carrier_delay_minutes") +
        F.col("weather_delay_minutes") +
        F.col("nas_delay_minutes") +
        F.col("security_delay_minutes") +
        F.col("late_aircraft_delay_minutes")
    )
)


# ---------------------------------------------------------
# Convert causes into dashboard-friendly long format
# ---------------------------------------------------------

annual_delay_cause_summary = (
    annual_delay_totals

    .selectExpr(
        """
        stack(
            5,

            'Late Aircraft',
            late_aircraft_delay_minutes,
            flights_with_late_aircraft_delay,

            'Carrier',
            carrier_delay_minutes,
            flights_with_carrier_delay,

            'NAS',
            nas_delay_minutes,
            flights_with_nas_delay,

            'Weather',
            weather_delay_minutes,
            flights_with_weather_delay,

            'Security',
            security_delay_minutes,
            flights_with_security_delay
        )
        as (
            delay_cause,
            delay_minutes,
            flights_affected
        )
        """,
        "total_cause_minutes"
    )

    .withColumn(
        "delay_share_pct",
        F.round(
            100.0 *
            F.col("delay_minutes") /
            F.col("total_cause_minutes"),
            2
        )
    )

    .withColumn(
        "avg_cause_minutes_per_affected_flight",
        F.round(
            F.col("delay_minutes") /
            F.col("flights_affected"),
            2
        )
    )

    .orderBy(
        F.desc("delay_minutes")
    )
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

totals = annual_delay_totals.first()


print("=== 2025 NETWORK DELAY-CAUSE TOTALS ===")

print(
    "Eligible late flights:",
    f"{totals['eligible_late_flights']:,}"
)

print(
    "Total arrival-delay minutes:",
    f"{totals['total_arrival_delay_minutes']:,}"
)

print(
    "Sum of five cause minutes:",
    f"{totals['total_cause_minutes']:,}"
)

print(
    "Cause-minute reconciliation:",
    totals["total_arrival_delay_minutes"] ==
    totals["total_cause_minutes"]
)


print("\n=== 2025 DELAY-CAUSE DECOMPOSITION ===")

annual_delay_cause_summary.select(
    "delay_cause",
    "delay_minutes",
    "delay_share_pct",
    "flights_affected",
    "avg_cause_minutes_per_affected_flight"
).show(
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 24
# Persist annual 2025 delay-cause decomposition
# ---------------------------------------------------------

ANNUAL_DELAY_CAUSE_TABLE = (
    "workspace.airline_gold.annual_delay_cause_decomposition_2025"
)


annual_delay_cause_gold = (
    annual_delay_cause_summary

    .withColumn(
        "source_year",
        F.lit(2025)
    )

    .select(
        "source_year",
        "delay_cause",
        "delay_minutes",
        "delay_share_pct",
        "flights_affected",
        "avg_cause_minutes_per_affected_flight"
    )
)


# ---------------------------------------------------------
# Persist
# ---------------------------------------------------------

(
    annual_delay_cause_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(ANNUAL_DELAY_CAUSE_TABLE)
)


# ---------------------------------------------------------
# Read-back validation
# ---------------------------------------------------------

annual_delay_cause_table = spark.table(
    ANNUAL_DELAY_CAUSE_TABLE
)


print("=== PERSISTED ANNUAL DELAY-CAUSE DECOMPOSITION ===")

print(
    "Rows:",
    annual_delay_cause_table.count()
)

print(
    "Total delay minutes:",
    f"{annual_delay_cause_table.agg(F.sum('delay_minutes')).first()[0]:,}"
)

print(
    "Total delay share:",
    f"{annual_delay_cause_table.agg(F.sum('delay_share_pct')).first()[0]:.2f}%"
)

print("\nTable:")
print(ANNUAL_DELAY_CAUSE_TABLE)


annual_delay_cause_table.orderBy(
    F.desc("delay_minutes")
).show(
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 25
# Build annual delay-cause profiles for major carriers
# ---------------------------------------------------------

major_carrier_codes = (
    spark.table(
        "workspace.airline_gold.annual_carrier_benchmark_2025"
    )
    .select("operating_carrier")
    .distinct()
)


# ---------------------------------------------------------
# Aggregate monthly delay-cause data to annual carrier level
# ---------------------------------------------------------

annual_carrier_delay_profile = (
    delay_cause_gold.alias("d")

    .join(
        major_carrier_codes.alias("m"),
        on="operating_carrier",
        how="inner"
    )

    .groupBy(
        "operating_carrier",
        "operating_carrier_name"
    )

    .agg(

        F.sum("eligible_late_flights")
            .alias("eligible_late_flights"),

        F.sum("total_arrival_delay_minutes")
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

    # -----------------------------------------------------
    # Annual cause shares
    # -----------------------------------------------------

    .withColumn(
        "carrier_share_pct",
        F.round(
            100.0 *
            F.col("carrier_delay_minutes") /
            F.col("total_delay_minutes"),
            2
        )
    )

    .withColumn(
        "weather_share_pct",
        F.round(
            100.0 *
            F.col("weather_delay_minutes") /
            F.col("total_delay_minutes"),
            2
        )
    )

    .withColumn(
        "nas_share_pct",
        F.round(
            100.0 *
            F.col("nas_delay_minutes") /
            F.col("total_delay_minutes"),
            2
        )
    )

    .withColumn(
        "security_share_pct",
        F.round(
            100.0 *
            F.col("security_delay_minutes") /
            F.col("total_delay_minutes"),
            2
        )
    )

    .withColumn(
        "late_aircraft_share_pct",
        F.round(
            100.0 *
            F.col("late_aircraft_delay_minutes") /
            F.col("total_delay_minutes"),
            2
        )
    )

    # -----------------------------------------------------
    # Identify dominant delay cause
    # -----------------------------------------------------

    .withColumn(
        "dominant_delay_minutes",
        F.greatest(
            "carrier_delay_minutes",
            "weather_delay_minutes",
            "nas_delay_minutes",
            "security_delay_minutes",
            "late_aircraft_delay_minutes"
        )
    )

    .withColumn(
        "dominant_delay_cause",

        F.when(
            F.col("dominant_delay_minutes") ==
            F.col("late_aircraft_delay_minutes"),
            "Late Aircraft"
        )

        .when(
            F.col("dominant_delay_minutes") ==
            F.col("carrier_delay_minutes"),
            "Carrier"
        )

        .when(
            F.col("dominant_delay_minutes") ==
            F.col("nas_delay_minutes"),
            "NAS"
        )

        .when(
            F.col("dominant_delay_minutes") ==
            F.col("weather_delay_minutes"),
            "Weather"
        )

        .otherwise("Security")
    )
)


print("=== 2025 MAJOR-CARRIER DELAY-CAUSE PROFILES ===")

print(
    "Carriers:",
    annual_carrier_delay_profile.count()
)


annual_carrier_delay_profile.select(
    "operating_carrier",
    "operating_carrier_name",
    "eligible_late_flights",
    "total_delay_minutes",

    "late_aircraft_share_pct",
    "carrier_share_pct",
    "nas_share_pct",
    "weather_share_pct",
    "security_share_pct",

    "dominant_delay_cause"
).orderBy(
    F.desc("total_delay_minutes")
).show(
    15,
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 26
# Persist annual major-carrier delay-cause profiles
# ---------------------------------------------------------

CARRIER_DELAY_PROFILE_TABLE = (
    "workspace.airline_gold.annual_carrier_delay_profile_2025"
)


carrier_delay_profile_gold = (
    annual_carrier_delay_profile

    .withColumn(
        "source_year",
        F.lit(2025)
    )

    .withColumn(
        "minimum_operations_threshold",
        F.lit(100000)
    )

    .select(
        "source_year",

        "operating_carrier",
        "operating_carrier_name",

        "eligible_late_flights",
        "total_delay_minutes",

        "carrier_delay_minutes",
        "weather_delay_minutes",
        "nas_delay_minutes",
        "security_delay_minutes",
        "late_aircraft_delay_minutes",

        "carrier_share_pct",
        "weather_share_pct",
        "nas_share_pct",
        "security_share_pct",
        "late_aircraft_share_pct",

        "dominant_delay_cause",

        "minimum_operations_threshold"
    )
)


# ---------------------------------------------------------
# Persist
# ---------------------------------------------------------

(
    carrier_delay_profile_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CARRIER_DELAY_PROFILE_TABLE)
)


# ---------------------------------------------------------
# Read-back validation
# ---------------------------------------------------------

carrier_delay_profile_table = spark.table(
    CARRIER_DELAY_PROFILE_TABLE
)


print("=== PERSISTED CARRIER DELAY PROFILE ===")

print(
    "Rows:",
    carrier_delay_profile_table.count()
)

print(
    "Distinct carriers:",
    carrier_delay_profile_table
        .select("operating_carrier")
        .distinct()
        .count()
)

print("\nDominant cause distribution:")

carrier_delay_profile_table.groupBy(
    "dominant_delay_cause"
).count().orderBy(
    F.desc("count")
).show()


print("Table:")
print(CARRIER_DELAY_PROFILE_TABLE)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 27
# Validate final BI Gold-layer inventory
# ---------------------------------------------------------

bi_gold_tables = [

    (
        "Monthly Network Performance",
        "workspace.airline_gold.monthly_network_performance"
    ),

    (
        "Annual Carrier Benchmark",
        "workspace.airline_gold.annual_carrier_benchmark_2025"
    ),

    (
        "Annual Airport Benchmark",
        "workspace.airline_gold.annual_airport_benchmark_2025"
    ),

    (
        "Annual Route Benchmark",
        "workspace.airline_gold.annual_route_benchmark_2025"
    ),

    (
        "Annual Delay-Cause Decomposition",
        "workspace.airline_gold.annual_delay_cause_decomposition_2025"
    ),

    (
        "Annual Carrier Delay Profile",
        "workspace.airline_gold.annual_carrier_delay_profile_2025"
    ),

    (
        "Airport Monthly Operational Segments",
        "workspace.airline_gold.airport_month_operational_segments"
    ),

    (
        "Airport Operational Stability",
        "workspace.airline_gold.airport_operational_stability_2025"
    ),

    (
        "Predictive Model Comparison",
        "workspace.airline_gold.predictive_model_comparison"
    )
]


print("=== PHASE 10 BI GOLD INVENTORY ===\n")


inventory_results = []


for logical_name, table_name in bi_gold_tables:

    df = spark.table(table_name)

    rows = df.count()
    cols = len(df.columns)

    inventory_results.append(
        (
            logical_name,
            table_name,
            rows,
            cols
        )
    )


inventory_df = spark.createDataFrame(
    inventory_results,
    [
        "dataset",
        "table_name",
        "rows",
        "columns"
    ]
)


inventory_df.show(
    truncate=False
)


print("\nTotal dashboard-ready datasets:",
      len(inventory_results))

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 28
# Persist one-row 2025 executive network summary
# ---------------------------------------------------------

EXECUTIVE_SUMMARY_TABLE = (
    "workspace.airline_gold.executive_network_summary_2025"
)


executive_network_summary = (
    flights
    .agg(

        # -------------------------------------------------
        # Network volume
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
        # Diversions
        # -------------------------------------------------

        F.sum(
            F.when(
                F.col("has_diversion_event_data") == True,
                1
            ).otherwise(0)
        ).alias("diversion_event_operations"),

        # -------------------------------------------------
        # Network dimensions
        # -------------------------------------------------

        F.countDistinct("operating_carrier")
            .alias("operating_carriers"),

        F.countDistinct("origin")
            .alias("airports"),

        F.countDistinct(
            "origin",
            "destination"
        ).alias("directional_routes")
    )

    # -----------------------------------------------------
    # Executive rates
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

    .withColumn(
        "source_year",
        F.lit(2025)
    )

    .select(
        "source_year",
        "total_operations",
        "cancelled_operations",
        "cancellation_rate_pct",
        "completed_arrivals",
        "late_arrivals_15_plus",
        "late_arrival_rate_pct",
        "avg_departure_delay_minutes",
        "avg_arrival_delay_minutes",
        "diversion_event_operations",
        "diversion_event_rate_pct",
        "operating_carriers",
        "airports",
        "directional_routes"
    )
)


# ---------------------------------------------------------
# Persist
# ---------------------------------------------------------

(
    executive_network_summary.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(EXECUTIVE_SUMMARY_TABLE)
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

executive_summary_table = spark.table(
    EXECUTIVE_SUMMARY_TABLE
)


print("=== EXECUTIVE NETWORK SUMMARY ===")

executive_summary_table.show(
    truncate=False
)

print("\nRows:", executive_summary_table.count())

print("\nTable:")
print(EXECUTIVE_SUMMARY_TABLE)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 10 — Step 29
# Final validation of dashboard-ready Gold layer
# ---------------------------------------------------------

final_bi_tables = [

    (
        "Executive Network Summary",
        "workspace.airline_gold.executive_network_summary_2025"
    ),

    (
        "Monthly Network Performance",
        "workspace.airline_gold.monthly_network_performance"
    ),

    (
        "Annual Carrier Benchmark",
        "workspace.airline_gold.annual_carrier_benchmark_2025"
    ),

    (
        "Annual Airport Benchmark",
        "workspace.airline_gold.annual_airport_benchmark_2025"
    ),

    (
        "Annual Route Benchmark",
        "workspace.airline_gold.annual_route_benchmark_2025"
    ),

    (
        "Annual Delay-Cause Decomposition",
        "workspace.airline_gold.annual_delay_cause_decomposition_2025"
    ),

    (
        "Annual Carrier Delay Profile",
        "workspace.airline_gold.annual_carrier_delay_profile_2025"
    ),

    (
        "Airport Monthly Operational Segments",
        "workspace.airline_gold.airport_month_operational_segments"
    ),

    (
        "Airport Operational Stability",
        "workspace.airline_gold.airport_operational_stability_2025"
    ),

    (
        "Predictive Model Comparison",
        "workspace.airline_gold.predictive_model_comparison"
    )
]


final_inventory = []


for dataset_name, table_name in final_bi_tables:

    df = spark.table(table_name)

    final_inventory.append(
        (
            dataset_name,
            df.count(),
            len(df.columns)
        )
    )


final_inventory_df = spark.createDataFrame(
    final_inventory,
    [
        "dataset",
        "rows",
        "columns"
    ]
)


print("=== FINAL DASHBOARD-READY GOLD LAYER ===")

final_inventory_df.show(
    truncate=False
)

print(
    "\nTotal dashboard-ready datasets:",
    len(final_inventory)
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 11 — Step 2
# Executive Overview dashboard source validation
# ---------------------------------------------------------

executive_summary = spark.table(
    "workspace.airline_gold.executive_network_summary_2025"
)

monthly_network = spark.table(
    "workspace.airline_gold.monthly_network_performance"
)

delay_cause_summary = spark.table(
    "workspace.airline_gold.annual_delay_cause_decomposition_2025"
)


print("=== EXECUTIVE OVERVIEW DATA SOURCES ===")

print(
    "Executive KPI rows:",
    executive_summary.count()
)

print(
    "Monthly trend rows:",
    monthly_network.count()
)

print(
    "Delay-cause rows:",
    delay_cause_summary.count()
)


print("\nMonthly periods:")

monthly_network.select(
    "source_month"
).orderBy(
    "source_month"
).show(
    12,
    truncate=False
)


print("Delay causes:")

delay_cause_summary.select(
    "delay_cause",
    "delay_share_pct",
    "avg_cause_minutes_per_affected_flight"
).orderBy(
    F.desc("delay_share_pct")
).show(
    truncate=False
)