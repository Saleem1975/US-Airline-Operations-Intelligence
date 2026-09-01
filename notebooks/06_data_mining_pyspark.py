# Databricks notebook source
from pyspark.sql import functions as F


# ---------------------------------------------------------
# Phase 9 — Data Mining / Predictive Analytics
# Load the enriched Silver source
# ---------------------------------------------------------

ENRICHED_TABLE = "workspace.airline_silver.flights_enriched"

flights = spark.table(ENRICHED_TABLE)


# ---------------------------------------------------------
# Modeling population
#
# Goal:
# Predict whether a completed flight arrives 15+ minutes late.
#
# We exclude:
# - superseded duplicate records
# - operational cancellations
# - flights without an actual gate arrival
#
# Important:
# We will later use only PRE-OPERATIONAL predictors.
# ---------------------------------------------------------

model_population = (
    flights
    .filter(
        (F.col("canonical_operation_flag") == True) &
        (F.col("operational_cancellation_flag") == False) &
        F.col("actual_gate_arrival").isNotNull()
    )
    .withColumn(
        "target_arrival_delay_15_plus",
        F.when(
            F.col("arrival_delay_minutes") >= 15,
            1
        ).otherwise(0)
    )
)


print("=== MODELING POPULATION ===")

print(
    "Rows:",
    model_population.count()
)

print(
    "Positive class — arrival delay 15+:",
    model_population
        .filter(
            F.col("target_arrival_delay_15_plus") == 1
        )
        .count()
)

print(
    "Negative class — arrival delay < 15:",
    model_population
        .filter(
            F.col("target_arrival_delay_15_plus") == 0
        )
        .count()
)


print("\n=== TARGET DISTRIBUTION ===")

model_population.groupBy(
    "target_arrival_delay_15_plus"
).count().withColumn(
    "percentage",
    F.round(
        100.0 *
        F.col("count") /
        model_population.count(),
        2
    )
).orderBy(
    "target_arrival_delay_15_plus"
).show()

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 2
# Build leakage-safe pre-flight feature base
# ---------------------------------------------------------

model_features_base = (
    model_population

    # Scheduled departure hour
    .withColumn(
        "scheduled_departure_hour",
        F.substring(
            F.col("scheduled_departure_crs"),
            1,
            2
        ).cast("int")
    )

    # Scheduled arrival hour
    .withColumn(
        "scheduled_arrival_hour",
        F.substring(
            F.col("scheduled_arrival_crs"),
            1,
            2
        ).cast("int")
    )

    # Simple directional route identifier
    .withColumn(
        "route",
        F.concat_ws(
            "_",
            F.col("origin"),
            F.col("destination")
        )
    )

    .select(
        "flight_date",
        "source_month",
        "day_of_week",

        "marketing_carrier",
        "origin",
        "destination",
        "route",

        "scheduled_departure_hour",
        "scheduled_arrival_hour",
        "scheduled_elapsed_minutes",

        "target_arrival_delay_15_plus"
    )
)


print("=== PRE-FLIGHT FEATURE BASE ===")

print(
    "Rows:",
    model_features_base.count()
)

print(
    "Columns:",
    len(model_features_base.columns)
)


# ---------------------------------------------------------
# Missing-value audit
# ---------------------------------------------------------

print("\n=== FEATURE MISSING-VALUE CHECK ===")

model_features_base.agg(

    F.sum(
        F.when(
            F.col("marketing_carrier").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_marketing_carrier"),

    F.sum(
        F.when(
            F.col("origin").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_origin"),

    F.sum(
        F.when(
            F.col("destination").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_destination"),

    F.sum(
        F.when(
            F.col("scheduled_departure_hour").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_departure_hour"),

    F.sum(
        F.when(
            F.col("scheduled_arrival_hour").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_arrival_hour"),

    F.sum(
        F.when(
            F.col("scheduled_elapsed_minutes").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_scheduled_elapsed")

).show(truncate=False)


# ---------------------------------------------------------
# Categorical cardinality
# ---------------------------------------------------------

print("\n=== CATEGORICAL CARDINALITY ===")

model_features_base.agg(

    F.countDistinct("marketing_carrier")
        .alias("marketing_carriers"),

    F.countDistinct("origin")
        .alias("origins"),

    F.countDistinct("destination")
        .alias("destinations"),

    F.countDistinct("route")
        .alias("directional_routes")

).show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 3
# Time-based train / validation / test split
# ---------------------------------------------------------

model_df = (
    model_features_base
    .select(
        "flight_date",
        "source_month",
        "day_of_week",
        "marketing_carrier",
        "origin",
        "destination",
        "scheduled_departure_hour",
        "scheduled_arrival_hour",
        "scheduled_elapsed_minutes",
        "target_arrival_delay_15_plus"
    )
    .withColumn(
        "dataset_split",
        F.when(
            F.col("flight_date") <= F.to_date(F.lit("2025-09-30")),
            "train"
        )
        .when(
            F.col("flight_date") <= F.to_date(F.lit("2025-11-30")),
            "validation"
        )
        .otherwise("test")
    )
)


# ---------------------------------------------------------
# Split profile and target balance
# ---------------------------------------------------------

split_profile = (
    model_df
    .groupBy("dataset_split")
    .agg(
        F.count("*").alias("records"),

        F.sum("target_arrival_delay_15_plus")
            .alias("late_flights")
    )
    .withColumn(
        "non_late_flights",
        F.col("records") - F.col("late_flights")
    )
    .withColumn(
        "late_rate_pct",
        F.round(
            100.0 *
            F.col("late_flights") /
            F.col("records"),
            2
        )
    )
)


print("=== TIME-BASED MODEL SPLIT ===")

split_profile.orderBy(
    F.when(F.col("dataset_split") == "train", 1)
     .when(F.col("dataset_split") == "validation", 2)
     .otherwise(3)
).show(truncate=False)


print("=== SPLIT RECONCILIATION ===")

split_total = (
    split_profile
    .agg(F.sum("records"))
    .first()[0]
)

print("Original modeling population:", f"{model_df.count():,}")
print("Sum of splits:", f"{split_total:,}")
print("Exact match:", model_df.count() == split_total)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 4
# Spark ML preprocessing pipeline
# ---------------------------------------------------------

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

import math


# ---------------------------------------------------------
# Create leakage-safe engineered features
# ---------------------------------------------------------

ml_df = (
    model_df

    # Spark ML classifiers expect the target column
    # to normally be named "label"
    .withColumn(
        "label",
        F.col("target_arrival_delay_15_plus").cast("double")
    )

    # Cyclic scheduled departure hour
    .withColumn(
        "departure_hour_sin",
        F.sin(
            2 * math.pi *
            F.col("scheduled_departure_hour") / 24.0
        )
    )
    .withColumn(
        "departure_hour_cos",
        F.cos(
            2 * math.pi *
            F.col("scheduled_departure_hour") / 24.0
        )
    )

    # Cyclic scheduled arrival hour
    .withColumn(
        "arrival_hour_sin",
        F.sin(
            2 * math.pi *
            F.col("scheduled_arrival_hour") / 24.0
        )
    )
    .withColumn(
        "arrival_hour_cos",
        F.cos(
            2 * math.pi *
            F.col("scheduled_arrival_hour") / 24.0
        )
    )

    # Cyclic month representation
    .withColumn(
        "month_sin",
        F.sin(
            2 * math.pi *
            (F.col("source_month") - 1) / 12.0
        )
    )
    .withColumn(
        "month_cos",
        F.cos(
            2 * math.pi *
            (F.col("source_month") - 1) / 12.0
        )
    )
)


# ---------------------------------------------------------
# Time-based datasets
# ---------------------------------------------------------

train_df = ml_df.filter(
    F.col("dataset_split") == "train"
)

validation_df = ml_df.filter(
    F.col("dataset_split") == "validation"
)

test_df = ml_df.filter(
    F.col("dataset_split") == "test"
)


# ---------------------------------------------------------
# Categorical predictors
# ---------------------------------------------------------

categorical_columns = [
    "marketing_carrier",
    "origin",
    "destination",
    "day_of_week"
]


indexers = [
    StringIndexer(
        inputCol=column,
        outputCol=f"{column}_index",
        handleInvalid="keep",
        stringOrderType="alphabetAsc"
    )
    for column in categorical_columns
]


indexed_columns = [
    f"{column}_index"
    for column in categorical_columns
]


encoded_columns = [
    f"{column}_ohe"
    for column in categorical_columns
]


encoder = OneHotEncoder(
    inputCols=indexed_columns,
    outputCols=encoded_columns,
    handleInvalid="keep",
    dropLast=True
)


# ---------------------------------------------------------
# Numeric / cyclic predictors
# ---------------------------------------------------------

numeric_features = [
    "scheduled_elapsed_minutes",
    "departure_hour_sin",
    "departure_hour_cos",
    "arrival_hour_sin",
    "arrival_hour_cos",
    "month_sin",
    "month_cos"
]


assembler = VectorAssembler(
    inputCols=encoded_columns + numeric_features,
    outputCol="features",
    handleInvalid="keep"
)


# ---------------------------------------------------------
# Fit preprocessing ONLY on training data
# ---------------------------------------------------------

preprocessing_pipeline = Pipeline(
    stages=indexers + [encoder, assembler]
)

preprocessing_model = preprocessing_pipeline.fit(
    train_df
)


# ---------------------------------------------------------
# Transform train, validation, and test
# ---------------------------------------------------------

train_prepared = (
    preprocessing_model
    .transform(train_df)
    .select(
        "flight_date",
        "dataset_split",
        "label",
        "features"
    )
)

validation_prepared = (
    preprocessing_model
    .transform(validation_df)
    .select(
        "flight_date",
        "dataset_split",
        "label",
        "features"
    )
)

test_prepared = (
    preprocessing_model
    .transform(test_df)
    .select(
        "flight_date",
        "dataset_split",
        "label",
        "features"
    )
)


print("=== ML PREPROCESSING VALIDATION ===")

print("Training rows:", train_prepared.count())
print("Validation rows:", validation_prepared.count())
print("Test rows:", test_prepared.count())

feature_vector_size = (
    train_prepared
    .select("features")
    .first()["features"]
    .size
)

print("Feature vector size:", feature_vector_size)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 5
# Baseline Logistic Regression
#
# Train:      Jan-Sep
# Validation: Oct-Nov
# Test:       December remains untouched
# ---------------------------------------------------------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# ---------------------------------------------------------
# Baseline model
# No class weighting yet.
# No threshold tuning yet.
# ---------------------------------------------------------

baseline_lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    maxIter=30,
    regParam=0.01,
    elasticNetParam=0.0,
    standardization=True
)


print("Training baseline logistic regression...")

baseline_lr_model = baseline_lr.fit(
    train_prepared
)

print("Model training complete.")


# ---------------------------------------------------------
# Score VALIDATION data only
# ---------------------------------------------------------

validation_predictions = (
    baseline_lr_model
    .transform(validation_prepared)
    .cache()
)

# Materialize once because several metrics will use it
validation_prediction_count = validation_predictions.count()

print(
    "Validation predictions:",
    f"{validation_prediction_count:,}"
)


# ---------------------------------------------------------
# ROC-AUC and PR-AUC
# ---------------------------------------------------------

roc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

pr_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)


roc_auc = roc_evaluator.evaluate(
    validation_predictions
)

pr_auc = pr_evaluator.evaluate(
    validation_predictions
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = validation_predictions.agg(

    F.sum(
        F.when(
            (F.col("label") == 1) &
            (F.col("prediction") == 1),
            1
        ).otherwise(0)
    ).alias("tp"),

    F.sum(
        F.when(
            (F.col("label") == 0) &
            (F.col("prediction") == 0),
            1
        ).otherwise(0)
    ).alias("tn"),

    F.sum(
        F.when(
            (F.col("label") == 0) &
            (F.col("prediction") == 1),
            1
        ).otherwise(0)
    ).alias("fp"),

    F.sum(
        F.when(
            (F.col("label") == 1) &
            (F.col("prediction") == 0),
            1
        ).otherwise(0)
    ).alias("fn")

).first()


tp = cm["tp"]
tn = cm["tn"]
fp = cm["fp"]
fn = cm["fn"]

total = tp + tn + fp + fn

accuracy = (tp + tn) / total

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ---------------------------------------------------------
# Naive baseline:
# Predict every validation flight as NOT late
# ---------------------------------------------------------

validation_positive = tp + fn
validation_negative = tn + fp

naive_accuracy = (
    validation_negative /
    (validation_positive + validation_negative)
)


print("\n=== VALIDATION — LOGISTIC REGRESSION BASELINE ===")

print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print()
print("Confusion matrix:")
print(f"TP: {tp:,}")
print(f"FP: {fp:,}")
print(f"TN: {tn:,}")
print(f"FN: {fn:,}")

print()
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 score:  {f1:.4f}")

print()
print(
    "Naive all-not-late accuracy:",
    f"{naive_accuracy * 100:.2f}%"
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 5A
# Serverless-safe validation scoring
#
# The logistic regression model already trained.
# We simply transform without .cache()
# ---------------------------------------------------------

from pyspark.ml.evaluation import BinaryClassificationEvaluator


validation_predictions = (
    baseline_lr_model
    .transform(validation_prepared)
)


print("Scoring validation data...")

validation_prediction_count = validation_predictions.count()

print(
    "Validation predictions:",
    f"{validation_prediction_count:,}"
)


# ---------------------------------------------------------
# ROC-AUC and PR-AUC
# ---------------------------------------------------------

roc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

pr_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)


roc_auc = roc_evaluator.evaluate(
    validation_predictions
)

pr_auc = pr_evaluator.evaluate(
    validation_predictions
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = validation_predictions.agg(

    F.sum(
        F.when(
            (F.col("label") == 1) &
            (F.col("prediction") == 1),
            1
        ).otherwise(0)
    ).alias("tp"),

    F.sum(
        F.when(
            (F.col("label") == 0) &
            (F.col("prediction") == 0),
            1
        ).otherwise(0)
    ).alias("tn"),

    F.sum(
        F.when(
            (F.col("label") == 0) &
            (F.col("prediction") == 1),
            1
        ).otherwise(0)
    ).alias("fp"),

    F.sum(
        F.when(
            (F.col("label") == 1) &
            (F.col("prediction") == 0),
            1
        ).otherwise(0)
    ).alias("fn")

).first()


tp = cm["tp"]
tn = cm["tn"]
fp = cm["fp"]
fn = cm["fn"]

total = tp + tn + fp + fn


# ---------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------

accuracy = (tp + tn) / total

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ---------------------------------------------------------
# Naive baseline
# Predict every flight as NOT late
# ---------------------------------------------------------

validation_positive = tp + fn
validation_negative = tn + fp

naive_accuracy = (
    validation_negative /
    (validation_positive + validation_negative)
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n=== VALIDATION — LOGISTIC REGRESSION BASELINE ===")

print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print()
print("Confusion matrix:")
print(f"TP: {tp:,}")
print(f"FP: {fp:,}")
print(f"TN: {tn:,}")
print(f"FN: {fn:,}")

print()
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 score:  {f1:.4f}")

print()
print(
    "Naive all-not-late accuracy:",
    f"{naive_accuracy * 100:.2f}%"
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 5B
# Inspect validation probability distribution
# ---------------------------------------------------------

from pyspark.ml.functions import vector_to_array


validation_probability_df = (
    validation_predictions
    .withColumn(
        "late_probability",
        vector_to_array("probability")[1]
    )
)


# ---------------------------------------------------------
# Overall probability distribution
# ---------------------------------------------------------

probability_quantiles = (
    validation_probability_df
    .approxQuantile(
        "late_probability",
        [
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ],
        0.001
    )
)


print("=== VALIDATION PREDICTED-PROBABILITY QUANTILES ===")

quantile_labels = [
    "1%",
    "5%",
    "10%",
    "25%",
    "50%",
    "75%",
    "90%",
    "95%",
    "99%"
]

for label, value in zip(
    quantile_labels,
    probability_quantiles
):
    print(
        f"{label:>3} percentile: "
        f"{value:.4f}"
    )


# ---------------------------------------------------------
# Compare probabilities for actual late vs non-late flights
# ---------------------------------------------------------

print("\n=== PROBABILITY BY ACTUAL CLASS ===")

validation_probability_df.groupBy(
    "label"
).agg(

    F.count("*").alias("records"),

    F.round(
        F.avg("late_probability"),
        4
    ).alias("avg_probability"),

    F.round(
        F.expr(
            "percentile_approx(late_probability, 0.5)"
        ),
        4
    ).alias("median_probability"),

    F.round(
        F.max("late_probability"),
        4
    ).alias("max_probability")

).orderBy("label").show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 5C
# Threshold tuning on VALIDATION data only
# ---------------------------------------------------------

thresholds = [
    round(x / 100, 2)
    for x in range(10, 36)
]


# ---------------------------------------------------------
# Calculate confusion-matrix counts for all thresholds
# in a single Spark aggregation
# ---------------------------------------------------------

aggregate_expressions = []

for threshold in thresholds:

    key = str(threshold).replace(".", "_")

    aggregate_expressions.extend([

        F.sum(
            F.when(
                (F.col("label") == 1) &
                (F.col("late_probability") >= threshold),
                1
            ).otherwise(0)
        ).alias(f"tp_{key}"),

        F.sum(
            F.when(
                (F.col("label") == 0) &
                (F.col("late_probability") >= threshold),
                1
            ).otherwise(0)
        ).alias(f"fp_{key}"),

        F.sum(
            F.when(
                (F.col("label") == 0) &
                (F.col("late_probability") < threshold),
                1
            ).otherwise(0)
        ).alias(f"tn_{key}"),

        F.sum(
            F.when(
                (F.col("label") == 1) &
                (F.col("late_probability") < threshold),
                1
            ).otherwise(0)
        ).alias(f"fn_{key}")
    ])


threshold_counts = (
    validation_probability_df
    .agg(*aggregate_expressions)
    .first()
)


# ---------------------------------------------------------
# Calculate metrics in Python
# ---------------------------------------------------------

threshold_results = []

for threshold in thresholds:

    key = str(threshold).replace(".", "_")

    tp = threshold_counts[f"tp_{key}"]
    fp = threshold_counts[f"fp_{key}"]
    tn = threshold_counts[f"tn_{key}"]
    fn = threshold_counts[f"fn_{key}"]

    total = tp + fp + tn + fn

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0 else 0
    )

    accuracy = (
        (tp + tn) / total
    )

    predicted_late_rate = (
        (tp + fp) / total
    )

    threshold_results.append({
        "threshold": threshold,
        "precision_pct": round(precision * 100, 2),
        "recall_pct": round(recall * 100, 2),
        "f1": round(f1, 4),
        "accuracy_pct": round(accuracy * 100, 2),
        "predicted_late_rate_pct":
            round(predicted_late_rate * 100, 2),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    })


threshold_results_df = spark.createDataFrame(
    threshold_results
)


print("=== TOP VALIDATION THRESHOLDS BY F1 ===")

threshold_results_df.orderBy(
    F.desc("f1")
).show(
    15,
    truncate=False
)


# ---------------------------------------------------------
# Best threshold
# ---------------------------------------------------------

best_threshold_result = max(
    threshold_results,
    key=lambda x: x["f1"]
)

best_threshold = best_threshold_result["threshold"]


print("\n=== BEST VALIDATION THRESHOLD ===")

for key, value in best_threshold_result.items():
    print(f"{key}: {value}")

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 6
# FINAL evaluation on untouched December test data
#
# Threshold was selected using validation data only:
# LOCKED THRESHOLD = 0.17
# ---------------------------------------------------------

from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator


LOCKED_THRESHOLD = 0.17


# ---------------------------------------------------------
# Score December test data
# ---------------------------------------------------------

test_predictions = (
    baseline_lr_model
    .transform(test_prepared)
    .withColumn(
        "late_probability",
        vector_to_array("probability")[1]
    )
    .withColumn(
        "threshold_prediction",
        F.when(
            F.col("late_probability") >= LOCKED_THRESHOLD,
            1.0
        ).otherwise(0.0)
    )
)


print("Scoring untouched December test data...")

test_prediction_count = test_predictions.count()

print(
    "Test predictions:",
    f"{test_prediction_count:,}"
)


# ---------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------

test_roc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

test_pr_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)


test_roc_auc = test_roc_evaluator.evaluate(
    test_predictions
)

test_pr_auc = test_pr_evaluator.evaluate(
    test_predictions
)


# ---------------------------------------------------------
# Confusion matrix at LOCKED threshold
# ---------------------------------------------------------

test_cm = test_predictions.agg(

    F.sum(
        F.when(
            (F.col("label") == 1) &
            (F.col("threshold_prediction") == 1),
            1
        ).otherwise(0)
    ).alias("tp"),

    F.sum(
        F.when(
            (F.col("label") == 0) &
            (F.col("threshold_prediction") == 0),
            1
        ).otherwise(0)
    ).alias("tn"),

    F.sum(
        F.when(
            (F.col("label") == 0) &
            (F.col("threshold_prediction") == 1),
            1
        ).otherwise(0)
    ).alias("fp"),

    F.sum(
        F.when(
            (F.col("label") == 1) &
            (F.col("threshold_prediction") == 0),
            1
        ).otherwise(0)
    ).alias("fn")

).first()


tp = test_cm["tp"]
tn = test_cm["tn"]
fp = test_cm["fp"]
fn = test_cm["fn"]

total = tp + tn + fp + fn


# ---------------------------------------------------------
# Test metrics
# ---------------------------------------------------------

test_accuracy = (
    (tp + tn) / total
)

test_precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

test_recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

test_f1 = (
    2 * test_precision * test_recall /
    (test_precision + test_recall)
    if (test_precision + test_recall) > 0
    else 0
)


# Naive December baseline
test_positive = tp + fn
test_negative = tn + fp

test_naive_accuracy = (
    test_negative /
    (test_positive + test_negative)
)


# ---------------------------------------------------------
# Final results
# ---------------------------------------------------------

print("\n=== FINAL DECEMBER TEST RESULTS ===")

print(
    "Locked threshold:",
    LOCKED_THRESHOLD
)

print(f"ROC-AUC:   {test_roc_auc:.4f}")
print(f"PR-AUC:    {test_pr_auc:.4f}")

print()
print("Confusion matrix:")
print(f"TP: {tp:,}")
print(f"FP: {fp:,}")
print(f"TN: {tn:,}")
print(f"FN: {fn:,}")

print()
print(
    f"Accuracy:  {test_accuracy * 100:.2f}%"
)
print(
    f"Precision: {test_precision * 100:.2f}%"
)
print(
    f"Recall:    {test_recall * 100:.2f}%"
)
print(
    f"F1 score:  {test_f1:.4f}"
)

print()
print(
    "Naive all-not-late accuracy:",
    f"{test_naive_accuracy * 100:.2f}%"
)

print(
    "Actual December late-flight rate:",
    f"{test_positive / total * 100:.2f}%"
)

print(
    "Predicted December late-flight rate:",
    f"{(tp + fp) / total * 100:.2f}%"
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 7
# Gradient-Boosted Tree challenger
#
# Train: Jan-Sep
# Tune/evaluate: Oct-Nov validation
# December: do not evaluate yet
# ---------------------------------------------------------

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    rawPredictionCol="rawPrediction",
    probabilityCol="probability",

    maxIter=40,
    maxDepth=5,
    stepSize=0.05,
    subsamplingRate=0.8,

    seed=42
)


print("Training Gradient-Boosted Tree model...")

gbt_model = gbt.fit(
    train_prepared
)

print("GBT training complete.")


# ---------------------------------------------------------
# Validation scoring only
# ---------------------------------------------------------

gbt_validation_predictions = (
    gbt_model
    .transform(validation_prepared)
)


print("Scoring validation data...")

print(
    "Validation predictions:",
    f"{gbt_validation_predictions.count():,}"
)


# ---------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------

gbt_roc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

gbt_pr_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)


gbt_validation_roc_auc = (
    gbt_roc_evaluator.evaluate(
        gbt_validation_predictions
    )
)

gbt_validation_pr_auc = (
    gbt_pr_evaluator.evaluate(
        gbt_validation_predictions
    )
)


print("\n=== VALIDATION MODEL COMPARISON ===")

print("Logistic Regression")
print("ROC-AUC: 0.6443")
print("PR-AUC:  0.3038")

print()

print("Gradient-Boosted Trees")
print(
    f"ROC-AUC: {gbt_validation_roc_auc:.4f}"
)
print(
    f"PR-AUC:  {gbt_validation_pr_auc:.4f}"
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 7A
# Gradient-Boosted Tree challenger — corrected for
# Databricks Serverless / current PySpark GBT API
# ---------------------------------------------------------

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",

    maxIter=40,
    maxDepth=5,
    stepSize=0.05,
    subsamplingRate=0.8,

    seed=42
)


print("Training Gradient-Boosted Tree model...")

gbt_model = gbt.fit(
    train_prepared
)

print("GBT training complete.")


# ---------------------------------------------------------
# Score validation data only
# ---------------------------------------------------------

gbt_validation_predictions = (
    gbt_model
    .transform(validation_prepared)
)


print("Scoring validation data...")

gbt_validation_count = (
    gbt_validation_predictions.count()
)

print(
    "Validation predictions:",
    f"{gbt_validation_count:,}"
)


# ---------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------

gbt_roc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

gbt_pr_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)


gbt_validation_roc_auc = (
    gbt_roc_evaluator.evaluate(
        gbt_validation_predictions
    )
)

gbt_validation_pr_auc = (
    gbt_pr_evaluator.evaluate(
        gbt_validation_predictions
    )
)


print("\n=== VALIDATION MODEL COMPARISON ===")

print("Logistic Regression")
print("ROC-AUC: 0.6443")
print("PR-AUC:  0.3038")

print()

print("Gradient-Boosted Trees")
print(
    f"ROC-AUC: {gbt_validation_roc_auc:.4f}"
)
print(
    f"PR-AUC:  {gbt_validation_pr_auc:.4f}"
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 7B
# Tree-specific preprocessing
#
# Trees do NOT need one-hot encoded categorical variables.
# This dramatically reduces feature dimensionality.
# ---------------------------------------------------------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler


tree_categorical_columns = [
    "marketing_carrier",
    "origin",
    "destination",
    "day_of_week"
]


tree_indexers = [
    StringIndexer(
        inputCol=column,
        outputCol=f"{column}_tree_index",
        handleInvalid="keep",
        stringOrderType="alphabetAsc"
    )
    for column in tree_categorical_columns
]


tree_categorical_features = [
    f"{column}_tree_index"
    for column in tree_categorical_columns
]


tree_numeric_features = [
    "scheduled_elapsed_minutes",
    "departure_hour_sin",
    "departure_hour_cos",
    "arrival_hour_sin",
    "arrival_hour_cos",
    "month_sin",
    "month_cos"
]


tree_assembler = VectorAssembler(
    inputCols=(
        tree_categorical_features +
        tree_numeric_features
    ),
    outputCol="gbt_features",
    handleInvalid="keep"
)


tree_preprocessing_pipeline = Pipeline(
    stages=tree_indexers + [tree_assembler]
)


# Fit ONLY on training data
tree_preprocessing_model = (
    tree_preprocessing_pipeline.fit(train_df)
)


gbt_train_prepared = (
    tree_preprocessing_model
    .transform(train_df)
    .select(
        "flight_date",
        "label",
        "gbt_features"
    )
)

gbt_validation_prepared = (
    tree_preprocessing_model
    .transform(validation_df)
    .select(
        "flight_date",
        "label",
        "gbt_features"
    )
)


print("=== TREE-SPECIFIC PREPROCESSING ===")

print(
    "Training rows:",
    gbt_train_prepared.count()
)

print(
    "Validation rows:",
    gbt_validation_prepared.count()
)

tree_feature_size = (
    gbt_train_prepared
    .select("gbt_features")
    .first()["gbt_features"]
    .size
)

print(
    "GBT feature vector size:",
    tree_feature_size
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 7C
# Lightweight Gradient-Boosted Tree challenger
# ---------------------------------------------------------

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


gbt_light = GBTClassifier(
    featuresCol="gbt_features",
    labelCol="label",

    # Much lighter than our first attempt
    maxIter=15,
    maxDepth=4,
    stepSize=0.1,
    subsamplingRate=0.8,

    # Needed because origin/destination have 364 categories
    maxBins=400,

    seed=42
)


print("Training lightweight Gradient-Boosted Tree model...")

gbt_light_model = gbt_light.fit(
    gbt_train_prepared
)

print("GBT training complete.")


# ---------------------------------------------------------
# Score VALIDATION data only
# ---------------------------------------------------------

gbt_validation_predictions = (
    gbt_light_model
    .transform(gbt_validation_prepared)
)


print("Scoring validation data...")

print(
    "Validation predictions:",
    f"{gbt_validation_predictions.count():,}"
)


# ---------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------

gbt_roc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

gbt_pr_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)


gbt_validation_roc_auc = (
    gbt_roc_evaluator.evaluate(
        gbt_validation_predictions
    )
)

gbt_validation_pr_auc = (
    gbt_pr_evaluator.evaluate(
        gbt_validation_predictions
    )
)


print("\n=== VALIDATION MODEL COMPARISON ===")

print("Logistic Regression")
print("ROC-AUC: 0.6443")
print("PR-AUC:  0.3038")

print()

print("Lightweight Gradient-Boosted Trees")
print(
    f"ROC-AUC: {gbt_validation_roc_auc:.4f}"
)
print(
    f"PR-AUC:  {gbt_validation_pr_auc:.4f}"
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 8
# Tune GBT classification threshold on VALIDATION only
# ---------------------------------------------------------

from pyspark.ml.functions import vector_to_array


# Extract probability of late flight
gbt_validation_probability_df = (
    gbt_validation_predictions
    .withColumn(
        "late_probability",
        vector_to_array("probability")[1]
    )
)


# ---------------------------------------------------------
# Test thresholds from 0.10 through 0.50
# ---------------------------------------------------------

thresholds = [
    round(x / 100, 2)
    for x in range(10, 51)
]


aggregate_expressions = []

for threshold in thresholds:

    key = str(threshold).replace(".", "_")

    aggregate_expressions.extend([

        F.sum(
            F.when(
                (F.col("label") == 1) &
                (F.col("late_probability") >= threshold),
                1
            ).otherwise(0)
        ).alias(f"tp_{key}"),

        F.sum(
            F.when(
                (F.col("label") == 0) &
                (F.col("late_probability") >= threshold),
                1
            ).otherwise(0)
        ).alias(f"fp_{key}"),

        F.sum(
            F.when(
                (F.col("label") == 0) &
                (F.col("late_probability") < threshold),
                1
            ).otherwise(0)
        ).alias(f"tn_{key}"),

        F.sum(
            F.when(
                (F.col("label") == 1) &
                (F.col("late_probability") < threshold),
                1
            ).otherwise(0)
        ).alias(f"fn_{key}")
    ])


threshold_counts = (
    gbt_validation_probability_df
    .agg(*aggregate_expressions)
    .first()
)


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

gbt_threshold_results = []

for threshold in thresholds:

    key = str(threshold).replace(".", "_")

    tp = threshold_counts[f"tp_{key}"]
    fp = threshold_counts[f"fp_{key}"]
    tn = threshold_counts[f"tn_{key}"]
    fn = threshold_counts[f"fn_{key}"]

    total = tp + fp + tn + fn

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    accuracy = (
        (tp + tn) / total
    )

    predicted_late_rate = (
        (tp + fp) / total
    )

    gbt_threshold_results.append({

        "threshold": threshold,

        "precision_pct":
            round(precision * 100, 2),

        "recall_pct":
            round(recall * 100, 2),

        "f1":
            round(f1, 4),

        "accuracy_pct":
            round(accuracy * 100, 2),

        "predicted_late_rate_pct":
            round(predicted_late_rate * 100, 2),

        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    })


gbt_threshold_results_df = spark.createDataFrame(
    gbt_threshold_results
)


print("=== TOP GBT VALIDATION THRESHOLDS BY F1 ===")

gbt_threshold_results_df.orderBy(
    F.desc("f1")
).show(
    15,
    truncate=False
)


# ---------------------------------------------------------
# Best validation threshold
# ---------------------------------------------------------

best_gbt_threshold_result = max(
    gbt_threshold_results,
    key=lambda x: x["f1"]
)

best_gbt_threshold = (
    best_gbt_threshold_result["threshold"]
)


print("\n=== BEST GBT VALIDATION THRESHOLD ===")

for key, value in best_gbt_threshold_result.items():
    print(f"{key}: {value}")

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 8A
# Fix mixed numeric types in threshold-results table
# No model retraining or rescoring required
# ---------------------------------------------------------

from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    LongType
)


# ---------------------------------------------------------
# Explicit schema
# ---------------------------------------------------------

threshold_schema = StructType([

    StructField("threshold", DoubleType(), False),

    StructField("precision_pct", DoubleType(), False),

    StructField("recall_pct", DoubleType(), False),

    StructField("f1", DoubleType(), False),

    StructField("accuracy_pct", DoubleType(), False),

    StructField(
        "predicted_late_rate_pct",
        DoubleType(),
        False
    ),

    StructField("tp", LongType(), False),
    StructField("fp", LongType(), False),
    StructField("tn", LongType(), False),
    StructField("fn", LongType(), False)
])


# ---------------------------------------------------------
# Force consistent Python numeric types
# ---------------------------------------------------------

gbt_threshold_results_clean = [

    (
        float(row["threshold"]),
        float(row["precision_pct"]),
        float(row["recall_pct"]),
        float(row["f1"]),
        float(row["accuracy_pct"]),
        float(row["predicted_late_rate_pct"]),
        int(row["tp"]),
        int(row["fp"]),
        int(row["tn"]),
        int(row["fn"])
    )

    for row in gbt_threshold_results
]


# ---------------------------------------------------------
# Create Spark DataFrame safely
# ---------------------------------------------------------

gbt_threshold_results_df = spark.createDataFrame(
    gbt_threshold_results_clean,
    schema=threshold_schema
)


print("=== TOP GBT VALIDATION THRESHOLDS BY F1 ===")

gbt_threshold_results_df.orderBy(
    F.desc("f1")
).show(
    15,
    truncate=False
)


# ---------------------------------------------------------
# Select best validation threshold
# ---------------------------------------------------------

best_gbt_threshold_result = max(
    gbt_threshold_results,
    key=lambda x: x["f1"]
)

best_gbt_threshold = float(
    best_gbt_threshold_result["threshold"]
)


print("\n=== BEST GBT VALIDATION THRESHOLD ===")

print(
    "threshold:",
    best_gbt_threshold
)

print(
    "precision_pct:",
    float(best_gbt_threshold_result["precision_pct"])
)

print(
    "recall_pct:",
    float(best_gbt_threshold_result["recall_pct"])
)

print(
    "f1:",
    float(best_gbt_threshold_result["f1"])
)

print(
    "accuracy_pct:",
    float(best_gbt_threshold_result["accuracy_pct"])
)

print(
    "predicted_late_rate_pct:",
    float(
        best_gbt_threshold_result[
            "predicted_late_rate_pct"
        ]
    )
)

print(
    "tp:",
    int(best_gbt_threshold_result["tp"])
)

print(
    "fp:",
    int(best_gbt_threshold_result["fp"])
)

print(
    "tn:",
    int(best_gbt_threshold_result["tn"])
)

print(
    "fn:",
    int(best_gbt_threshold_result["fn"])
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 9
# FINAL out-of-time test of Gradient-Boosted Trees
#
# Threshold selected using Oct-Nov validation only:
# LOCKED GBT THRESHOLD = 0.18
# ---------------------------------------------------------

from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator


LOCKED_GBT_THRESHOLD = 0.18


# ---------------------------------------------------------
# Prepare December using the already-fitted
# tree preprocessing model
# ---------------------------------------------------------

gbt_test_prepared = (
    tree_preprocessing_model
    .transform(test_df)
    .select(
        "flight_date",
        "label",
        "gbt_features"
    )
)


print("December test rows:", f"{gbt_test_prepared.count():,}")


# ---------------------------------------------------------
# Score December
# ---------------------------------------------------------

gbt_test_predictions = (
    gbt_light_model
    .transform(gbt_test_prepared)

    .withColumn(
        "late_probability",
        vector_to_array("probability")[1]
    )

    .withColumn(
        "threshold_prediction",
        F.when(
            F.col("late_probability") >= LOCKED_GBT_THRESHOLD,
            1.0
        ).otherwise(0.0)
    )
)


print("Scoring December with GBT...")

print(
    "Test predictions:",
    f"{gbt_test_predictions.count():,}"
)


# ---------------------------------------------------------
# ROC-AUC / PR-AUC
# ---------------------------------------------------------

gbt_test_roc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

gbt_test_pr_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)


gbt_test_roc_auc = gbt_test_roc_evaluator.evaluate(
    gbt_test_predictions
)

gbt_test_pr_auc = gbt_test_pr_evaluator.evaluate(
    gbt_test_predictions
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

gbt_test_cm = gbt_test_predictions.agg(

    F.sum(
        F.when(
            (F.col("label") == 1) &
            (F.col("threshold_prediction") == 1),
            1
        ).otherwise(0)
    ).alias("tp"),

    F.sum(
        F.when(
            (F.col("label") == 0) &
            (F.col("threshold_prediction") == 0),
            1
        ).otherwise(0)
    ).alias("tn"),

    F.sum(
        F.when(
            (F.col("label") == 0) &
            (F.col("threshold_prediction") == 1),
            1
        ).otherwise(0)
    ).alias("fp"),

    F.sum(
        F.when(
            (F.col("label") == 1) &
            (F.col("threshold_prediction") == 0),
            1
        ).otherwise(0)
    ).alias("fn")

).first()


tp = gbt_test_cm["tp"]
tn = gbt_test_cm["tn"]
fp = gbt_test_cm["fp"]
fn = gbt_test_cm["fn"]

total = tp + tn + fp + fn


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = (tp + tn) / total

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0 else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0 else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0 else 0
)

actual_late_rate = (
    (tp + fn) / total
)

predicted_late_rate = (
    (tp + fp) / total
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n=== FINAL DECEMBER GBT TEST RESULTS ===")

print("Locked threshold:", LOCKED_GBT_THRESHOLD)

print(f"ROC-AUC:   {gbt_test_roc_auc:.4f}")
print(f"PR-AUC:    {gbt_test_pr_auc:.4f}")

print()

print("Confusion matrix:")
print(f"TP: {tp:,}")
print(f"FP: {fp:,}")
print(f"TN: {tn:,}")
print(f"FN: {fn:,}")

print()

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 score:  {f1:.4f}")

print()

print(
    "Actual December late-flight rate:",
    f"{actual_late_rate * 100:.2f}%"
)

print(
    "Predicted December late-flight rate:",
    f"{predicted_late_rate * 100:.2f}%"
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 10
# Final out-of-time model comparison
# ---------------------------------------------------------

model_comparison_data = [

    (
        "Logistic Regression",
        0.17,
        0.6139,
        0.3558,
        59.23,
        34.41,
        56.04,
        0.4264,
        44.04,
        27.04,
        96105,
        183208,
        279529,
        75379
    ),

    (
        "Gradient-Boosted Trees",
        0.18,
        0.6198,
        0.3597,
        51.96,
        32.46,
        71.85,
        0.4472,
        59.86,
        27.04,
        123218,
        256421,
        206316,
        48266
    )
]


model_comparison_columns = [
    "model",
    "locked_threshold",
    "roc_auc",
    "pr_auc",
    "accuracy_pct",
    "precision_pct",
    "recall_pct",
    "f1",
    "predicted_late_rate_pct",
    "actual_late_rate_pct",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives"
]


final_model_comparison_df = spark.createDataFrame(
    model_comparison_data,
    model_comparison_columns
)


# ---------------------------------------------------------
# Add model-selection indicator
# ---------------------------------------------------------

final_model_comparison_df = (
    final_model_comparison_df
    .withColumn(
        "selected_model_flag",
        F.col("model") == "Gradient-Boosted Trees"
    )
)


print("=== FINAL OUT-OF-TIME MODEL COMPARISON ===")

display(
    final_model_comparison_df
    .orderBy(F.desc("f1"))
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 11
# Persist final predictive-model comparison
# ---------------------------------------------------------

MODEL_COMPARISON_TABLE = (
    "workspace.airline_gold.predictive_model_comparison"
)


(
    final_model_comparison_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(MODEL_COMPARISON_TABLE)
)


# ---------------------------------------------------------
# Read-back validation
# ---------------------------------------------------------

model_comparison_table = spark.table(
    MODEL_COMPARISON_TABLE
)


print("=== PERSISTED MODEL COMPARISON ===")

print(
    "Rows:",
    model_comparison_table.count()
)

print(
    "Models:",
    model_comparison_table
        .select("model")
        .distinct()
        .count()
)

print(
    "Selected models:",
    model_comparison_table
        .filter(
            F.col("selected_model_flag") == True
        )
        .count()
)


display(
    model_comparison_table
    .orderBy(F.desc("f1"))
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 12
# Build airport-month clustering base
# ---------------------------------------------------------

departure_kpi = spark.table(
    "workspace.airline_gold.monthly_airport_departure_kpi"
).alias("d")

arrival_kpi = spark.table(
    "workspace.airline_gold.monthly_airport_arrival_kpi"
).alias("a")


airport_cluster_base = (
    departure_kpi

    .join(
        arrival_kpi,
        on=[
            "source_year",
            "source_month",
            "airport_code"
        ],
        how="inner"
    )

    .select(
        "source_year",
        "source_month",
        "airport_code",

        F.col("d.airport_name").alias("airport_name"),
        F.col("d.city_name").alias("city_name"),
        F.col("d.state_code").alias("state_code"),

        # Volume
        F.col("d.total_departure_operations")
            .alias("departure_operations"),

        F.col("a.scheduled_arrival_operations")
            .alias("arrival_operations"),

        # Departure-side performance
        F.col("d.cancellation_rate_pct"),
        F.col("d.late_departure_rate_pct"),
        F.col("d.avg_departure_delay_minutes"),

        # Arrival-side performance
        F.col("a.arrival_completion_rate_pct"),
        F.col("a.late_arrival_rate_pct"),
        F.col("a.avg_arrival_delay_minutes"),

        # Diversion exposure
        F.col("d.diversion_event_rate_pct")
            .alias("departure_diversion_rate_pct"),

        F.col("a.diversion_event_rate_pct")
            .alias("arrival_diversion_rate_pct")
    )

    # Log-transform volume because airport sizes vary enormously
    .withColumn(
        "log_departure_operations",
        F.log1p("departure_operations")
    )

    .withColumn(
        "log_arrival_operations",
        F.log1p("arrival_operations")
    )
)


print("=== AIRPORT-MONTH CLUSTERING BASE ===")

print(
    "Rows:",
    airport_cluster_base.count()
)

print(
    "Distinct airports:",
    airport_cluster_base
        .select("airport_code")
        .distinct()
        .count()
)

print(
    "Months:",
    airport_cluster_base
        .select("source_month")
        .distinct()
        .count()
)


# ---------------------------------------------------------
# Missing-value audit for clustering features
# ---------------------------------------------------------

cluster_features = [
    "log_departure_operations",
    "log_arrival_operations",
    "cancellation_rate_pct",
    "late_departure_rate_pct",
    "avg_departure_delay_minutes",
    "arrival_completion_rate_pct",
    "late_arrival_rate_pct",
    "avg_arrival_delay_minutes",
    "departure_diversion_rate_pct",
    "arrival_diversion_rate_pct"
]


missing_expressions = [

    F.sum(
        F.when(
            F.col(column).isNull(),
            1
        ).otherwise(0)
    ).alias(column)

    for column in cluster_features
]


print("\n=== CLUSTER FEATURE MISSING VALUES ===")

airport_cluster_base.agg(
    *missing_expressions
).show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 13
# Compact clustering missing-value audit
# ---------------------------------------------------------

missing_row = (
    airport_cluster_base
    .agg(*missing_expressions)
    .first()
)


print("=== CLUSTER FEATURE MISSING VALUES ===")

total_missing = 0

for column in cluster_features:

    missing_count = missing_row[column]

    print(
        f"{column:35} "
        f"{missing_count:,}"
    )

    total_missing += missing_count


print()
print(
    "Total missing values across clustering features:",
    f"{total_missing:,}"
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 14
# Standardize clustering features
# ---------------------------------------------------------

from pyspark.ml.feature import VectorAssembler, StandardScaler


# ---------------------------------------------------------
# Assemble the 10 clustering variables
# ---------------------------------------------------------

cluster_assembler = VectorAssembler(
    inputCols=cluster_features,
    outputCol="raw_cluster_features",
    handleInvalid="error"
)


airport_cluster_vectorized = (
    cluster_assembler
    .transform(airport_cluster_base)
)


# ---------------------------------------------------------
# Standardize:
# mean = 0
# standard deviation = 1
# ---------------------------------------------------------

cluster_scaler = StandardScaler(
    inputCol="raw_cluster_features",
    outputCol="features",
    withMean=True,
    withStd=True
)


cluster_scaler_model = cluster_scaler.fit(
    airport_cluster_vectorized
)


airport_cluster_prepared = (
    cluster_scaler_model
    .transform(airport_cluster_vectorized)
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

print("=== STANDARDIZED CLUSTERING DATA ===")

print(
    "Rows:",
    airport_cluster_prepared.count()
)

print(
    "Clustering features:",
    len(cluster_features)
)

feature_vector_size = (
    airport_cluster_prepared
    .select("features")
    .first()["features"]
    .size
)

print(
    "Standardized feature vector size:",
    feature_vector_size
)


print("\n=== FEATURES USED FOR CLUSTERING ===")

for i, feature in enumerate(
    cluster_features,
    start=1
):
    print(f"{i:2}. {feature}")

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 15
# Select number of K-Means clusters using silhouette
# ---------------------------------------------------------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


cluster_evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)


k_values = list(range(2, 9))

cluster_selection_results = []
cluster_models = {}


print("=== K-MEANS CLUSTER SELECTION ===")

for k in k_values:

    print(f"Testing k = {k} ...")

    kmeans = KMeans(
        featuresCol="features",
        predictionCol="prediction",
        k=k,
        seed=42,
        maxIter=50,
        tol=1e-4
    )

    model = kmeans.fit(
        airport_cluster_prepared
    )

    predictions = model.transform(
        airport_cluster_prepared
    )

    silhouette = cluster_evaluator.evaluate(
        predictions
    )

    cluster_sizes = (
        predictions
        .groupBy("prediction")
        .count()
        .agg(
            F.min("count").alias("smallest_cluster"),
            F.max("count").alias("largest_cluster")
        )
        .first()
    )

    cluster_selection_results.append({
        "k": k,
        "silhouette": float(silhouette),
        "smallest_cluster": int(
            cluster_sizes["smallest_cluster"]
        ),
        "largest_cluster": int(
            cluster_sizes["largest_cluster"]
        )
    })

    cluster_models[k] = model


print("\n=== CLUSTER-SELECTION RESULTS ===")

for result in cluster_selection_results:

    print(
        f"k={result['k']} | "
        f"silhouette={result['silhouette']:.4f} | "
        f"smallest={result['smallest_cluster']:,} | "
        f"largest={result['largest_cluster']:,}"
    )


best_cluster_result = max(
    cluster_selection_results,
    key=lambda x: x["silhouette"]
)

best_k = best_cluster_result["k"]


print("\n=== BEST SILHOUETTE SOLUTION ===")

print("Best k:", best_k)
print(
    "Silhouette:",
    f"{best_cluster_result['silhouette']:.4f}"
)
print(
    "Smallest cluster:",
    f"{best_cluster_result['smallest_cluster']:,}"
)
print(
    "Largest cluster:",
    f"{best_cluster_result['largest_cluster']:,}"
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 16
# Compare operational profiles for k = 2 and k = 3
# ---------------------------------------------------------

def build_cluster_profile(k):

    model = cluster_models[k]

    predictions = (
        model
        .transform(airport_cluster_prepared)
    )

    profile = (
        predictions
        .groupBy("prediction")
        .agg(

            F.count("*")
                .alias("airport_month_profiles"),

            F.countDistinct("airport_code")
                .alias("distinct_airports"),

            F.round(
                F.avg("departure_operations"),
                1
            ).alias("avg_departure_operations"),

            F.round(
                F.avg("arrival_operations"),
                1
            ).alias("avg_arrival_operations"),

            F.round(
                F.avg("cancellation_rate_pct"),
                2
            ).alias("avg_cancellation_rate_pct"),

            F.round(
                F.avg("late_departure_rate_pct"),
                2
            ).alias("avg_late_departure_rate_pct"),

            F.round(
                F.avg("avg_departure_delay_minutes"),
                2
            ).alias("avg_departure_delay_minutes"),

            F.round(
                F.avg("arrival_completion_rate_pct"),
                2
            ).alias("avg_arrival_completion_rate_pct"),

            F.round(
                F.avg("late_arrival_rate_pct"),
                2
            ).alias("avg_late_arrival_rate_pct"),

            F.round(
                F.avg("avg_arrival_delay_minutes"),
                2
            ).alias("avg_arrival_delay_minutes"),

            F.round(
                F.avg("departure_diversion_rate_pct"),
                3
            ).alias("avg_departure_diversion_rate_pct"),

            F.round(
                F.avg("arrival_diversion_rate_pct"),
                3
            ).alias("avg_arrival_diversion_rate_pct")
        )
        .orderBy("prediction")
    )

    return predictions, profile


# ---------------------------------------------------------
# k = 2
# ---------------------------------------------------------

cluster2_predictions, cluster2_profile = (
    build_cluster_profile(2)
)

print("=== K = 2 CLUSTER PROFILE ===")

display(cluster2_profile)


# ---------------------------------------------------------
# k = 3
# ---------------------------------------------------------

cluster3_predictions, cluster3_profile = (
    build_cluster_profile(3)
)

print("=== K = 3 CLUSTER PROFILE ===")

display(cluster3_profile)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 16A
# Print complete k=2 and k=3 profiles vertically
# ---------------------------------------------------------

def print_cluster_profiles(profile_df, k):

    print(f"\n{'=' * 60}")
    print(f"K = {k} COMPLETE CLUSTER PROFILES")
    print(f"{'=' * 60}")

    rows = profile_df.orderBy("prediction").collect()

    for row in rows:

        print(f"\n--- Cluster {row['prediction']} ---")

        print(
            "Airport-month profiles:",
            f"{row['airport_month_profiles']:,}"
        )

        print(
            "Distinct airports:",
            f"{row['distinct_airports']:,}"
        )

        print(
            "Avg departure operations:",
            row["avg_departure_operations"]
        )

        print(
            "Avg arrival operations:",
            row["avg_arrival_operations"]
        )

        print(
            "Avg cancellation rate %:",
            row["avg_cancellation_rate_pct"]
        )

        print(
            "Avg late departure rate %:",
            row["avg_late_departure_rate_pct"]
        )

        print(
            "Avg departure delay minutes:",
            row["avg_departure_delay_minutes"]
        )

        print(
            "Avg arrival completion rate %:",
            row["avg_arrival_completion_rate_pct"]
        )

        print(
            "Avg late arrival rate %:",
            row["avg_late_arrival_rate_pct"]
        )

        print(
            "Avg arrival delay minutes:",
            row["avg_arrival_delay_minutes"]
        )

        print(
            "Avg departure diversion rate %:",
            row["avg_departure_diversion_rate_pct"]
        )

        print(
            "Avg arrival diversion rate %:",
            row["avg_arrival_diversion_rate_pct"]
        )


print_cluster_profiles(
    cluster2_profile,
    2
)

print_cluster_profiles(
    cluster3_profile,
    3
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 16B
# Compact comparison of k=2 and k=3
# ---------------------------------------------------------

cluster2_compact = (
    cluster2_profile
    .select(
        F.lit(2).alias("k"),
        F.col("prediction").alias("cluster"),
        "airport_month_profiles",
        "avg_departure_operations",
        "avg_cancellation_rate_pct",
        "avg_late_departure_rate_pct",
        "avg_departure_delay_minutes",
        "avg_late_arrival_rate_pct",
        "avg_arrival_delay_minutes",
        "avg_arrival_diversion_rate_pct"
    )
)

cluster3_compact = (
    cluster3_profile
    .select(
        F.lit(3).alias("k"),
        F.col("prediction").alias("cluster"),
        "airport_month_profiles",
        "avg_departure_operations",
        "avg_cancellation_rate_pct",
        "avg_late_departure_rate_pct",
        "avg_departure_delay_minutes",
        "avg_late_arrival_rate_pct",
        "avg_arrival_delay_minutes",
        "avg_arrival_diversion_rate_pct"
    )
)

cluster_comparison = (
    cluster2_compact
    .unionByName(cluster3_compact)
    .orderBy("k", "cluster")
)

print("=== COMPACT CLUSTER COMPARISON ===")

cluster_comparison.show(
    truncate=False,
    vertical=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 17
# Final K-Means solution
#
# Selected k = 3 based on:
# - strong silhouette score
# - reasonable cluster sizes
# - superior managerial interpretability
# ---------------------------------------------------------

FINAL_K = 3

final_cluster_model = cluster_models[FINAL_K]


airport_cluster_results = (
    final_cluster_model
    .transform(airport_cluster_prepared)

    .withColumnRenamed(
        "prediction",
        "cluster_id"
    )

    .withColumn(
        "cluster_label",

        F.when(
            F.col("cluster_id") == 0,
            "Delay-Stressed High-Volume"
        )

        .when(
            F.col("cluster_id") == 1,
            "Stable High-Volume"
        )

        .when(
            F.col("cluster_id") == 2,
            "Low-Volume Cancellation-Prone"
        )

        .otherwise("Unknown")
    )
)


print("=== FINAL AIRPORT OPERATIONAL SEGMENTS ===")

airport_cluster_results.groupBy(
    "cluster_id",
    "cluster_label"
).agg(

    F.count("*")
        .alias("airport_month_profiles"),

    F.countDistinct("airport_code")
        .alias("distinct_airports"),

    F.round(
        F.avg("departure_operations"),
        1
    ).alias("avg_departure_operations"),

    F.round(
        F.avg("cancellation_rate_pct"),
        2
    ).alias("avg_cancellation_rate_pct"),

    F.round(
        F.avg("late_departure_rate_pct"),
        2
    ).alias("avg_late_departure_rate_pct"),

    F.round(
        F.avg("avg_departure_delay_minutes"),
        2
    ).alias("avg_departure_delay_minutes"),

    F.round(
        F.avg("late_arrival_rate_pct"),
        2
    ).alias("avg_late_arrival_rate_pct"),

    F.round(
        F.avg("avg_arrival_delay_minutes"),
        2
    ).alias("avg_arrival_delay_minutes")

).orderBy(
    "cluster_id"
).show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 18
# Analyze airport movement between operational segments
# ---------------------------------------------------------

airport_cluster_stability = (
    airport_cluster_results

    .groupBy(
        "airport_code",
        "airport_name"
    )

    .agg(

        F.count("*")
            .alias("months_observed"),

        F.countDistinct("cluster_id")
            .alias("distinct_clusters_visited"),

        F.sum(
            F.when(
                F.col("cluster_id") == 0,
                1
            ).otherwise(0)
        ).alias("months_delay_stressed"),

        F.sum(
            F.when(
                F.col("cluster_id") == 1,
                1
            ).otherwise(0)
        ).alias("months_stable"),

        F.sum(
            F.when(
                F.col("cluster_id") == 2,
                1
            ).otherwise(0)
        ).alias("months_cancellation_prone")
    )
)


print("=== AIRPORT CLUSTER STABILITY ===")

airport_cluster_stability.groupBy(
    "distinct_clusters_visited"
).agg(

    F.count("*")
        .alias("airports"),

    F.round(
        F.avg("months_observed"),
        2
    ).alias("avg_months_observed")

).orderBy(
    "distinct_clusters_visited"
).show(truncate=False)


print("\n=== AIRPORTS THAT VISITED ALL THREE STATES ===")

airport_cluster_stability.filter(
    F.col("distinct_clusters_visited") == 3
).orderBy(
    F.desc("months_delay_stressed"),
    F.desc("months_cancellation_prone")
).show(
    30,
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 18A
# Correct airport stability analysis
#
# Airport code is the unique identifier.
# Airport name may change historically during 2025.
# ---------------------------------------------------------

airport_cluster_stability = (
    airport_cluster_results

    .groupBy(
        "airport_code"
    )

    .agg(

        # Use the latest observed airport name in 2025
        F.max_by(
            "airport_name",
            "source_month"
        ).alias("airport_name"),

        F.count("*")
            .alias("months_observed"),

        F.countDistinct("cluster_id")
            .alias("distinct_clusters_visited"),

        F.sum(
            F.when(
                F.col("cluster_id") == 0,
                1
            ).otherwise(0)
        ).alias("months_delay_stressed"),

        F.sum(
            F.when(
                F.col("cluster_id") == 1,
                1
            ).otherwise(0)
        ).alias("months_stable"),

        F.sum(
            F.when(
                F.col("cluster_id") == 2,
                1
            ).otherwise(0)
        ).alias("months_cancellation_prone")
    )
)


print("=== CORRECTED AIRPORT CLUSTER STABILITY ===")

stability_summary = (
    airport_cluster_stability
    .groupBy(
        "distinct_clusters_visited"
    )
    .agg(

        F.count("*")
            .alias("airports"),

        F.round(
            F.avg("months_observed"),
            2
        ).alias("avg_months_observed")
    )
    .orderBy(
        "distinct_clusters_visited"
    )
)

stability_summary.show(truncate=False)


print("=== STABILITY RECONCILIATION ===")

total_airports = (
    airport_cluster_stability
    .select("airport_code")
    .distinct()
    .count()
)

print(
    "Distinct airports:",
    total_airports
)

print(
    "Expected airports:",
    364
)

print(
    "Exact match:",
    total_airports == 364
)


print("\n=== AIRPORTS THAT VISITED ALL THREE STATES ===")

airport_cluster_stability.filter(
    F.col("distinct_clusters_visited") == 3
).orderBy(
    F.desc("months_delay_stressed"),
    F.desc("months_cancellation_prone"),
    "airport_code"
).show(
    30,
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 19
# Determine dominant operational state for each airport
# ---------------------------------------------------------

airport_dominant_state = (
    airport_cluster_stability

    .withColumn(
        "dominant_months",
        F.greatest(
            "months_delay_stressed",
            "months_stable",
            "months_cancellation_prone"
        )
    )

    .withColumn(
        "dominant_state",

        F.when(
            (F.col("months_delay_stressed") >
             F.col("months_stable")) &
            (F.col("months_delay_stressed") >
             F.col("months_cancellation_prone")),
            "Delay-Stressed High-Volume"
        )

        .when(
            (F.col("months_stable") >
             F.col("months_delay_stressed")) &
            (F.col("months_stable") >
             F.col("months_cancellation_prone")),
            "Stable High-Volume"
        )

        .when(
            (F.col("months_cancellation_prone") >
             F.col("months_delay_stressed")) &
            (F.col("months_cancellation_prone") >
             F.col("months_stable")),
            "Low-Volume Cancellation-Prone"
        )

        .otherwise("Mixed / Tie")
    )

    .withColumn(
        "stability_class",

        F.when(
            F.col("distinct_clusters_visited") == 1,
            "Persistent"
        )

        .when(
            F.col("distinct_clusters_visited") == 2,
            "Variable"
        )

        .when(
            F.col("distinct_clusters_visited") == 3,
            "Highly Variable"
        )
    )
)


print("=== DOMINANT AIRPORT OPERATIONAL STATES ===")

airport_dominant_state.groupBy(
    "dominant_state"
).count().orderBy(
    F.desc("count")
).show(truncate=False)


print("\n=== STABILITY CLASS ===")

airport_dominant_state.groupBy(
    "stability_class"
).count().orderBy(
    F.desc("count")
).show(truncate=False)


print("\n=== PERSISTENT AIRPORTS ===")

airport_dominant_state.filter(
    F.col("stability_class") == "Persistent"
).select(
    "airport_code",
    "airport_name",
    "months_observed",
    "dominant_state",
    "months_delay_stressed",
    "months_stable",
    "months_cancellation_prone"
).orderBy(
    "dominant_state",
    "airport_code"
).show(
    50,
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 20
# Profile airports that stayed in one operational state
# ---------------------------------------------------------

persistent_airports = (
    airport_dominant_state
    .filter(
        F.col("stability_class") == "Persistent"
    )
)


print("=== PERSISTENT AIRPORTS BY OPERATIONAL STATE ===")

persistent_airports.groupBy(
    "dominant_state"
).count().orderBy(
    F.desc("count")
).show(truncate=False)


print("\n=== PERSISTENT STABLE HIGH-VOLUME AIRPORTS ===")

persistent_airports.filter(
    F.col("dominant_state") == "Stable High-Volume"
).select(
    "airport_code",
    "airport_name",
    "months_observed",
    "months_stable"
).orderBy(
    "airport_code"
).show(100, truncate=False)


print("\n=== PERSISTENT DELAY-STRESSED AIRPORTS ===")

persistent_airports.filter(
    F.col("dominant_state") == "Delay-Stressed High-Volume"
).select(
    "airport_code",
    "airport_name",
    "months_observed",
    "months_delay_stressed"
).orderBy(
    "airport_code"
).show(100, truncate=False)


print("\n=== PERSISTENT CANCELLATION-PRONE AIRPORTS ===")

persistent_airports.filter(
    F.col("dominant_state") == "Low-Volume Cancellation-Prone"
).select(
    "airport_code",
    "airport_name",
    "months_observed",
    "months_cancellation_prone"
).orderBy(
    "airport_code"
).show(100, truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 20A
# Inspect persistently delay-stressed airports
# ---------------------------------------------------------

print("=== PERSISTENT DELAY-STRESSED AIRPORTS ===")

persistent_airports.filter(
    F.col("dominant_state") == "Delay-Stressed High-Volume"
).select(
    "airport_code",
    "airport_name",
    "months_observed",
    "months_delay_stressed",
    "months_stable",
    "months_cancellation_prone"
).orderBy(
    "airport_code"
).show(
    truncate=False
)


print("\n=== PERSISTENT CANCELLATION-PRONE AIRPORTS ===")

persistent_airports.filter(
    F.col("dominant_state") == "Low-Volume Cancellation-Prone"
).select(
    "airport_code",
    "airport_name",
    "months_observed",
    "months_cancellation_prone"
).show(
    truncate=False
)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 20B
# Refine persistence classification
#
# "Full-Year Persistent" requires:
#   - 12 months observed
#   - exactly one cluster visited
#
# Airports observed for fewer than 12 months should not
# be interpreted as persistent for the full year.
# ---------------------------------------------------------

airport_dominant_state_refined = (
    airport_dominant_state

    .withColumn(
        "refined_stability_class",

        F.when(
            (F.col("months_observed") == 12) &
            (F.col("distinct_clusters_visited") == 1),
            "Full-Year Persistent"
        )

        .when(
            (F.col("months_observed") < 12) &
            (F.col("distinct_clusters_visited") == 1),
            "Limited-Observation Single-State"
        )

        .when(
            F.col("distinct_clusters_visited") == 2,
            "Variable"
        )

        .when(
            F.col("distinct_clusters_visited") == 3,
            "Highly Variable"
        )

        .otherwise("Other")
    )
)


print("=== REFINED AIRPORT STABILITY CLASS ===")

airport_dominant_state_refined.groupBy(
    "refined_stability_class"
).count().orderBy(
    F.desc("count")
).show(truncate=False)


print("\n=== FULL-YEAR PERSISTENT AIRPORTS BY STATE ===")

airport_dominant_state_refined.filter(
    F.col("refined_stability_class") == "Full-Year Persistent"
).groupBy(
    "dominant_state"
).count().orderBy(
    F.desc("count")
).show(truncate=False)


print("\n=== FULL-YEAR PERSISTENT DELAY-STRESSED AIRPORTS ===")

airport_dominant_state_refined.filter(
    (F.col("refined_stability_class") == "Full-Year Persistent") &
    (F.col("dominant_state") == "Delay-Stressed High-Volume")
).select(
    "airport_code",
    "airport_name",
    "months_observed",
    "months_delay_stressed"
).show(truncate=False)


print("\n=== LIMITED-OBSERVATION SINGLE-STATE AIRPORTS ===")

airport_dominant_state_refined.filter(
    F.col("refined_stability_class") ==
    "Limited-Observation Single-State"
).select(
    "airport_code",
    "airport_name",
    "months_observed",
    "dominant_state"
).orderBy(
    "months_observed",
    "airport_code"
).show(50, truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 20C
# Identify exceptional persistence cases
# ---------------------------------------------------------

print("=== ONLY FULL-YEAR PERSISTENT DELAY-STRESSED AIRPORT ===")

airport_dominant_state_refined.filter(
    (F.col("refined_stability_class") == "Full-Year Persistent") &
    (F.col("dominant_state") == "Delay-Stressed High-Volume")
).select(
    "airport_code",
    "airport_name",
    "months_observed",
    "months_delay_stressed",
    "months_stable",
    "months_cancellation_prone"
).show(truncate=False)


print("\n=== LIMITED-OBSERVATION SINGLE-STATE AIRPORTS ===")

airport_dominant_state_refined.filter(
    F.col("refined_stability_class") ==
    "Limited-Observation Single-State"
).select(
    "airport_code",
    "airport_name",
    "months_observed",
    "dominant_state"
).orderBy(
    "months_observed",
    "airport_code"
).show(truncate=False)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 21
# Persist final clustering analytics
# ---------------------------------------------------------

AIRPORT_SEGMENT_TABLE = (
    "workspace.airline_gold.airport_month_operational_segments"
)

AIRPORT_STABILITY_TABLE = (
    "workspace.airline_gold.airport_operational_stability_2025"
)


# ---------------------------------------------------------
# Business-ready airport-month segmentation table
# Exclude Spark ML vector columns from persisted Gold table
# ---------------------------------------------------------

airport_segment_gold = (
    airport_cluster_results
    .select(
        "source_year",
        "source_month",
        "airport_code",
        "airport_name",
        "city_name",
        "state_code",

        "departure_operations",
        "arrival_operations",

        "cancellation_rate_pct",
        "late_departure_rate_pct",
        "avg_departure_delay_minutes",

        "arrival_completion_rate_pct",
        "late_arrival_rate_pct",
        "avg_arrival_delay_minutes",

        "departure_diversion_rate_pct",
        "arrival_diversion_rate_pct",

        "cluster_id",
        "cluster_label"
    )

    .withColumn(
        "selected_k",
        F.lit(3)
    )

    .withColumn(
        "selected_k_silhouette",
        F.lit(0.3906)
    )
)


# ---------------------------------------------------------
# Annual airport stability table
# ---------------------------------------------------------

airport_stability_gold = (
    airport_dominant_state_refined
    .select(
        "airport_code",
        "airport_name",
        "months_observed",
        "distinct_clusters_visited",

        "months_delay_stressed",
        "months_stable",
        "months_cancellation_prone",

        "dominant_months",
        "dominant_state",
        "refined_stability_class"
    )
)


# ---------------------------------------------------------
# Persist both as Delta
# ---------------------------------------------------------

(
    airport_segment_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(AIRPORT_SEGMENT_TABLE)
)


(
    airport_stability_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(AIRPORT_STABILITY_TABLE)
)


print("Clustering Gold tables created successfully.")

print()
print("Airport-month segments:")
print(AIRPORT_SEGMENT_TABLE)

print()
print("Airport stability summary:")
print(AIRPORT_STABILITY_TABLE)

# COMMAND ----------

# ---------------------------------------------------------
# Phase 9 — Step 22
# Validate persisted clustering Gold tables
# ---------------------------------------------------------

AIRPORT_SEGMENT_TABLE = (
    "workspace.airline_gold.airport_month_operational_segments"
)

AIRPORT_STABILITY_TABLE = (
    "workspace.airline_gold.airport_operational_stability_2025"
)


segment_table = spark.table(
    AIRPORT_SEGMENT_TABLE
)

stability_table = spark.table(
    AIRPORT_STABILITY_TABLE
)


# ---------------------------------------------------------
# Airport-month segmentation table
# ---------------------------------------------------------

print("=== PERSISTED AIRPORT-MONTH SEGMENTS ===")

print(
    "Rows:",
    segment_table.count()
)

print(
    "Distinct airports:",
    segment_table
        .select("airport_code")
        .distinct()
        .count()
)

print(
    "Months:",
    segment_table
        .select("source_month")
        .distinct()
        .count()
)

print(
    "Clusters:",
    segment_table
        .select("cluster_id")
        .distinct()
        .count()
)


print("\n=== CLUSTER DISTRIBUTION ===")

segment_table.groupBy(
    "cluster_id",
    "cluster_label"
).count().orderBy(
    "cluster_id"
).show(truncate=False)


# ---------------------------------------------------------
# Airport annual stability table
# ---------------------------------------------------------

print("\n=== PERSISTED AIRPORT STABILITY ===")

print(
    "Rows:",
    stability_table.count()
)

print(
    "Distinct airports:",
    stability_table
        .select("airport_code")
        .distinct()
        .count()
)


print("\n=== REFINED STABILITY DISTRIBUTION ===")

stability_table.groupBy(
    "refined_stability_class"
).count().orderBy(
    F.desc("count")
).show(truncate=False)