"""
ml_models.py
============
Phase 3: Three machine learning models for aerospace quality analytics.

  Model 1 — Anomaly Detection   : Isolation Forest (unsupervised)
  Model 2 — Defect Prediction   : Random Forest classifier (supervised)
  Model 3 — Root Cause Analysis : XGBoost + feature importance (supervised)

WHAT YOU'LL LEARN IN THIS FILE
--------------------------------
1. The difference between supervised and unsupervised ML
2. Train/test splits — why we hold out data to evaluate models honestly
3. How to evaluate a classifier: accuracy, precision, recall, confusion matrix
4. Feature importance — what the model thinks matters most
5. How to save a trained model so the dashboard can load it instantly
"""

import pandas as pd
import numpy as np
import os
import json
import pickle   # Built into Python. Serializes Python objects to binary files.
                # We use it to save trained models so we don't retrain every time.

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# XGBoost is a separate package — already installed with our pip command
import xgboost as xgb

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================

def load_features() -> pd.DataFrame:
    """Loads the ML feature DataFrame built in Phase 2."""
    path = "data/ml_features.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "data/ml_features.csv not found. Run pipeline.py first."
        )
    df = pd.read_csv(path)
    print(f"Loaded ML features: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Failure rate: {round(100 * df['target_fail'].mean(), 1)}%")
    return df


# =============================================================================
# SECTION 2: MODEL 1 — ANOMALY DETECTION (Isolation Forest)
# =============================================================================

def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses Isolation Forest to find statistically unusual inspections.

    LEARNING NOTE — Unsupervised vs Supervised ML:
    -----------------------------------------------
    Supervised learning: you give the model labeled examples
    ("this inspection failed, this one passed") and it learns the pattern.
    Random Forest and XGBoost are supervised.

    Unsupervised learning: you give the model unlabeled data and say
    "find the weird stuff." No labels required. Isolation Forest is unsupervised.

    This is useful in manufacturing because:
    - Sometimes you don't have reliable labels
    - Sometimes the "weird" inspection is a NEW type of failure you've
      never seen before — a supervised model won't catch it, but an
      anomaly detector will

    HOW ISOLATION FOREST WORKS:
    It builds random decision trees. Normal points (the majority) require
    many splits to isolate. Anomalies (the weird ones) are isolated quickly
    in just a few splits. The anomaly score = how quickly it was isolated.
    Low score = normal. High score = anomaly.
    """
    print(f"\n{'='*55}")
    print("  Model 1: Anomaly Detection (Isolation Forest)")
    print(f"{'='*55}")

    # For anomaly detection we use ALL features except the target
    # LEARNING NOTE: .drop(columns=[...]) removes columns from a DataFrame.
    # We store the result in X (convention: X = features, y = target label).
    feature_cols = [c for c in df.columns if c != "target_fail"]
    X = df[feature_cols].copy()

    # LEARNING NOTE — Feature Scaling:
    # Isolation Forest is sensitive to scale differences between columns.
    # visual_score ranges from 1-10, but planned_qty ranges from 10-200.
    # Without scaling, planned_qty would dominate just because it's bigger.
    # StandardScaler transforms each column to have mean=0, std=1.
    # After scaling: visual_score and planned_qty are on equal footing.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # ^ fit_transform() does two things at once:
    #   fit()       = calculate the mean and std of each column
    #   transform() = apply the scaling using those stats
    # On new data you'd only call transform() (not fit again) to use the
    # same scale learned from training data.

    # Train the Isolation Forest
    # contamination = our estimate of what fraction of data are anomalies
    # We set it to match our actual failure rate (~18%)
    # n_estimators = number of trees to build (more = more stable, slower)
    # random_state = seed for reproducibility (same seed = same result every run)
    iso_forest = IsolationForest(
        contamination=0.18,
        n_estimators=100,
        random_state=42,
    )
    iso_forest.fit(X_scaled)

    # Get predictions and anomaly scores
    # predict() returns: 1 = normal, -1 = anomaly (sklearn convention)
    # decision_function() returns the raw anomaly score (lower = more anomalous)
    predictions = iso_forest.predict(X_scaled)
    scores      = iso_forest.decision_function(X_scaled)

    # Build results DataFrame
    df_anomalies = df[feature_cols].copy()
    df_anomalies["anomaly_score"]  = scores
    df_anomalies["is_anomaly"]     = (predictions == -1).astype(int)
    # ^ (predictions == -1) gives True/False. .astype(int) converts to 1/0.
    df_anomalies["target_fail"]    = df["target_fail"]

    # Normalize score to 0-100 for display (higher = more anomalous)
    min_score = df_anomalies["anomaly_score"].min()
    max_score = df_anomalies["anomaly_score"].max()
    df_anomalies["anomaly_score_pct"] = (
        100 * (df_anomalies["anomaly_score"] - min_score) / (max_score - min_score)
    ).round(1)
    # This is called min-max normalization. Formula: (x - min) / (max - min)
    # It scales any range of numbers to 0-100.
    # We INVERT it because lower raw scores = more anomalous:
    df_anomalies["anomaly_score_pct"] = 100 - df_anomalies["anomaly_score_pct"]

    n_anomalies = df_anomalies["is_anomaly"].sum()
    print(f"  Total inspections:  {len(df_anomalies)}")
    print(f"  Flagged anomalies:  {n_anomalies} ({round(100*n_anomalies/len(df_anomalies),1)}%)")

    # How well does anomaly = actual failure? (just as a sanity check)
    # LEARNING NOTE: A confusion matrix shows:
    #   [[True Negatives,  False Positives],
    #    [False Negatives, True Positives ]]
    actual    = df_anomalies["target_fail"]
    predicted = df_anomalies["is_anomaly"]
    cm = confusion_matrix(actual, predicted)
    print(f"\n  Confusion matrix vs actual failures:")
    print(f"  True Negatives  (correct normal):   {cm[0][0]}")
    print(f"  False Positives (wrong alarm):       {cm[0][1]}")
    print(f"  False Negatives (missed failure):    {cm[1][0]}")
    print(f"  True Positives  (caught failure):    {cm[1][1]}")

    return df_anomalies


# =============================================================================
# SECTION 3: MODEL 2 — DEFECT PREDICTION (Random Forest)
# =============================================================================

def run_defect_prediction(df: pd.DataFrame) -> tuple:
    """
    Trains a Random Forest classifier to predict whether an inspection
    will fail, given the production line, shift, part, etc.

    Returns: (trained model, predictions DataFrame, feature importance DataFrame)

    LEARNING NOTE — Train / Test Split:
    ------------------------------------
    The cardinal sin of ML is evaluating your model on the same data it
    trained on. Of course it scores well — it memorized the answers!

    train_test_split() randomly divides your data:
    - Training set (80%): the model learns from this
    - Test set (20%):     we hide this from the model, then evaluate on it

    If the model scores well on the test set, it has genuinely learned
    a pattern — not just memorized the training data.
    This is called generalization.
    """
    print(f"\n{'='*55}")
    print("  Model 2: Defect Prediction (Random Forest)")
    print(f"{'='*55}")

    feature_cols = [c for c in df.columns if c != "target_fail"]
    X = df[feature_cols]
    y = df["target_fail"]

    # Split into training and test sets
    # test_size=0.2  → 20% held out for testing
    # random_state=42 → reproducible split
    # stratify=y     → ensures the test set has the same failure rate
    #                   as the full dataset (important for imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {len(X_train)} rows")
    print(f"  Test set:     {len(X_test)} rows")

    # Train the Random Forest
    # n_estimators=200  → build 200 decision trees
    # max_depth=8        → limit tree depth to prevent overfitting
    # class_weight="balanced" → automatically handles imbalanced classes
    #   (we have ~18% failures vs 82% passes — without this, the model
    #    might just predict "Pass" every time and score 82% accuracy!)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,   # Use all CPU cores — faster training
    )
    rf_model.fit(X_train, y_train)
    print("  Model trained.")

    # Evaluate on the held-out test set
    y_pred       = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    # ^ predict_proba returns [[prob_pass, prob_fail], ...] for each row
    # [:, 1] slices out just the "fail" probability column

    # LEARNING NOTE — Evaluation Metrics:
    # Accuracy  = (correct predictions) / (total predictions)
    #             Not great alone for imbalanced data
    # Precision = of all predicted failures, what % were actual failures?
    #             High precision = few false alarms
    # Recall    = of all actual failures, what % did we catch?
    #             High recall = few missed failures
    # ROC-AUC   = overall model discrimination ability, 0.5=random, 1.0=perfect
    #
    # In quality engineering: RECALL matters more than precision.
    # A missed failure (false negative) that escapes to the field is
    # far more costly than a false alarm that triggers an extra inspection.

    auc_score = roc_auc_score(y_test, y_pred_proba)
    report    = classification_report(y_test, y_pred, target_names=["Pass", "Fail"])

    print(f"\n  ROC-AUC score: {round(auc_score, 3)}")
    print(f"  (0.5 = random guessing, 1.0 = perfect)")
    print(f"\n  Classification report (test set):")
    for line in report.split('\n'):
        if line.strip():
            print(f"    {line}")

    # Build predictions DataFrame — all rows, with probabilities
    df_predictions = df[feature_cols].copy()
    df_predictions["target_fail"]       = y.values
    df_predictions["predicted_fail"]    = rf_model.predict(X)
    df_predictions["fail_probability"]  = rf_model.predict_proba(X)[:, 1].round(3)
    df_predictions["risk_tier"] = pd.cut(
        df_predictions["fail_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )
    # ^ pd.cut() bins a continuous value into categories.
    # fail_probability 0.0-0.3 → "Low", 0.3-0.6 → "Medium", 0.6-1.0 → "High"

    # Feature importance from the Random Forest
    df_importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    # ^ .sort_values() sorts a DataFrame by a column. ascending=False = highest first.
    # .reset_index(drop=True) resets row numbers after sorting (0, 1, 2, ...)

    print(f"\n  Top 10 most important features:")
    for _, row in df_importance.head(10).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"    {row['feature']:<30} {bar} {round(row['importance']*100, 1)}%")
    # ^ iterrows() lets you loop through DataFrame rows as (index, Series) pairs

    high_risk = (df_predictions["risk_tier"] == "High").sum()
    print(f"\n  High-risk inspections: {high_risk} ({round(100*high_risk/len(df_predictions),1)}%)")

    return rf_model, df_predictions, df_importance


# =============================================================================
# SECTION 4: MODEL 3 — ROOT CAUSE ANALYSIS (XGBoost)
# =============================================================================

def run_rca_model(df: pd.DataFrame) -> tuple:
    """
    Trains an XGBoost model and extracts feature importances as the
    automated Root Cause Analysis engine.

    LEARNING NOTE — Why XGBoost for RCA?
    --------------------------------------
    XGBoost (eXtreme Gradient Boosting) builds trees sequentially.
    Each new tree focuses on the mistakes the previous trees made.
    This "boosting" process makes it very good at finding subtle patterns.

    But the real reason we use it for RCA is SHAP-style feature importance:
    XGBoost can tell you not just WHICH features matter, but HOW they
    contribute. "Night shift increases failure probability by X%" is a
    root cause finding. That's what makes this dashboard valuable.

    LEARNING NOTE — Gradient Boosting vs Random Forest:
    Random Forest: many trees in parallel, majority vote
    Gradient Boosting: trees in sequence, each corrects the last
    Generally: XGBoost is more accurate, Random Forest is more stable
    We use both to show you different approaches.
    """
    print(f"\n{'='*55}")
    print("  Model 3: Root Cause Analysis (XGBoost)")
    print(f"{'='*55}")

    feature_cols = [c for c in df.columns if c != "target_fail"]
    X = df[feature_cols]
    y = df["target_fail"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate scale_pos_weight to handle class imbalance
    # This tells XGBoost how much more to weight the minority class (failures)
    # Formula: count of negatives / count of positives
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"  Class balance — Pass: {neg_count}, Fail: {pos_count}")
    print(f"  scale_pos_weight: {round(scale_pos_weight, 2)}")

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        # ^ learning_rate (also called "eta") controls how much each tree
        # contributes. Smaller = more conservative = less overfitting.
        # Typical range: 0.01 to 0.3
        subsample=0.8,
        # ^ Use 80% of rows per tree (adds randomness, reduces overfitting)
        colsample_bytree=0.8,
        # ^ Use 80% of features per tree (same idea)
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        verbosity=0,   # Silence XGBoost's training output
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print("  Model trained.")

    # Evaluate
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    auc_score    = roc_auc_score(y_test, y_pred_proba)
    print(f"  ROC-AUC score: {round(auc_score, 3)}")

    # --- Feature Importance for RCA ---
    # XGBoost gives us three types of importance:
    # "weight"  = how often a feature is used to split
    # "gain"    = average improvement in accuracy when this feature is used
    # "cover"   = average number of samples affected by splits on this feature
    #
    # "gain" is the most meaningful for RCA — it tells you which features
    # actually IMPROVE predictions, not just which ones are used most.

    importance_weight = xgb_model.get_booster().get_score(importance_type="weight")
    importance_gain   = xgb_model.get_booster().get_score(importance_type="gain")
    importance_cover  = xgb_model.get_booster().get_score(importance_type="cover")

    # Build a clean RCA DataFrame combining all three importance types
    all_features = set(importance_gain.keys()) | set(importance_weight.keys())
    rca_rows = []
    for feat in all_features:
        rca_rows.append({
            "feature":          feat,
            "importance_gain":  importance_gain.get(feat, 0),
            "importance_weight": importance_weight.get(feat, 0),
            "importance_cover": importance_cover.get(feat, 0),
        })

    df_rca = pd.DataFrame(rca_rows)

    # Normalize gain to percentage of total (easier to explain)
    total_gain = df_rca["importance_gain"].sum()
    df_rca["gain_pct"] = (df_rca["importance_gain"] / total_gain * 100).round(2)

    df_rca = df_rca.sort_values("gain_pct", ascending=False).reset_index(drop=True)

    # Add human-readable category labels for the dashboard
    # LEARNING NOTE: .apply() runs a function on every value in a column.
    # Here we use a lambda (anonymous function) to categorize each feature name.
    def categorize_feature(feat_name: str) -> str:
        if feat_name.startswith("shift_"):      return "Shift"
        if feat_name.startswith("line_"):       return "Production Line"
        if feat_name.startswith("part_"):       return "Part Number"
        if feat_name.startswith("insp_type_"): return "Inspection Type"
        if feat_name in ("visual_score", "dimension_error"): return "Inspection Measurement"
        if feat_name in ("planned_qty", "actual_qty", "yield_pct"): return "Production Metrics"
        return "Time / Other"

    df_rca["category"] = df_rca["feature"].apply(categorize_feature)

    # Roll up to category level for the dashboard summary
    df_rca_summary = (
        df_rca.groupby("category")["gain_pct"]
        .sum()
        .reset_index()
        .sort_values("gain_pct", ascending=False)
        .rename(columns={"gain_pct": "total_gain_pct"})
    )
    # LEARNING NOTE: .groupby("category")["gain_pct"].sum() is the Pandas
    # equivalent of SQL: SELECT category, SUM(gain_pct) GROUP BY category
    # .rename(columns={...}) renames specific columns

    print(f"\n  RCA — Risk contribution by category:")
    for _, row in df_rca_summary.iterrows():
        bar = "█" * int(row["total_gain_pct"] / 2)
        print(f"    {row['category']:<28} {bar} {row['total_gain_pct']:.1f}%")

    print(f"\n  Top 10 individual risk factors:")
    for _, row in df_rca.head(10).iterrows():
        print(f"    {row['feature']:<35} {row['gain_pct']:.1f}% (gain)")

    return xgb_model, df_rca, df_rca_summary


# =============================================================================
# SECTION 5: SAVE MODEL OUTPUTS
# =============================================================================

def save_model_outputs(
    df_anomalies:   pd.DataFrame,
    rf_model,
    df_predictions: pd.DataFrame,
    df_rf_importance: pd.DataFrame,
    xgb_model,
    df_rca:         pd.DataFrame,
    df_rca_summary: pd.DataFrame,
) -> None:
    """
    Saves:
    - CSVs for the dashboard to display
    - Trained model files (.pkl) so the dashboard loads instantly
      without retraining every time it opens
    """
    print(f"\n{'='*55}")
    print("  Saving model outputs...")
    print(f"{'='*55}")

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # --- Save CSVs ---
    csv_outputs = {
        "data/anomaly_scores.csv":  df_anomalies[["anomaly_score_pct", "is_anomaly", "target_fail"]],
        "data/predictions.csv":     df_predictions[["target_fail", "predicted_fail", "fail_probability", "risk_tier"]],
        "data/rf_importance.csv":   df_rf_importance,
        "data/rca_factors.csv":     df_rca,
        "data/rca_summary.csv":     df_rca_summary,
    }
    for path, df in csv_outputs.items():
        df.to_csv(path, index=False)
        print(f"  Saved {path}")

    # --- Save trained models as .pkl files ---
    # LEARNING NOTE: pickle.dump() serializes a Python object to a binary file.
    # This means the dashboard can load the already-trained model in milliseconds
    # instead of retraining from scratch every time someone opens the page.
    # "wb" = write binary (always use this mode for pickle files)
    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    print("  Saved models/random_forest.pkl")

    with open("models/xgboost.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    print("  Saved models/xgboost.pkl")

    # Save a model metadata JSON — useful for the dashboard "about" section
    # and for anyone reading your GitHub repo
    metadata = {
        "random_forest": {
            "type": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 8,
            "purpose": "Predict inspection pass/fail probability",
        },
        "xgboost": {
            "type": "XGBClassifier",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "purpose": "Root cause analysis via feature importance",
        },
        "isolation_forest": {
            "type": "IsolationForest",
            "contamination": 0.18,
            "purpose": "Unsupervised anomaly detection",
        },
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Saved models/metadata.json")


# =============================================================================
# SECTION 6: MAIN
# =============================================================================

def main():
    print(f"\n{'='*55}")
    print("  Aerospace Quality Analytics — ML Models")
    print(f"{'='*55}\n")

    # Load the feature data built in Phase 2
    df = load_features()

    # Run all three models
    df_anomalies                        = run_anomaly_detection(df)
    rf_model, df_predictions, df_rf_imp = run_defect_prediction(df)
    xgb_model, df_rca, df_rca_summary   = run_rca_model(df)

    # Save everything
    save_model_outputs(
        df_anomalies,
        rf_model, df_predictions, df_rf_imp,
        xgb_model, df_rca, df_rca_summary,
    )

    print(f"\n{'='*55}")
    print("  ML models complete! Ready for Phase 4: Dashboard")
    print(f"{'='*55}\n")

    # STRETCH EXERCISE (try before Phase 4):
    # ----------------------------------------
    # Look at data/predictions.csv and answer this question:
    # What percentage of High-risk inspections were actual failures?
    # Hint: load the CSV with pd.read_csv(), then filter for
    # risk_tier == "High" and check what target_fail looks like.
    # This is called "precision for the High risk tier."
    # ----------------------------------------


if __name__ == "__main__":
    main()