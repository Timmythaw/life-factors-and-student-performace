"""
Complete training pipeline for the Student Risk Classifier model.
"""

import logging
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DATA = DATA_DIR / "interim"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FEATURE CONFIGURATION (from your notebook)
# =============================================================================
FEATURE_NAMES = [
    "absences", "studytime", "freetime", "health", "goout", "Dalc", "Walc",
    "higher_yes", "schoolsup_yes", "romantic_yes", "paid_yes",
    "guardian_mother", "guardian_other",
    "Mjob_health", "Mjob_other", "Mjob_services", "Mjob_teacher",
    "reason_home", "reason_other", "reason_reputation",
    "internet_yes", "sex_M", "subject_por",
    "alc_total", "study_absence_ratio", "alc_goout",
    "absences_bin_low", "absences_bin_medium", "absences_bin_high",
    "goout_binned_medium", "goout_binned_high",
    "alc_level_moderate", "alc_level_high",
]

TARGET_COLUMN = "at_risk"

# =============================================================================
# MODEL CONFIGURATION (from your notebook screenshot)
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25

def load_data(data_path: Path) -> pd.DataFrame:
    """Load the processed data from a CSV file."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df

def prepare_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target variable."""
    X = df.drop(columns=[TARGET_COLUMN], axis=1)
    y = df[TARGET_COLUMN]

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    logger.info(f"Target at_risk ratio: {y.mean():.2%}")

    return X, y

def handle_missing_values(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Handle missing values by dropping rows with any missing values."""
    logger.info("Checking for missing values.")

    nan_in_y = y.isna().sum()
    nan_in_X = X.isna().sum().sum()

    logger.info(f"NaN values in y: {nan_in_y}")
    logger.info(f"NaN values in X: {nan_in_X}")

    if nan_in_y > 0 or nan_in_X > 0:
        # Create mask for valid rows
        mask = (~y.isna()) & (~X.isna().any(axis=1))
        X = X[mask]
        y = y[mask]
        logger.info(f"After removing NaN: {len(X)} records remaining.")

    return X, y

def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into train, validation and test sets."""
    logger.info("Splitting data into train, validation, and test sets...")

    # First Split: seperate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Second Split: sperate validation set (25% of remaining = 20% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VAL_SIZE,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    logger.info(f"Train set: {len(X_train)} samples ({len(X_train)/len(X):.1%})")
    logger.info(f"Val set:   {len(X_val)} samples ({len(X_val)/len(X):.1%})")
    logger.info(f"Test set:  {len(X_test)} samples ({len(X_test)/len(X):.1%})")
    
    logger.info(f"Train at_risk ratio: {y_train.mean():.2%}")
    logger.info(f"Val at_risk ratio:   {y_val.mean():.2%}")
    logger.info(f"Test at_risk ratio:  {y_test.mean():.2%}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on training data."""
    logger.info("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    logger.info(f"StandardScaler fitted on {scaler.n_features_in_} features.")
    return scaler

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBClassifier:
    """Train XGBoost classifier with early stopping."""
    logger.info("Training XGBoost classifier...")

    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight: {scale_pos_weight:.4f}")

    # XGBoost configuration (Final paremeters from notebook after hyperparameter tuning)
    model = XGBClassifier(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=50,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0.5,
        reg_lamda=1,
        colsample_bytree=0.9,
        subsample=0.9,
        use_label_encoder=False,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=RANDOM_STATE,
    )

    # Fit with early stopping using validation set
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    logger.info(f"Best score: {model.best_score:.4f}")
    
    return model

def evaluate_model(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "Test"
) -> dict:
    """Evaluate model performance."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating on {dataset_name} set...")
    logger.info(f"{'='*50}")
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }
    
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    logger.info(f"\nClassification Report:\n{classification_report(y, y_pred)}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y, y_pred)}")
    
    return metrics

def get_feature_importance(model: XGBClassifier, feature_names: list) -> pd.DataFrame:
    """Extract and display feature importance."""
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    logger.info("\nTop 10 Most Important Features:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance_df

def save_artifacts(
    model: XGBClassifier,
    scaler: StandardScaler,
    feature_names: list,
    metrics: dict,
    importance_df: pd.DataFrame
):
    """Save model, scaler, and metadata."""
    logger.info("\nSaving model artifacts...")
    
    # Save model
    model_path = MODELS_DIR / "combined_xgb_final_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = MODELS_DIR / "combined_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Save feature names
    feature_names_path = MODELS_DIR / "combined_feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"Feature names saved to {feature_names_path}")
    
    # Save metrics
    metrics_path = REPORTS_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save feature importance
    importance_path = REPORTS_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to {importance_path}")

def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("STUDENT RISK CLASSIFIER - TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # 1. Load data
    data_path = INTERIM_DATA / "student-final-engineered.csv"
    df = load_data(data_path)
    
    # 2. Prepare features and target
    X, y = prepare_features_target(df)
    
    # 3. Handle missing values
    X, y = handle_missing_values(X, y)
    
    # 4. Ensure correct feature order
    logger.info("Ensuring correct feature order...")
    X = X[FEATURE_NAMES]
    logger.info(f"Features reordered: {list(X.columns)}")
    
    # 5. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # 6. Fit scaler on training data
    scaler = fit_scaler(X_train)
    
    # 7. Scale all sets
    logger.info("Scaling features...")
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=FEATURE_NAMES,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=FEATURE_NAMES,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURE_NAMES,
        index=X_test.index
    )
    
    # 8. Train model
    model = train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # 9. Evaluate on all sets
    train_metrics = evaluate_model(model, X_train_scaled, y_train, "Train")
    val_metrics = evaluate_model(model, X_val_scaled, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test_scaled, y_test, "Test")
    
    # 10. Feature importance
    importance_df = get_feature_importance(model, FEATURE_NAMES)
    
    # 11. Save artifacts
    all_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }
    save_artifacts(model, scaler, FEATURE_NAMES, all_metrics, importance_df)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score:  {test_metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    logger.info(f"\nArtifacts saved to: {MODELS_DIR}")
    logger.info(f"Reports saved to: {REPORTS_DIR}")
    logger.info("\n🚀 Next step: Start the API with:")
    logger.info("   uvicorn src.main:app --reload")


if __name__ == "__main__":
    main()