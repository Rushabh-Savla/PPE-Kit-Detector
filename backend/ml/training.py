"""
Model Training Pipeline.

Trains the ensemble of ML models on historical data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, classification_report
from .models import PredictionModelEnsemble


def train_models(training_df: pd.DataFrame, test_size: float = 0.2) -> PredictionModelEnsemble:
    """
    Train all models in the ensemble.

    Args:
        training_df: DataFrame with features and labels
        test_size: Fraction of data to use for testing

    Returns:
        Trained PredictionModelEnsemble
    """
    print("="*60)
    print("Training Predictive Models")
    print("="*60)

    # Separate features and labels
    feature_columns = [col for col in training_df.columns
                      if not col.startswith("label_") and
                      col not in ["worker_id", "employee_id"]]

    X = training_df[feature_columns]
    print(f"Features: {len(feature_columns)} columns")
    print(f"Training examples: {len(X)}")

    # Handle categorical features
    X = _encode_categorical_features(X)

    # Split data
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize ensemble
    ensemble = PredictionModelEnsemble()
    ensemble.feature_scaler = scaler
    ensemble.feature_names = list(X.columns)

    # 1. Train Violation Prediction Model (Regression)
    print("\n" + "="*60)
    print("1. Training Violation Prediction Model (Regression)")
    print("="*60)

    y_violations_train = training_df.loc[X_train.index, "label_violations_next_30d"]
    y_violations_test = training_df.loc[X_test.index, "label_violations_next_30d"]

    violation_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    violation_model.fit(X_train_scaled, y_violations_train)
    ensemble.violation_model = violation_model

    # Evaluate
    y_pred = violation_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_violations_test, y_pred))
    r2 = r2_score(y_violations_test, y_pred)
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.3f}")

    # 2. Train Attendance Classification Model (Binary)
    print("\n" + "="*60)
    print("2. Training Attendance Classification Model (Binary)")
    print("="*60)

    y_attendance_train = training_df.loc[X_train.index, "label_has_attendance_issue"]
    y_attendance_test = training_df.loc[X_test.index, "label_has_attendance_issue"]

    # Check class distribution
    print(f"Class distribution: {y_attendance_train.value_counts().to_dict()}")

    attendance_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        class_weight='balanced',  # Handle imbalanced data
        random_state=42
    )
    attendance_model.fit(X_train_scaled, y_attendance_train)
    ensemble.attendance_model = attendance_model

    # Evaluate
    y_pred = attendance_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_attendance_test, y_pred)
    f1 = f1_score(y_attendance_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_attendance_test, y_pred))

    # 3. Train Risk Scoring Model (Multi-class)
    print("\n" + "="*60)
    print("3. Training Risk Scoring Model (Multi-class)")
    print("="*60)

    y_risk_train = training_df.loc[X_train.index, "label_risk_category"]
    y_risk_test = training_df.loc[X_test.index, "label_risk_category"]

    # Encode risk categories
    risk_mapping = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    y_risk_train_encoded = y_risk_train.map(risk_mapping)
    y_risk_test_encoded = y_risk_test.map(risk_mapping)

    print(f"Class distribution: {y_risk_train.value_counts().to_dict()}")

    risk_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    risk_model.fit(X_train_scaled, y_risk_train_encoded)
    ensemble.risk_model = risk_model

    # Evaluate
    y_pred = risk_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_risk_test_encoded, y_pred)
    f1 = f1_score(y_risk_test_encoded, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(
        y_risk_test_encoded,
        y_pred,
        target_names=["low", "medium", "high", "critical"]
    ))

    # Feature importance
    print("\n" + "="*60)
    print("Top 10 Important Features (Risk Model)")
    print("="*60)
    importance_dict = ensemble.get_feature_importance()
    for i, (feature, importance) in enumerate(list(importance_dict.items())[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    return ensemble


def _encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features"""
    X = X.copy()

    # Experience level
    if "experience_level" in X.columns:
        X["experience_level"] = X["experience_level"].map(
            {"new": 0, "intermediate": 1, "experienced": 2}
        ).fillna(1)

    # Assigned shift
    if "assigned_shift" in X.columns:
        X["assigned_shift"] = X["assigned_shift"].map(
            {"day": 0, "afternoon": 1, "night": 2}
        ).fillna(0)

    # Zone risk level
    if "zone_risk_level" in X.columns:
        X["zone_risk_level"] = X["zone_risk_level"].map(
            {"low": 0, "normal": 1, "high": 2, "critical": 3}
        ).fillna(1)

    # Convert boolean columns
    bool_columns = X.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        X[col] = X[col].astype(int)

    # Fill any remaining NaN with 0
    X = X.fillna(0)

    return X


async def train_and_save_models(db, model_save_path: str):
    """
    Complete training pipeline: generate data, train models, save.

    Args:
        db: MongoDB database instance
        model_save_path: Path to save trained models

    Returns:
        Trained ensemble
    """
    from .training_data_generator import generate_training_dataset

    # Generate training data
    print("Step 1: Generating training dataset from historical data...")
    training_df = await generate_training_dataset(db, lookback_days=180)

    if len(training_df) < 100:
        raise ValueError(f"Insufficient training data: {len(training_df)} examples. Need at least 100.")

    # Train models
    print("\nStep 2: Training models...")
    ensemble = train_models(training_df)

    # Save models
    print(f"\nStep 3: Saving models to {model_save_path}...")
    ensemble.save(model_save_path)

    print("\nTraining pipeline complete!")
    return ensemble
