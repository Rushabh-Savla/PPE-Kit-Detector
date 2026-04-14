"""
ML Model Ensemble for worker risk prediction.

Ensemble of three specialized models:
1. Violation Prediction Model (Regression) - predicts violation count
2. Attendance Classification Model (Binary) - predicts attendance issues
3. Risk Scoring Model (Multi-class) - predicts overall risk category
"""

from typing import Dict, Any, Optional
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class PredictionModelEnsemble:
    """Ensemble of specialized models for worker risk prediction"""

    def __init__(self):
        self.violation_model: Optional[GradientBoostingRegressor] = None
        self.attendance_model: Optional[RandomForestClassifier] = None
        self.risk_model: Optional[GradientBoostingClassifier] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.model_version = "v1.0.0"
        self.feature_names = []

    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert feature dictionary to numpy array for model input.

        Args:
            features: Dictionary of features

        Returns:
            Numpy array of feature values
        """
        # Define feature order (must match training)
        if not self.feature_names:
            # Set default feature names
            self.feature_names = self._get_feature_names()

        feature_values = []
        for fname in self.feature_names:
            value = features.get(fname, 0)

            # Handle categorical features
            if fname == "experience_level":
                # One-hot encode
                value = {"new": 0, "intermediate": 1, "experienced": 2}.get(value, 1)
            elif fname == "assigned_shift":
                value = {"day": 0, "afternoon": 1, "night": 2}.get(value, 0)
            elif fname == "zone_risk_level":
                value = {"low": 0, "normal": 1, "high": 2, "critical": 3}.get(value, 1)
            elif isinstance(value, bool):
                value = 1 if value else 0

            feature_values.append(float(value))

        return np.array(feature_values).reshape(1, -1)

    def _get_feature_names(self) -> list:
        """Get ordered list of feature names"""
        return [
            # Violation features
            "violations_last_7d",
            "violations_last_30d",
            "violation_rate_7d",
            "violation_rate_30d",
            "helmet_violations",
            "vest_violations",
            "goggles_violations",
            "gloves_violations",
            "mask_violations",
            "safety_shoes_violations",
            "days_since_last_violation",
            "violation_trend",
            # Attendance features
            "attendance_rate_30d",
            "attendance_rate_7d",
            "consecutive_absences_current",
            "max_consecutive_absences_30d",
            "attendance_variability",
            "shift_consistency",
            # Compliance features
            "compliance_score_current",
            "total_violations_lifetime",
            "badges_count",
            "has_safety_star_badge",
            "has_perfect_record_badge",
            # Behavioral features
            "warning_count_30d",
            "days_since_last_warning",
            "alert_count_related",
            # Temporal features
            "tenure_days",
            "experience_level",
            "assigned_shift",
            # Contextual features
            "zone_risk_level",
            "mine_compliance_rate"
        ]

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate predictions for a single worker.

        Args:
            features: Feature dictionary from FeatureExtractor

        Returns:
            Dictionary with predictions
        """
        if not all([self.violation_model, self.attendance_model, self.risk_model]):
            raise ValueError("Models not trained or loaded")

        # Prepare features
        X = self._prepare_features(features)

        # Scale features if scaler exists
        if self.feature_scaler:
            X_scaled = self.feature_scaler.transform(X)
        else:
            X_scaled = X

        # Violation prediction (regression)
        predicted_violations = self.violation_model.predict(X_scaled)[0]
        predicted_violations = max(0, int(round(predicted_violations)))

        # Attendance prediction (binary classification)
        attendance_risk_prob = self.attendance_model.predict_proba(X_scaled)[0][1]

        # Risk category prediction (multi-class)
        risk_probs = self.risk_model.predict_proba(X_scaled)[0]
        risk_categories = ["low", "medium", "high", "critical"]
        predicted_risk_category = risk_categories[np.argmax(risk_probs)]

        # Determine high-risk PPE items (items with >30% violation rate)
        high_risk_items = []
        total_entries = features.get("total_entries_30d", 1)
        if total_entries > 0:
            for item in ["helmet", "vest", "goggles", "gloves", "mask", "safety_shoes"]:
                violations = features.get(f"{item}_violations", 0)
                if violations / total_entries > 0.3:
                    high_risk_items.append(item)

        # Determine attendance pattern
        attendance_rate = features.get("attendance_rate_30d", 1.0)
        attendance_variability = features.get("attendance_variability", 0)
        if attendance_rate >= 0.9 and attendance_variability < 0.3:
            attendance_pattern = "regular"
        elif attendance_rate < 0.75:
            attendance_pattern = "declining"
        else:
            attendance_pattern = "irregular"

        # Predicted absences (based on attendance rate)
        current_absences_30d = 30 - features.get("total_entries_30d", 30)
        predicted_absences = int(current_absences_30d * 1.1)  # 10% increase trend

        # Consecutive absence risk
        consecutive_absence_risk = attendance_risk_prob * 100

        # Compliance trend (compare current to expected based on violation rate)
        violation_rate_30d = features.get("violation_rate_30d", 0)
        compliance_trend = 100 - (violation_rate_30d * 200)

        # Model confidence (average of prediction probabilities)
        confidence = (
            (1 - abs(predicted_violations - features.get("violations_last_30d", 0)) / 10) * 0.3 +
            max(risk_probs) * 0.4 +
            (1 - attendance_risk_prob if attendance_risk_prob < 0.5 else attendance_risk_prob) * 0.3
        )

        return {
            "predicted_violations": predicted_violations,
            "violation_risk": violation_rate_30d,  # 0-1 scale
            "attendance_risk": attendance_risk_prob,  # 0-1 scale
            "compliance_trend": compliance_trend,  # 0-100 scale
            "risk_category": predicted_risk_category,
            "high_risk_items": high_risk_items,
            "attendance_pattern": attendance_pattern,
            "predicted_absences": predicted_absences,
            "consecutive_absence_prob": consecutive_absence_risk / 100,
            "confidence": min(1.0, max(0.0, confidence))
        }

    def predict_batch(self, features_list: list) -> list:
        """
        Generate predictions for multiple workers (batch).

        Args:
            features_list: List of feature dictionaries

        Returns:
            List of prediction dictionaries
        """
        predictions = []
        for features in features_list:
            try:
                pred = self.predict(features)
                predictions.append(pred)
            except Exception as e:
                print(f"Error predicting for worker: {e}")
                predictions.append(None)

        return predictions

    def save(self, path: str):
        """
        Save models to disk using joblib.

        Args:
            path: File path to save models
        """
        model_data = {
            "violation_model": self.violation_model,
            "attendance_model": self.attendance_model,
            "risk_model": self.risk_model,
            "feature_scaler": self.feature_scaler,
            "feature_names": self.feature_names,
            "model_version": self.model_version
        }
        joblib.dump(model_data, path)
        print(f"Models saved to {path}")

    def load(self, path: str):
        """
        Load pre-trained models from disk.

        Args:
            path: File path to load models from
        """
        try:
            model_data = joblib.load(path)
            self.violation_model = model_data["violation_model"]
            self.attendance_model = model_data["attendance_model"]
            self.risk_model = model_data["risk_model"]
            self.feature_scaler = model_data.get("feature_scaler")
            self.feature_names = model_data.get("feature_names", self._get_feature_names())
            self.model_version = model_data.get("model_version", "v1.0.0")
            print(f"Models loaded from {path}")
        except FileNotFoundError:
            print(f"Model file not found at {path}. Models need to be trained first.")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from models for interpretability.

        Returns:
            Dictionary of feature_name -> importance score
        """
        if not self.risk_model or not self.feature_names:
            return {}

        # Get feature importance from risk model (main model)
        importances = self.risk_model.feature_importances_

        importance_dict = {}
        for fname, importance in zip(self.feature_names, importances):
            importance_dict[fname] = round(float(importance), 4)

        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return sorted_importance
