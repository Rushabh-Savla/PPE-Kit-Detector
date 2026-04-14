"""
Core Prediction Service.

Coordinates feature extraction, model inference, and risk calculation.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import numpy as np
import os
from .feature_engineering import FeatureExtractor
from .models import PredictionModelEnsemble
from .explainability import RiskExplainer


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for MongoDB compatibility.

    Args:
        obj: Object to convert (dict, list, or value)

    Returns:
        Object with all numpy types converted to Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class PredictionService:
    """Centralized service for generating worker risk predictions"""

    def __init__(self, db):
        """
        Initialize prediction service.

        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.feature_extractor = FeatureExtractor(db)
        self.model_ensemble = PredictionModelEnsemble()
        self.explainer = RiskExplainer()

        # Load pre-trained models
        model_path = os.path.join(
            os.path.dirname(__file__),
            "trained_models",
            "ensemble_v1.pkl"
        )
        try:
            self.model_ensemble.load(model_path)
            self.models_loaded = True
        except FileNotFoundError:
            print(f"Warning: Pre-trained models not found at {model_path}")
            print("Models need to be trained first. Using rule-based fallback.")
            self.models_loaded = False

    async def predict_worker_risk(
        self,
        worker_id: str,
        prediction_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive risk prediction for a worker.

        Args:
            worker_id: Worker's MongoDB ObjectId as string
            prediction_date: Date to generate prediction for (default: now)

        Returns:
            Prediction dictionary ready for MongoDB storage
        """
        if prediction_date is None:
            prediction_date = datetime.utcnow()

        # 1. Extract features
        features = await self.feature_extractor.extract_features(
            worker_id,
            prediction_date
        )

        # 2. Run ML models (or fallback to rules if models not loaded)
        if self.models_loaded:
            predictions = self.model_ensemble.predict(features)
        else:
            predictions = self._rule_based_prediction(features)

        # 3. Calculate composite overall risk score
        overall_risk_score = self._calculate_overall_risk(predictions, features)

        # 4. Determine risk category
        risk_category = self._classify_risk_category(overall_risk_score)

        # 5. Generate explanations
        risk_factors = self.explainer.explain_risk(
            overall_risk_score,
            features,
            predictions
        )

        # 6. Determine intervention flag
        requires_intervention = (
            risk_category == "critical" or
            features.get("consecutive_absences_current", 0) >= 3 or
            features.get("violations_last_7d", 0) >= 5 or
            overall_risk_score >= 85
        )

        # 7. Build prediction document
        prediction_doc = {
            "worker_id": worker_id,
            "employee_id": features.get("employee_id"),
            "prediction_date": prediction_date,
            "prediction_period": "next_month",

            # Risk scores (0-100)
            "overall_risk_score": round(overall_risk_score, 2),
            "violation_risk_score": round(predictions["violation_risk"] * 100, 2),
            "attendance_risk_score": round(predictions["attendance_risk"] * 100, 2),
            "compliance_trend_score": round(predictions["compliance_trend"], 2),

            # Specific predictions
            "predicted_violations_count": int(predictions["predicted_violations"]),
            "predicted_absent_days": int(predictions["predicted_absences"]),
            "high_risk_ppe_items": predictions["high_risk_items"],

            # Patterns
            "attendance_pattern": predictions["attendance_pattern"],
            "consecutive_absence_risk": round(predictions["consecutive_absence_prob"] * 100, 2),
            "attendance_rate_30d": round(features.get("attendance_rate_30d", 0) * 100, 2),

            # Classification
            "risk_category": risk_category,
            "requires_intervention": requires_intervention,

            # Explainability
            "risk_factors": risk_factors,

            # Features (for audit/debugging)
            "features": features,

            # Metadata
            "model_version": self.model_ensemble.model_version,
            "confidence": round(predictions["confidence"], 3),
            "created_at": datetime.utcnow(),
            "expires_at": prediction_date + timedelta(days=7)  # 1 week validity
        }

        # Convert numpy types to Python types for MongoDB compatibility
        return convert_numpy_types(prediction_doc)

    def _calculate_overall_risk(self, predictions: Dict, features: Dict) -> float:
        """
        Combine individual predictions into overall risk score (0-100).
        Weighted combination of different risk factors.

        Args:
            predictions: Model predictions
            features: Extracted features

        Returns:
            Overall risk score (0-100)
        """
        weights = {
            "violation_risk": 0.40,      # 40% weight
            "attendance_risk": 0.30,     # 30% weight
            "compliance_trend": 0.20,    # 20% weight
            "behavioral_risk": 0.10      # 10% weight
        }

        # Normalize all to 0-100 scale
        violation_score = predictions["violation_risk"] * 100
        attendance_score = predictions["attendance_risk"] * 100

        # Compliance trend: lower compliance = higher risk
        compliance_score_current = features.get("compliance_score_current", 100)
        compliance_trend_score = max(0, 100 - compliance_score_current)

        # Behavioral risk: warnings, alerts
        warning_count = features.get("warning_count_30d", 0)
        behavioral_score = min(100, warning_count * 20)

        # Weighted sum
        overall_risk = (
            violation_score * weights["violation_risk"] +
            attendance_score * weights["attendance_risk"] +
            compliance_trend_score * weights["compliance_trend"] +
            behavioral_score * weights["behavioral_risk"]
        )

        return np.clip(overall_risk, 0, 100)

    def _classify_risk_category(self, risk_score: float) -> str:
        """
        Map risk score to category.

        Args:
            risk_score: Overall risk score (0-100)

        Returns:
            Risk category string
        """
        if risk_score < 30:
            return "low"
        elif risk_score < 60:
            return "medium"
        elif risk_score < 85:
            return "high"
        else:
            return "critical"

    def _rule_based_prediction(self, features: Dict) -> Dict[str, Any]:
        """
        Fallback rule-based prediction when ML models not loaded.

        Args:
            features: Extracted features

        Returns:
            Prediction dictionary
        """
        # Violation prediction (simple trend extrapolation)
        violations_30d = features.get("violations_last_30d", 0)
        violation_rate = features.get("violation_rate_30d", 0)
        predicted_violations = int(violations_30d * 1.1)  # 10% increase

        # Attendance risk (based on current rate)
        attendance_rate = features.get("attendance_rate_30d", 1.0)
        attendance_risk = max(0, min(1, (1 - attendance_rate) / 0.4))  # Scale to 0-1

        # High-risk PPE items
        high_risk_items = []
        total_entries = features.get("total_entries_30d", 1)
        if total_entries > 0:
            for item in ["helmet", "vest", "goggles", "gloves", "mask", "safety_shoes"]:
                violations = features.get(f"{item}_violations", 0)
                if violations / total_entries > 0.3:
                    high_risk_items.append(item)

        # Attendance pattern
        if attendance_rate >= 0.9:
            attendance_pattern = "regular"
        elif attendance_rate < 0.75:
            attendance_pattern = "declining"
        else:
            attendance_pattern = "irregular"

        # Predicted absences
        current_absences = 30 - features.get("total_entries_30d", 30)
        predicted_absences = int(current_absences * 1.1)

        return {
            "predicted_violations": predicted_violations,
            "violation_risk": violation_rate,
            "attendance_risk": attendance_risk,
            "compliance_trend": features.get("compliance_score_current", 100),
            "risk_category": "medium",  # Default
            "high_risk_items": high_risk_items,
            "attendance_pattern": attendance_pattern,
            "predicted_absences": predicted_absences,
            "consecutive_absence_prob": min(1.0, features.get("consecutive_absences_current", 0) / 5.0),
            "confidence": 0.6  # Lower confidence for rule-based
        }

    async def predict_batch(self, worker_ids: list) -> list:
        """
        Generate predictions for multiple workers.

        Args:
            worker_ids: List of worker IDs

        Returns:
            List of prediction documents
        """
        predictions = []
        for worker_id in worker_ids:
            try:
                pred = await self.predict_worker_risk(worker_id)
                predictions.append(pred)
            except Exception as e:
                print(f"Error predicting for worker {worker_id}: {e}")
                continue

        return predictions
