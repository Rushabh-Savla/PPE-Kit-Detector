"""
Feature extraction for worker risk prediction.

Extracts 40+ features from MongoDB collections:
- Violation history from gate_entries
- Attendance patterns from gate_entries
- Compliance scores from workers
- Behavioral features from warnings
- Temporal and contextual features
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from bson import ObjectId
import numpy as np


class FeatureExtractor:
    """Extract ML features from MongoDB collections"""

    def __init__(self, db):
        """
        Initialize feature extractor with database connection.

        Args:
            db: MongoDB database instance
        """
        self.db = db

    async def extract_features(
        self,
        worker_id: str,
        reference_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Extract all features for a worker at a specific point in time.

        Args:
            worker_id: Worker's MongoDB ObjectId as string
            reference_date: Point in time to extract features (default: now)

        Returns:
            Dictionary of feature_name -> value
        """
        if reference_date is None:
            reference_date = datetime.utcnow()

        # Get worker document
        worker = await self.db.workers.find_one({"_id": ObjectId(worker_id)})
        if not worker:
            raise ValueError(f"Worker {worker_id} not found")

        # Initialize features dictionary
        features = {
            "worker_id": worker_id,
            "employee_id": worker.get("employee_id", ""),
        }

        # Extract features from different sources
        violation_features = await self._extract_violation_features(worker_id, reference_date)
        attendance_features = await self._extract_attendance_features(worker_id, reference_date, worker)
        compliance_features = await self._extract_compliance_features(worker)
        behavioral_features = await self._extract_behavioral_features(worker_id, reference_date)
        temporal_features = await self._extract_temporal_features(worker_id, reference_date, worker)
        contextual_features = await self._extract_contextual_features(worker, reference_date)

        # Combine all features
        features.update(violation_features)
        features.update(attendance_features)
        features.update(compliance_features)
        features.update(behavioral_features)
        features.update(temporal_features)
        features.update(contextual_features)

        return features

    async def _extract_violation_features(
        self,
        worker_id: str,
        reference_date: datetime
    ) -> Dict[str, Any]:
        """Extract violation history features from gate_entries"""

        # Time windows
        date_7d_ago = reference_date - timedelta(days=7)
        date_14d_ago = reference_date - timedelta(days=14)
        date_30d_ago = reference_date - timedelta(days=30)

        # Count violations in different time windows
        violations_7d = await self.db.gate_entries.count_documents({
            "worker_id": worker_id,
            "timestamp": {"$gte": date_7d_ago, "$lte": reference_date},
            "violations": {"$ne": []}
        })

        violations_14d = await self.db.gate_entries.count_documents({
            "worker_id": worker_id,
            "timestamp": {"$gte": date_14d_ago, "$lte": reference_date},
            "violations": {"$ne": []}
        })

        violations_30d = await self.db.gate_entries.count_documents({
            "worker_id": worker_id,
            "timestamp": {"$gte": date_30d_ago, "$lte": reference_date},
            "violations": {"$ne": []}
        })

        # Count total entries (for rate calculation)
        entries_7d = await self.db.gate_entries.count_documents({
            "worker_id": worker_id,
            "entry_type": "entry",
            "timestamp": {"$gte": date_7d_ago, "$lte": reference_date}
        })

        entries_30d = await self.db.gate_entries.count_documents({
            "worker_id": worker_id,
            "entry_type": "entry",
            "timestamp": {"$gte": date_30d_ago, "$lte": reference_date}
        })

        # Calculate violation rates
        violation_rate_7d = violations_7d / max(entries_7d, 1)
        violation_rate_30d = violations_30d / max(entries_30d, 1)

        # Get per-PPE-item violation counts
        ppe_violations = {"helmet": 0, "vest": 0, "goggles": 0,
                         "gloves": 0, "mask": 0, "safety_shoes": 0}

        violation_entries = await self.db.gate_entries.find({
            "worker_id": worker_id,
            "timestamp": {"$gte": date_30d_ago, "$lte": reference_date},
            "violations": {"$ne": []}
        }).to_list(length=None)

        for entry in violation_entries:
            ppe_status = entry.get("ppe_status", {})
            for item, status in ppe_status.items():
                if not status and item in ppe_violations:
                    ppe_violations[item] += 1

        # Get last violation date
        last_violation_entry = await self.db.gate_entries.find_one(
            {
                "worker_id": worker_id,
                "timestamp": {"$lte": reference_date},
                "violations": {"$ne": []}
            },
            sort=[("timestamp", -1)]
        )

        days_since_last_violation = 999  # Default for no violations
        if last_violation_entry:
            delta = reference_date - last_violation_entry["timestamp"]
            days_since_last_violation = delta.days

        # Violation trend (comparing last 7d to previous 7d)
        date_14d_7d_ago = reference_date - timedelta(days=14)
        violations_prev_7d = await self.db.gate_entries.count_documents({
            "worker_id": worker_id,
            "timestamp": {"$gte": date_14d_7d_ago, "$lt": date_7d_ago},
            "violations": {"$ne": []}
        })

        entries_prev_7d = await self.db.gate_entries.count_documents({
            "worker_id": worker_id,
            "entry_type": "entry",
            "timestamp": {"$gte": date_14d_7d_ago, "$lt": date_7d_ago}
        })

        violation_rate_prev_7d = violations_prev_7d / max(entries_prev_7d, 1)
        violation_trend = violation_rate_7d - violation_rate_prev_7d

        return {
            "violations_last_7d": violations_7d,
            "violations_last_14d": violations_14d,
            "violations_last_30d": violations_30d,
            "violation_rate_7d": round(violation_rate_7d, 3),
            "violation_rate_30d": round(violation_rate_30d, 3),
            "helmet_violations": ppe_violations["helmet"],
            "vest_violations": ppe_violations["vest"],
            "goggles_violations": ppe_violations["goggles"],
            "gloves_violations": ppe_violations["gloves"],
            "mask_violations": ppe_violations["mask"],
            "safety_shoes_violations": ppe_violations["safety_shoes"],
            "days_since_last_violation": days_since_last_violation,
            "violation_trend": round(violation_trend, 3)
        }

    async def _extract_attendance_features(
        self,
        worker_id: str,
        reference_date: datetime,
        worker: Dict
    ) -> Dict[str, Any]:
        """Extract attendance patterns from gate_entries"""

        date_7d_ago = reference_date - timedelta(days=7)
        date_30d_ago = reference_date - timedelta(days=30)

        # Get all entries in last 30 days
        entries = await self.db.gate_entries.find({
            "worker_id": worker_id,
            "entry_type": "entry",
            "timestamp": {"$gte": date_30d_ago, "$lte": reference_date}
        }).sort("timestamp", 1).to_list(length=None)

        entries_7d = [e for e in entries if e["timestamp"] >= date_7d_ago]

        # Calculate expected entries based on assigned shift
        assigned_shift = worker.get("assigned_shift", "day")
        # Assuming 1 entry per day (could be 0 or 1)
        expected_entries_30d = 30  # 30 days
        expected_entries_7d = 7   # 7 days

        total_entries_30d = len(entries)
        total_entries_7d = len(entries_7d)

        attendance_rate_30d = total_entries_30d / expected_entries_30d
        attendance_rate_7d = total_entries_7d / expected_entries_7d

        # Calculate consecutive absences
        consecutive_absences = 0
        max_consecutive_absences = 0
        current_consecutive = 0

        # Check each day in last 30 days
        for i in range(30):
            day = reference_date - timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            has_entry = any(day_start <= e["timestamp"] < day_end for e in entries)

            if not has_entry:
                current_consecutive += 1
                max_consecutive_absences = max(max_consecutive_absences, current_consecutive)
            else:
                if i == 0:  # Most recent day
                    consecutive_absences = 0
                current_consecutive = 0

        # If counting from most recent day
        for i in range(30):
            day = reference_date - timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            has_entry = any(day_start <= e["timestamp"] < day_end for e in entries)

            if not has_entry:
                consecutive_absences += 1
            else:
                break

        # Attendance variability (standard deviation)
        # Count entries per day
        daily_entries = {}
        for entry in entries:
            day_key = entry["timestamp"].date()
            daily_entries[day_key] = daily_entries.get(day_key, 0) + 1

        if daily_entries:
            attendance_variability = np.std(list(daily_entries.values()))
        else:
            attendance_variability = 0.0

        # Shift consistency (% of entries in assigned shift)
        if entries:
            assigned_shift_count = sum(1 for e in entries if e.get("shift") == assigned_shift)
            shift_consistency = assigned_shift_count / len(entries)
        else:
            shift_consistency = 1.0

        return {
            "total_entries_30d": total_entries_30d,
            "total_entries_7d": total_entries_7d,
            "expected_entries_30d": expected_entries_30d,
            "attendance_rate_30d": round(attendance_rate_30d, 3),
            "attendance_rate_7d": round(attendance_rate_7d, 3),
            "consecutive_absences_current": consecutive_absences,
            "max_consecutive_absences_30d": max_consecutive_absences,
            "attendance_variability": round(attendance_variability, 3),
            "shift_consistency": round(shift_consistency, 3)
        }

    async def _extract_compliance_features(self, worker: Dict) -> Dict[str, Any]:
        """Extract compliance score features from worker document"""

        compliance_score = worker.get("compliance_score", 100.0)
        total_violations = worker.get("total_violations", 0)
        badges = worker.get("badges", [])

        return {
            "compliance_score_current": round(compliance_score, 1),
            "total_violations_lifetime": total_violations,
            "badges_count": len(badges),
            "has_safety_star_badge": "safety_star" in badges,
            "has_perfect_record_badge": "perfect_record" in badges
        }

    async def _extract_behavioral_features(
        self,
        worker_id: str,
        reference_date: datetime
    ) -> Dict[str, Any]:
        """Extract behavioral features from warnings and alerts"""

        date_30d_ago = reference_date - timedelta(days=30)

        # Count warnings in last 30 days
        warning_count_30d = await self.db.warnings.count_documents({
            "worker_id": worker_id,
            "issued_at": {"$gte": date_30d_ago, "$lte": reference_date}
        })

        # Get last warning date
        last_warning = await self.db.warnings.find_one(
            {
                "worker_id": worker_id,
                "issued_at": {"$lte": reference_date}
            },
            sort=[("issued_at", -1)]
        )

        days_since_last_warning = 999
        if last_warning:
            delta = reference_date - last_warning["issued_at"]
            days_since_last_warning = delta.days

        # Count alerts related to this worker
        alert_count = await self.db.alerts.count_documents({
            "worker_id": worker_id,
            "created_at": {"$gte": date_30d_ago, "$lte": reference_date}
        })

        return {
            "warning_count_30d": warning_count_30d,
            "days_since_last_warning": days_since_last_warning,
            "alert_count_related": alert_count
        }

    async def _extract_temporal_features(
        self,
        worker_id: str,
        reference_date: datetime,
        worker: Dict
    ) -> Dict[str, Any]:
        """Extract temporal features"""

        # Calculate tenure (days since worker created)
        created_at = worker.get("created_at", reference_date)
        tenure_days = (reference_date - created_at).days

        # Classify experience level
        if tenure_days < 30:
            experience_level = "new"
        elif tenure_days < 180:
            experience_level = "intermediate"
        else:
            experience_level = "experienced"

        # Get assigned shift
        assigned_shift = worker.get("assigned_shift", "day")

        return {
            "tenure_days": tenure_days,
            "experience_level": experience_level,
            "assigned_shift": assigned_shift
        }

    async def _extract_contextual_features(
        self,
        worker: Dict,
        reference_date: datetime
    ) -> Dict[str, Any]:
        """Extract contextual features (zone/mine averages)"""

        mine_id = worker.get("mine_id")
        zone_id = worker.get("zone_id")

        # Get zone risk level
        zone_risk_level = "normal"
        if zone_id:
            zone = await self.db.zones.find_one({"_id": zone_id})
            if zone:
                zone_risk_level = zone.get("risk_level", "normal")

        # Calculate mine-wide compliance rate (average)
        date_30d_ago = reference_date - timedelta(days=30)

        pipeline = [
            {
                "$match": {
                    "mine_id": mine_id,
                    "is_active": True
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_compliance": {"$avg": "$compliance_score"}
                }
            }
        ]

        result = await self.db.workers.aggregate(pipeline).to_list(length=1)
        mine_compliance_rate = result[0]["avg_compliance"] if result else 80.0

        return {
            "zone_risk_level": zone_risk_level,
            "mine_compliance_rate": round(mine_compliance_rate, 1)
        }

    async def extract_features_batch(
        self,
        worker_ids: list,
        reference_date: Optional[datetime] = None
    ) -> list:
        """
        Extract features for multiple workers (batch processing).

        Args:
            worker_ids: List of worker MongoDB ObjectId strings
            reference_date: Point in time to extract features

        Returns:
            List of feature dictionaries
        """
        features_list = []
        for worker_id in worker_ids:
            try:
                features = await self.extract_features(worker_id, reference_date)
                features_list.append(features)
            except Exception as e:
                print(f"Error extracting features for worker {worker_id}: {e}")
                continue

        return features_list
