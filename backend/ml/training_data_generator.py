"""
Training Data Generator for ML models.

Generates supervised learning dataset from historical data using sliding window approach.
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from bson import ObjectId


async def generate_training_dataset(db, lookback_days: int = 180) -> pd.DataFrame:
    """
    Generate training dataset from historical data.

    For each worker at each week in the past:
    - Extract features at that point in time (looking back 30 days)
    - Extract target labels for next 30 days (future violations, attendance)
    - Create training example

    Args:
        db: MongoDB database instance
        lookback_days: How many days of history to use (default 180)

    Returns:
        pandas DataFrame with features + labels
    """
    from .feature_engineering import FeatureExtractor

    print(f"Generating training dataset with {lookback_days} days of history...")

    extractor = FeatureExtractor(db)
    training_data = []

    # Get all workers
    workers = await db.workers.find({"is_active": True}).to_list(length=None)
    print(f"Found {len(workers)} active workers")

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=lookback_days)

    # Valid reference date range:
    # - Must be at least 30 days after start_date (for feature extraction)
    # - Must be at least 30 days before end_date (for label extraction)
    earliest_reference = start_date + timedelta(days=30)
    latest_reference = end_date - timedelta(days=30)

    # For each worker, generate training examples using sliding window
    for worker in workers:
        worker_id = str(worker["_id"])
        print(f"Processing worker {worker.get('employee_id', worker_id)}...")

        # Calculate number of weekly samples we can take
        valid_days = (latest_reference - earliest_reference).days
        num_samples = max(1, valid_days // 7)  # Sample weekly

        for i in range(num_samples):
            # Reference point (point in time to extract features)
            # Start from earliest_reference and move forward week by week
            reference_date = earliest_reference + timedelta(weeks=i)

            # Skip if reference date is beyond latest valid date
            if reference_date > latest_reference:
                break

            # Skip if reference date is before worker creation
            worker_created = worker.get("created_at", start_date)
            if reference_date < worker_created + timedelta(days=30):
                continue

            # Extract features at reference point
            try:
                features = await extractor.extract_features(worker_id, reference_date)
            except Exception as e:
                print(f"  Error extracting features: {e}")
                continue

            # Extract target labels (30 days after reference point)
            label_start = reference_date
            label_end = reference_date + timedelta(days=30)

            try:
                labels = await _extract_labels(db, worker_id, label_start, label_end)
            except Exception as e:
                print(f"  Error extracting labels: {e}")
                continue

            # Combine features and labels
            training_example = {**features, **labels}
            training_data.append(training_example)

    print(f"Generated {len(training_data)} training examples")

    # Convert to DataFrame
    df = pd.DataFrame(training_data)

    return df


async def _extract_labels(db, worker_id: str, start_date: datetime, end_date: datetime) -> dict:
    """
    Extract target labels for a worker in a future time window.

    Args:
        db: MongoDB database
        worker_id: Worker ID
        start_date: Start of label window
        end_date: End of label window

    Returns:
        Dictionary of label values
    """
    # Count violations in next 30 days
    future_violations = await db.gate_entries.count_documents({
        "worker_id": worker_id,
        "timestamp": {"$gte": start_date, "$lte": end_date},
        "violations": {"$ne": []}
    })

    # Count entries in next 30 days
    future_entries = await db.gate_entries.count_documents({
        "worker_id": worker_id,
        "entry_type": "entry",
        "timestamp": {"$gte": start_date, "$lte": end_date}
    })

    # Calculate attendance rate (expected: 30 days)
    future_attendance_rate = future_entries / 30.0

    # Check for attendance issues (attendance < 75% OR 3+ consecutive absences)
    has_attendance_issue = future_attendance_rate < 0.75

    # Check for consecutive absences
    max_consecutive_absences = await _calculate_consecutive_absences(
        db, worker_id, start_date, end_date
    )

    if max_consecutive_absences >= 3:
        has_attendance_issue = True

    # Determine risk category based on future data
    # This is what we want the model to learn to predict
    if future_violations >= 10 or future_attendance_rate < 0.6:
        risk_category = "critical"
    elif future_violations >= 5 or future_attendance_rate < 0.75:
        risk_category = "high"
    elif future_violations >= 2 or future_attendance_rate < 0.85:
        risk_category = "medium"
    else:
        risk_category = "low"

    return {
        "label_violations_next_30d": future_violations,
        "label_attendance_rate_next_30d": round(future_attendance_rate, 3),
        "label_has_attendance_issue": 1 if has_attendance_issue else 0,
        "label_risk_category": risk_category,
        "label_max_consecutive_absences": max_consecutive_absences
    }


async def _calculate_consecutive_absences(
    db, worker_id: str, start_date: datetime, end_date: datetime
) -> int:
    """Calculate maximum consecutive absences in a period"""

    # Get all entries in the period
    entries = await db.gate_entries.find({
        "worker_id": worker_id,
        "entry_type": "entry",
        "timestamp": {"$gte": start_date, "$lte": end_date}
    }).sort("timestamp", 1).to_list(length=None)

    max_consecutive = 0
    current_consecutive = 0

    # Check each day
    num_days = (end_date - start_date).days
    for i in range(num_days):
        day = start_date + timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        has_entry = any(day_start <= e["timestamp"] < day_end for e in entries)

        if not has_entry:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return max_consecutive


def save_training_data(df: pd.DataFrame, filepath: str):
    """
    Save training dataset to CSV.

    Args:
        df: Training DataFrame
        filepath: Path to save CSV
    """
    df.to_csv(filepath, index=False)
    print(f"Training data saved to {filepath}")


def load_training_data(filepath: str) -> pd.DataFrame:
    """
    Load training dataset from CSV.

    Args:
        filepath: Path to CSV file

    Returns:
        pandas DataFrame
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} training examples from {filepath}")
    return df
