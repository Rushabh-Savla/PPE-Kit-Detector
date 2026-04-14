"""
Background Scheduler for ML tasks.

Scheduled jobs:
- Daily prediction generation (2 AM)
- Weekly model retraining (Sunday 3 AM)
- Cleanup expired predictions (1 AM)
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime, timedelta
from bson import ObjectId


scheduler = AsyncIOScheduler()


def start_scheduler(db):
    """
    Start the background scheduler with all jobs.

    Args:
        db: MongoDB database instance
    """
    print("Starting ML prediction scheduler...")

    # Store db reference for job functions
    scheduler.db = db

    # Daily prediction generation at 2 AM
    @scheduler.scheduled_job('cron', hour=2, minute=0, id='daily_predictions')
    async def daily_prediction_job():
        """Generate predictions for all workers daily"""
        print(f"[{datetime.utcnow()}] Starting daily prediction generation...")

        from .prediction_service import PredictionService

        try:
            service = PredictionService(db)

            # Get all active workers
            workers = await db.workers.find({"is_active": True}).to_list(length=None)
            print(f"Found {len(workers)} active workers")

            success_count = 0
            error_count = 0

            for worker in workers:
                try:
                    # Generate prediction
                    prediction = await service.predict_worker_risk(str(worker["_id"]))

                    # Save to database
                    await db.predictions.insert_one(prediction)

                    # Create alert if critical risk
                    if prediction["risk_category"] == "critical" or prediction["requires_intervention"]:
                        await _create_risk_alert(db, worker, prediction)

                    success_count += 1

                except Exception as e:
                    print(f"Error predicting for worker {worker.get('employee_id')}: {e}")
                    error_count += 1

            print(f"Daily predictions complete: {success_count} success, {error_count} errors")

        except Exception as e:
            print(f"Error in daily prediction job: {e}")

    # Weekly model retraining on Sunday at 3 AM
    @scheduler.scheduled_job('cron', day_of_week='sun', hour=3, minute=0, id='weekly_retraining')
    async def weekly_retraining_job():
        """Retrain models on latest data weekly"""
        print(f"[{datetime.utcnow()}] Starting weekly model retraining...")

        from .training import train_and_save_models
        import os

        try:
            model_save_path = os.path.join(
                os.path.dirname(__file__),
                "trained_models",
                "ensemble_v1.pkl"
            )

            # Train and save models
            await train_and_save_models(db, model_save_path)

            print("Weekly model retraining complete")

        except Exception as e:
            print(f"Error in weekly retraining job: {e}")

    # Cleanup expired predictions daily at 1 AM
    @scheduler.scheduled_job('cron', hour=1, minute=0, id='cleanup_predictions')
    async def cleanup_expired_predictions():
        """Remove old predictions to save storage"""
        print(f"[{datetime.utcnow()}] Cleaning up expired predictions...")

        try:
            result = await db.predictions.delete_many({
                "expires_at": {"$lt": datetime.utcnow()}
            })

            print(f"Cleaned up {result.deleted_count} expired predictions")

        except Exception as e:
            print(f"Error in cleanup job: {e}")

    # Start the scheduler
    scheduler.start()
    print("ML prediction scheduler started successfully")
    print("Scheduled jobs:")
    print("  - Daily predictions: 2:00 AM")
    print("  - Weekly retraining: Sunday 3:00 AM")
    print("  - Cleanup expired: 1:00 AM")


async def _create_risk_alert(db, worker: dict, prediction: dict):
    """
    Create an alert for high-risk workers.

    Args:
        db: MongoDB database
        worker: Worker document
        prediction: Prediction document
    """
    # Check if alert already exists for this worker today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    existing_alert = await db.alerts.find_one({
        "worker_id": str(worker["_id"]),
        "alert_type": "worker_risk_prediction",
        "created_at": {"$gte": today_start}
    })

    if existing_alert:
        return  # Don't create duplicate alert

    # Build alert message
    risk_score = prediction["overall_risk_score"]
    risk_category = prediction["risk_category"]

    # Get primary risk factor
    risk_factors = prediction.get("risk_factors", [])
    primary_issue = risk_factors[0]["description"] if risk_factors else "Multiple risk indicators"

    alert_doc = {
        "alert_type": "worker_risk_prediction",
        "severity": "critical" if risk_category == "critical" else "high",
        "status": "active",
        "message": f"Worker {worker['name']} ({worker['employee_id']}) flagged as {risk_category.upper()} risk (score: {risk_score:.0f}/100). {primary_issue}",
        "mine_id": worker.get("mine_id"),
        "zone_id": worker.get("zone_id"),
        "worker_id": str(worker["_id"]),
        "metadata": {
            "risk_score": risk_score,
            "risk_category": risk_category,
            "predicted_violations": prediction["predicted_violations_count"],
            "attendance_rate": prediction["attendance_rate_30d"],
            "primary_issue": primary_issue,
            "requires_intervention": prediction["requires_intervention"]
        },
        "created_at": datetime.utcnow(),
        "created_by": "system_ml_predictions"
    }

    await db.alerts.insert_one(alert_doc)
    print(f"Created {risk_category} risk alert for worker {worker['employee_id']}")


def stop_scheduler():
    """Stop the scheduler"""
    if scheduler.running:
        scheduler.shutdown()
        print("ML prediction scheduler stopped")
