"""
Generate predictions using rule-based fallback (no ML models needed).
This provides immediate predictions while ML models are being developed.
"""

import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

MONGODB_URI = os.getenv("MONGO_ATLAS_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017/sih_safety_system"))


async def main():
    """Generate predictions using rule-based approach"""
    print("=" * 60)
    print("Generating Predictions (Rule-Based)")
    print("=" * 60)

    # Connect to database
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client.get_default_database()

    try:
        from ml.prediction_service import PredictionService

        prediction_service = PredictionService(db)

        # Get all active workers
        workers = await db.workers.find({"is_active": True}).to_list(length=None)
        print(f"\nFound {len(workers)} active workers")
        print("Generating predictions (this may take a minute)...\n")

        success_count = 0
        error_count = 0

        for i, worker in enumerate(workers, 1):
            worker_id = str(worker["_id"])
            try:
                # This will use rule-based fallback since models aren't loaded
                prediction = await prediction_service.predict_worker_risk(worker_id)

                # Save prediction to database
                await db.predictions.insert_one(prediction)

                success_count += 1
                if i % 10 == 0:
                    print(f"Processed {i}/{len(workers)} workers...")

            except Exception as e:
                error_count += 1
                print(f"Error for worker {worker_id}: {e}")

        print(f"\nGenerated {success_count} predictions")
        if error_count > 0:
            print(f"Failed for {error_count} workers")

        # Summary stats
        at_risk_count = await db.predictions.count_documents({
            "risk_category": {"$in": ["medium", "high", "critical"]},
            "expires_at": {"$gt": datetime.utcnow()}
        })
        critical_count = await db.predictions.count_documents({
            "risk_category": "critical",
            "expires_at": {"$gt": datetime.utcnow()}
        })
        high_count = await db.predictions.count_documents({
            "risk_category": "high",
            "expires_at": {"$gt": datetime.utcnow()}
        })
        medium_count = await db.predictions.count_documents({
            "risk_category": "medium",
            "expires_at": {"$gt": datetime.utcnow()}
        })

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total predictions: {success_count}")
        print(f"At-risk workers: {at_risk_count}")
        print(f"  - Medium risk: {medium_count}")
        print(f"  - High risk: {high_count}")
        print(f"  - Critical risk: {critical_count}")
        print("\nAll done! You can now view predictions in the UI.")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
