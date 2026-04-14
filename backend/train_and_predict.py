"""
Train ML models and generate predictions for all workers.
Run this after seeding historical data.
"""

import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGO_ATLAS_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017/sih_safety_system"))


async def main():
    """Train models and generate predictions"""
    print("="*60)
    print("ML Training & Prediction Generation")
    print("="*60)

    # Connect to database
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client.get_default_database()

    try:
        # Step 1: Train models
        print("\n[1/2] Training ML models...")
        from ml.training import train_and_save_models

        model_path = "ml/trained_models/ensemble_v1.pkl"
        await train_and_save_models(db, model_path)
        print("✓ Models trained and saved successfully!")

        # Step 2: Generate predictions for all workers
        print("\n[2/2] Generating predictions for all workers...")
        from ml.prediction_service import PredictionService
        from datetime import datetime

        prediction_service = PredictionService(db)

        # Get all active workers
        workers = await db.workers.find({"is_active": True}).to_list(length=None)
        print(f"Found {len(workers)} active workers")

        success_count = 0
        error_count = 0

        for i, worker in enumerate(workers, 1):
            worker_id = str(worker["_id"])
            try:
                prediction = await prediction_service.predict_worker_risk(worker_id)

                # Save prediction to database
                await db.predictions.insert_one(prediction)

                success_count += 1
                if i % 10 == 0:
                    print(f"  Processed {i}/{len(workers)} workers...")

            except Exception as e:
                error_count += 1
                print(f"  Error predicting for worker {worker_id}: {e}")

        print(f"\n✓ Generated predictions for {success_count} workers")
        if error_count > 0:
            print(f"  ⚠ Failed for {error_count} workers")

        # Summary stats
        at_risk_count = await db.predictions.count_documents({
            "risk_category": {"$in": ["medium", "high", "critical"]},
            "expires_at": {"$gt": datetime.utcnow()}
        })
        critical_count = await db.predictions.count_documents({
            "risk_category": "critical",
            "expires_at": {"$gt": datetime.utcnow()}
        })

        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"Total predictions: {success_count}")
        print(f"At-risk workers: {at_risk_count}")
        print(f"Critical risk: {critical_count}")
        print("\n✓ All done! You can now view predictions in the UI.")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
