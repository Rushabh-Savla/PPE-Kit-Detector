"""
Seed script to populate MongoDB with health monitoring data.
Run this script to generate sample SpO2, blood pressure, and heart rate data for workers.
"""

import os
import random
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_ATLAS_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017/sih_safety_system"))

def get_health_status(spo2: float, systolic: int, diastolic: int, heart_rate: int) -> tuple[str, list[str]]:
    """Determine health status and alerts based on vital signs."""
    alerts = []
    status = "normal"

    # SpO2 checks (normal: 95-100%)
    if spo2 < 90:
        alerts.append(f"Critical: SpO2 level dangerously low ({spo2}%)")
        status = "critical"
    elif spo2 < 94:
        alerts.append(f"Warning: SpO2 below normal ({spo2}%)")
        if status != "critical":
            status = "warning"

    # Blood pressure checks (normal: 90-120 / 60-80)
    if systolic > 180 or diastolic > 120:
        alerts.append(f"Critical: Hypertensive crisis ({systolic}/{diastolic} mmHg)")
        status = "critical"
    elif systolic > 140 or diastolic > 90:
        alerts.append(f"Warning: High blood pressure ({systolic}/{diastolic} mmHg)")
        if status != "critical":
            status = "warning"
    elif systolic < 90 or diastolic < 60:
        alerts.append(f"Warning: Low blood pressure ({systolic}/{diastolic} mmHg)")
        if status != "critical":
            status = "warning"

    # Heart rate checks (normal: 60-100 bpm)
    if heart_rate > 150:
        alerts.append(f"Critical: Heart rate dangerously high ({heart_rate} bpm)")
        status = "critical"
    elif heart_rate > 100:
        alerts.append(f"Warning: Elevated heart rate ({heart_rate} bpm)")
        if status != "critical":
            status = "warning"
    elif heart_rate < 50:
        alerts.append(f"Warning: Low heart rate ({heart_rate} bpm)")
        if status != "critical":
            status = "warning"

    return status, alerts


def generate_health_reading(worker: dict, mine: dict, timestamp: datetime, worker_profile: str = "normal") -> dict:
    """Generate a single health reading for a worker."""

    # Base values based on worker profile
    if worker_profile == "healthy":
        # Healthy worker - consistently good readings
        spo2 = round(random.uniform(96, 99), 1)
        systolic = random.randint(110, 125)
        diastolic = random.randint(70, 82)
        heart_rate = random.randint(65, 85)
    elif worker_profile == "at_risk":
        # At-risk worker - occasional concerning readings
        spo2 = round(random.uniform(92, 97), 1)
        systolic = random.randint(125, 145)
        diastolic = random.randint(78, 95)
        heart_rate = random.randint(75, 105)
    elif worker_profile == "critical":
        # Critical worker - frequently concerning readings
        spo2 = round(random.uniform(88, 94), 1)
        systolic = random.randint(140, 165)
        diastolic = random.randint(88, 105)
        heart_rate = random.randint(90, 120)
    else:
        # Normal worker - mostly good with occasional minor variations
        spo2 = round(random.uniform(94, 99), 1)
        systolic = random.randint(115, 135)
        diastolic = random.randint(72, 88)
        heart_rate = random.randint(68, 95)

    # Add some randomness for realism
    if random.random() < 0.05:  # 5% chance of anomaly
        anomaly_type = random.choice(["spo2", "bp", "hr"])
        if anomaly_type == "spo2":
            spo2 = round(random.uniform(89, 93), 1)
        elif anomaly_type == "bp":
            systolic = random.randint(145, 160)
            diastolic = random.randint(92, 100)
        else:
            heart_rate = random.randint(105, 125)

    status, alerts = get_health_status(spo2, systolic, diastolic, heart_rate)

    # Body temperature (normal: 36.1-37.2°C)
    body_temp = round(random.uniform(36.2, 37.1), 1)
    if random.random() < 0.03:  # 3% chance of fever
        body_temp = round(random.uniform(37.5, 38.5), 1)
        alerts.append(f"Warning: Elevated body temperature ({body_temp}°C)")
        if status == "normal":
            status = "warning"

    return {
        "_id": ObjectId(),
        "worker_id": str(worker["_id"]),
        "worker_name": worker.get("name", "Unknown"),
        "employee_id": worker.get("employee_id", ""),
        "mine_id": str(mine["_id"]),
        "mine_name": mine.get("name", "Unknown Mine"),
        "spo2": spo2,
        "systolic_bp": systolic,
        "diastolic_bp": diastolic,
        "heart_rate": heart_rate,
        "body_temperature": body_temp,
        "timestamp": timestamp,
        "sensor_id": f"HELMET-{worker.get('employee_id', 'UNKNOWN')[-4:]}",
        "status": status,
        "alerts": alerts,
        "created_at": datetime.utcnow()
    }


def seed_health_data():
    """Main function to seed health data into MongoDB."""
    print(f"Connecting to MongoDB: {MONGO_URI[:50]}...")

    client = MongoClient(MONGO_URI)

    # Get database name from URI or use default
    if "mongodb+srv" in MONGO_URI or "mongodb://" in MONGO_URI:
        db_name = MONGO_URI.split("/")[-1].split("?")[0]
        if not db_name or db_name == "":
            db_name = "sih"
    else:
        db_name = "sih_safety_system"

    db = client[db_name]
    print(f"Using database: {db_name}")

    # Get existing workers and mines
    workers = list(db.workers.find({"is_active": True}))
    mines = list(db.mines.find({"is_active": True}))

    print(f"Found {len(workers)} active workers and {len(mines)} active mines")

    if not workers:
        print("No workers found. Creating sample workers first...")
        # Create sample workers if none exist
        if not mines:
            # Create a sample mine first
            mine_id = db.mines.insert_one({
                "name": "Alpha Mine",
                "location": "Sector A, Industrial Zone",
                "description": "Main production mine",
                "is_active": True,
                "created_at": datetime.utcnow(),
                "zones": [],
                "gates": []
            }).inserted_id
            mines = [{"_id": mine_id, "name": "Alpha Mine"}]
            print("Created sample mine: Alpha Mine")

        # Create sample workers
        sample_workers = [
            {"employee_id": "EMP001", "name": "Rajesh Kumar", "department": "Mining Operations"},
            {"employee_id": "EMP002", "name": "Amit Singh", "department": "Safety"},
            {"employee_id": "EMP003", "name": "Priya Sharma", "department": "Engineering"},
            {"employee_id": "EMP004", "name": "Suresh Patel", "department": "Mining Operations"},
            {"employee_id": "EMP005", "name": "Vikram Yadav", "department": "Maintenance"},
            {"employee_id": "EMP006", "name": "Anita Gupta", "department": "Mining Operations"},
            {"employee_id": "EMP007", "name": "Ravi Verma", "department": "Safety"},
            {"employee_id": "EMP008", "name": "Deepak Joshi", "department": "Engineering"},
            {"employee_id": "EMP009", "name": "Kavita Reddy", "department": "Mining Operations"},
            {"employee_id": "EMP010", "name": "Arun Mishra", "department": "Maintenance"},
            {"employee_id": "EMP011", "name": "Sunita Das", "department": "Mining Operations"},
            {"employee_id": "EMP012", "name": "Manoj Tiwari", "department": "Safety"},
            {"employee_id": "EMP013", "name": "Pooja Nair", "department": "Engineering"},
            {"employee_id": "EMP014", "name": "Sanjay Mehta", "department": "Mining Operations"},
            {"employee_id": "EMP015", "name": "Neha Chopra", "department": "Maintenance"},
        ]

        for worker_data in sample_workers:
            worker_id = db.workers.insert_one({
                **worker_data,
                "mine_id": str(mines[0]["_id"]),
                "assigned_shift": random.choice(["day", "afternoon", "night"]),
                "is_active": True,
                "face_registered": True,
                "compliance_score": random.randint(75, 100),
                "total_violations": random.randint(0, 5),
                "badges": [],
                "created_at": datetime.utcnow()
            }).inserted_id
            workers.append({
                "_id": worker_id,
                **worker_data,
                "mine_id": str(mines[0]["_id"])
            })

        print(f"Created {len(sample_workers)} sample workers")

    # Create mine lookup
    mine_lookup = {str(m["_id"]): m for m in mines}

    # Assign worker profiles for realistic data distribution
    # 60% healthy, 25% normal, 10% at_risk, 5% critical
    worker_profiles = {}
    for i, worker in enumerate(workers):
        rand = random.random()
        if rand < 0.60:
            worker_profiles[str(worker["_id"])] = "healthy"
        elif rand < 0.85:
            worker_profiles[str(worker["_id"])] = "normal"
        elif rand < 0.95:
            worker_profiles[str(worker["_id"])] = "at_risk"
        else:
            worker_profiles[str(worker["_id"])] = "critical"

    # Generate health readings for the past 30 days
    health_readings = []
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)

    print(f"Generating health readings from {start_date.date()} to {end_date.date()}...")

    for worker in workers:
        worker_id = str(worker["_id"])
        mine_id = worker.get("mine_id", str(mines[0]["_id"]))
        mine = mine_lookup.get(mine_id, mines[0])
        profile = worker_profiles.get(worker_id, "normal")

        # Generate 3-8 readings per day for each worker (during work hours)
        current_date = start_date
        while current_date <= end_date:
            # Skip some days randomly (weekends, leave, etc.)
            if random.random() > 0.15:  # 85% attendance
                readings_today = random.randint(3, 8)

                for _ in range(readings_today):
                    # Generate reading during work hours (6 AM to 10 PM)
                    hour = random.randint(6, 22)
                    minute = random.randint(0, 59)
                    reading_time = current_date.replace(hour=hour, minute=minute, second=random.randint(0, 59))

                    reading = generate_health_reading(worker, mine, reading_time, profile)
                    health_readings.append(reading)

            current_date += timedelta(days=1)

    print(f"Generated {len(health_readings)} health readings")

    # Clear existing health readings
    result = db.health_readings.delete_many({})
    print(f"Cleared {result.deleted_count} existing health readings")

    # Insert health readings in batches
    batch_size = 1000
    for i in range(0, len(health_readings), batch_size):
        batch = health_readings[i:i + batch_size]
        db.health_readings.insert_many(batch)
        print(f"Inserted batch {i // batch_size + 1}/{(len(health_readings) + batch_size - 1) // batch_size}")

    # Create indexes
    print("Creating indexes...")
    db.health_readings.create_index([("worker_id", 1), ("timestamp", -1)])
    db.health_readings.create_index([("mine_id", 1), ("timestamp", -1)])
    db.health_readings.create_index("timestamp")
    db.health_readings.create_index("status")

    # Print summary statistics
    total_readings = db.health_readings.count_documents({})
    normal_count = db.health_readings.count_documents({"status": "normal"})
    warning_count = db.health_readings.count_documents({"status": "warning"})
    critical_count = db.health_readings.count_documents({"status": "critical"})

    print("\n" + "=" * 50)
    print("HEALTH DATA SEEDING COMPLETE")
    print("=" * 50)
    print(f"Total readings: {total_readings}")
    print(f"  - Normal: {normal_count} ({normal_count/total_readings*100:.1f}%)")
    print(f"  - Warning: {warning_count} ({warning_count/total_readings*100:.1f}%)")
    print(f"  - Critical: {critical_count} ({critical_count/total_readings*100:.1f}%)")
    print(f"Workers monitored: {len(workers)}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print("=" * 50)

    client.close()
    print("\nDone! Health data has been seeded successfully.")


if __name__ == "__main__":
    seed_health_data()
