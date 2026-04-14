"""
Comprehensive seed script that populates the database with a FULL YEAR of data.
This includes:
- Gate entries with realistic seasonal/weekly patterns
- Violations with trend variations
- Health readings
- Predictions
- Alerts
- All shifts covered

Run: python seed_yearly_data.py
"""
import asyncio
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
import bcrypt
from bson import ObjectId
import os
import random
import math
from dotenv import load_dotenv
import time

load_dotenv()

MONGODB_URI = os.getenv("MONGO_ATLAS_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017/sih_safety_system"))


async def safe_insert_many(collection, documents, max_retries=3):
    """Insert documents with retry logic for connection issues."""
    for attempt in range(max_retries):
        try:
            await collection.insert_many(documents)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    [RETRY] Insert failed, attempt {attempt + 2}/{max_retries}...")
                await asyncio.sleep(2)
            else:
                print(f"    [ERROR] Insert failed after {max_retries} attempts: {str(e)[:100]}")
                return False
    return False


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


async def seed_yearly_data():
    """Main function to seed the database with a full year of data."""
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client.get_default_database()

    print("=" * 70)
    print("       YEARLY DATABASE SEEDING (365 DAYS OF DATA)")
    print("=" * 70)

    # Get existing data
    mines = await db.mines.find({"is_active": True}).to_list(length=None)
    workers = await db.workers.find({"is_active": True}).to_list(length=None)
    gates = await db.gates.find({"is_active": True}).to_list(length=None)
    zones = await db.zones.find({}).to_list(length=None)

    if not mines or not workers or not gates:
        print("\nERROR: No base data found!")
        print("Please run 'python seed_full_data.py' first to create base data.")
        client.close()
        return

    print(f"\nFound: {len(mines)} mines, {len(workers)} workers, {len(gates)} gates")

    # Clear old historical data (but keep base entities)
    print("\nClearing old entries (keeping last 7 days)...")
    cutoff = datetime.utcnow() - timedelta(days=7)
    result = await db.gate_entries.delete_many({"timestamp": {"$lt": cutoff}})
    print(f"  Deleted {result.deleted_count} old entries")

    # Generate full year of data
    print("\nGenerating 365 days of gate entries...")
    await create_yearly_entries(db, workers, gates)

    print("\nGenerating health readings...")
    await create_yearly_health_data(db, workers, mines)

    print("\nGenerating prediction data...")
    await create_yearly_predictions(db, workers)

    print("\nGenerating alerts history...")
    await create_yearly_alerts(db, mines, gates)

    # Print summary
    await print_summary(db)

    client.close()
    print("\n[DONE] Yearly data seeding complete!")


async def create_yearly_entries(db, workers, gates, days=365):
    """Create gate entries for the past year with realistic patterns."""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    total_entries = 0
    total_violations = 0
    batch = []
    batch_size = 1000

    # Create worker -> gates mapping for efficiency
    worker_gates_map = {}
    for worker in workers:
        mine_id = worker.get("mine_id")
        worker_gates = [g for g in gates if g.get("mine_id") == mine_id]
        worker_gates_map[str(worker["_id"])] = worker_gates if worker_gates else gates

    print(f"  Processing {days} days...")

    for day_offset in range(1, days + 1):
        day_date = today_start - timedelta(days=day_offset)
        day_of_week = day_date.weekday()  # 0=Monday, 6=Sunday

        # Progress indicator
        if day_offset % 30 == 0:
            print(f"    Day {day_offset}/{days} ({day_date.strftime('%Y-%m-%d')})...")
            # Flush batch
            if batch:
                await db.gate_entries.insert_many(batch)
                total_entries += len(batch)
                batch = []

        # Weekend attendance is lower
        if day_of_week >= 5:  # Saturday/Sunday
            attendance_rate = random.uniform(0.3, 0.5)
        else:
            attendance_rate = random.uniform(0.65, 0.85)

        # Seasonal variations - worse compliance in monsoon (Jul-Sep)
        month = day_date.month
        if month in [7, 8, 9]:  # Monsoon months
            base_violation_rate = 0.18
        elif month in [12, 1, 2]:  # Winter (foggy conditions)
            base_violation_rate = 0.12
        else:
            base_violation_rate = 0.08

        # Add some random variance
        violation_rate = base_violation_rate * random.uniform(0.7, 1.3)

        # Get daily workers
        num_workers = int(len(workers) * attendance_rate)
        daily_workers = random.sample(workers, k=num_workers)

        # Create entries for all 3 shifts
        shifts = [
            ("day", 6, 14),
            ("afternoon", 14, 22),
            ("night", 22, 6),
        ]

        for shift_name, start_hour, end_hour in shifts:
            # Different shifts have different worker counts
            if shift_name == "night":
                shift_workers = random.sample(daily_workers, k=int(len(daily_workers) * 0.6))
            else:
                shift_workers = random.sample(daily_workers, k=int(len(daily_workers) * 0.8))

            for worker in shift_workers:
                worker_id = str(worker["_id"])
                worker_gates_list = worker_gates_map.get(worker_id, gates)
                gate = random.choice(worker_gates_list)

                # Entry time
                entry_hour = start_hour + random.randint(0, 2)
                if entry_hour >= 24:
                    entry_hour -= 24
                entry_time = day_date.replace(hour=entry_hour, minute=random.randint(0, 59))

                # PPE status
                has_violation = random.random() < violation_rate
                violations = []
                ppe_status = {"helmet": True, "vest": True, "mask": True, "boots": True}

                if has_violation:
                    # Weighted violation types (helmet most common)
                    violation_weights = [
                        ("helmet", 0.4),
                        ("vest", 0.3),
                        ("mask", 0.2),
                        ("boots", 0.1),
                    ]
                    for item, weight in violation_weights:
                        if random.random() < weight:
                            ppe_status[item] = False
                            violations.append(f"NO-{item.title()}")
                            break

                if violations:
                    total_violations += 1

                entry = {
                    "gate_id": str(gate["_id"]),
                    "gate_name": gate.get("name", ""),
                    "mine_id": worker.get("mine_id"),
                    "worker_id": worker_id,
                    "employee_id": worker.get("employee_id", ""),
                    "worker_name": worker.get("name", ""),
                    "entry_type": "entry",
                    "status": "denied" if violations else "approved",
                    "ppe_status": ppe_status,
                    "violations": violations,
                    "timestamp": entry_time,
                    "shift": shift_name,
                    "recorded_by": "yearly_seed",
                    "created_at": entry_time,
                }
                batch.append(entry)

                # Also create exit entry (for some workers)
                if random.random() > 0.3:  # 70% have exit recorded
                    exit_hour = end_hour + random.randint(-1, 1)
                    if exit_hour < 0:
                        exit_hour += 24
                    if exit_hour >= 24:
                        exit_hour -= 24
                    exit_time = entry_time.replace(hour=exit_hour)

                    exit_entry = entry.copy()
                    exit_entry["_id"] = ObjectId()
                    exit_entry["entry_type"] = "exit"
                    exit_entry["timestamp"] = exit_time
                    exit_entry["created_at"] = exit_time
                    exit_entry["status"] = "approved"
                    exit_entry["violations"] = []
                    batch.append(exit_entry)

        # Insert batch if getting too large
        if len(batch) >= batch_size:
            await db.gate_entries.insert_many(batch)
            total_entries += len(batch)
            batch = []

    # Insert remaining batch
    if batch:
        await db.gate_entries.insert_many(batch)
        total_entries += len(batch)

    print(f"  [OK] Created {total_entries:,} gate entries")
    print(f"  [OK] {total_violations:,} entries with violations")


async def create_yearly_health_data(db, workers, mines, days=365):
    """Create health readings for the past year."""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Clear old health data
    cutoff = today_start - timedelta(days=7)
    await db.health_readings.delete_many({"timestamp": {"$lt": cutoff}})

    total_readings = 0
    batch = []
    batch_size = 2000

    # Sample workers for health data (not all have sensors)
    sensor_workers = random.sample(workers, k=int(len(workers) * 0.4))

    print(f"  Generating health data for {len(sensor_workers)} workers with sensors...")

    for day_offset in range(1, min(days, 90) + 1):  # Last 90 days only for health
        day_date = today_start - timedelta(days=day_offset)

        if day_offset % 30 == 0:
            print(f"    Day {day_offset}...")
            if batch:
                await db.health_readings.insert_many(batch)
                total_readings += len(batch)
                batch = []

        # Sample of workers for this day
        daily_workers = random.sample(sensor_workers, k=int(len(sensor_workers) * 0.7))

        for worker in daily_workers:
            mine = next((m for m in mines if m["_id"] == worker.get("mine_id")), None)

            # 2-4 readings per worker per day
            num_readings = random.randint(2, 4)
            for _ in range(num_readings):
                reading_time = day_date + timedelta(hours=random.randint(6, 20), minutes=random.randint(0, 59))

                # Normal ranges with occasional anomalies
                is_anomaly = random.random() < 0.05  # 5% anomaly rate

                if is_anomaly:
                    heart_rate = random.randint(40, 150)
                    spo2 = random.randint(85, 100)
                    body_temp = round(random.uniform(35.5, 39.0), 1)
                else:
                    heart_rate = random.randint(65, 95)
                    spo2 = random.randint(95, 100)
                    body_temp = round(random.uniform(36.2, 37.2), 1)

                # Determine status
                if spo2 < 90 or heart_rate > 120 or heart_rate < 50 or body_temp > 38.5:
                    status = "critical"
                elif spo2 < 94 or heart_rate > 100 or body_temp > 37.8:
                    status = "warning"
                else:
                    status = "normal"

                batch.append({
                    "worker_id": worker["_id"],
                    "employee_id": worker.get("employee_id"),
                    "worker_name": worker.get("name"),
                    "mine_id": worker.get("mine_id"),
                    "mine_name": mine["name"] if mine else "Unknown",
                    "heart_rate": heart_rate,
                    "spo2": spo2,
                    "body_temp": body_temp,
                    "status": status,
                    "source": "helmet_sensor",
                    "timestamp": reading_time,
                    "created_at": reading_time,
                })

        if len(batch) >= batch_size:
            await db.health_readings.insert_many(batch)
            total_readings += len(batch)
            batch = []

    if batch:
        await db.health_readings.insert_many(batch)
        total_readings += len(batch)

    print(f"  [OK] Created {total_readings:,} health readings")


async def create_yearly_predictions(db, workers, days=90):
    """Create prediction records for workers."""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Clear old predictions
    await db.predictions.delete_many({})

    predictions = []
    risk_factors = [
        "Frequent PPE violations",
        "Inconsistent attendance patterns",
        "Working in high-risk zones",
        "Recent safety incidents",
        "Fatigue indicators from long shifts",
        "New to high-risk area",
        "History of minor injuries",
    ]

    recommendations = [
        "Schedule refresher safety training",
        "Review PPE compliance with supervisor",
        "Consider reassignment to lower-risk zone",
        "Mandatory safety briefing required",
        "Recommend shift schedule review",
        "Pair with experienced mentor",
        "Medical checkup recommended",
    ]

    for worker in workers:
        # Calculate risk based on various factors
        base_score = random.uniform(20, 95)

        # Some workers are consistently high/low risk
        worker_tendency = random.gauss(0, 15)
        risk_score = min(100, max(0, base_score + worker_tendency))

        # Determine category
        if risk_score >= 80:
            category = "critical"
        elif risk_score >= 60:
            category = "high"
        elif risk_score >= 40:
            category = "medium"
        else:
            category = "low"

        requires_intervention = risk_score >= 75

        # Select relevant risk factors
        num_factors = 1 if category == "low" else (2 if category == "medium" else random.randint(2, 4))
        worker_factors = random.sample(risk_factors, k=min(num_factors, len(risk_factors)))

        prediction_date = today_start - timedelta(days=random.randint(0, 7))

        predictions.append({
            "_id": ObjectId(),
            "worker_id": worker["_id"],
            "employee_id": worker.get("employee_id"),
            "worker_name": worker.get("name"),
            "mine_id": worker.get("mine_id"),
            "overall_risk_score": round(risk_score, 1),
            "risk_category": category,
            "risk_factors": worker_factors,
            "recommendations": random.sample(recommendations, k=min(2, len(recommendations))) if category != "low" else [],
            "requires_intervention": requires_intervention,
            "prediction_date": prediction_date,
            "expires_at": prediction_date + timedelta(days=7),
            "model_version": "v2.1",
            "confidence_score": round(random.uniform(0.75, 0.98), 2),
            "created_at": prediction_date,
        })

    await db.predictions.insert_many(predictions)
    print(f"  [OK] Created {len(predictions)} worker predictions")

    # Summary
    critical = len([p for p in predictions if p["risk_category"] == "critical"])
    high = len([p for p in predictions if p["risk_category"] == "high"])
    print(f"    Critical: {critical}, High: {high}")


async def create_yearly_alerts(db, mines, gates, days=365):
    """Create alert history for the past year."""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Keep recent alerts, clear old ones
    cutoff = today_start - timedelta(days=30)
    await db.alerts.delete_many({"created_at": {"$lt": cutoff}})

    alerts = []

    alert_templates = [
        ("ppe_violation", "PPE violation detected - Worker not wearing {item}", ["medium", "high"]),
        ("unauthorized_access", "Unauthorized access attempt at {location}", ["high", "critical"]),
        ("equipment_malfunction", "Equipment malfunction reported - {equipment}", ["low", "medium"]),
        ("gas_detection", "Elevated {gas} levels detected in Zone {zone}", ["high", "critical"]),
        ("worker_distress", "Worker distress signal from {location}", ["critical"]),
        ("zone_breach", "Restricted zone breach detected", ["high"]),
        ("overtime_alert", "Worker exceeded maximum shift hours", ["medium"]),
        ("compliance_drop", "Compliance rate dropped below threshold", ["medium", "high"]),
    ]

    items = ["helmet", "safety vest", "mask", "boots"]
    equipment = ["conveyor belt", "drill unit", "ventilation system", "elevator"]
    gases = ["CO", "methane", "SO2"]

    # Create alerts with declining frequency as we go back in time
    for day_offset in range(1, min(days, 180) + 1):  # Last 6 months of alerts
        day_date = today_start - timedelta(days=day_offset)

        # Fewer alerts as we go back
        base_alerts = max(1, 8 - (day_offset // 30))
        num_daily_alerts = random.randint(base_alerts // 2, base_alerts)

        for _ in range(num_daily_alerts):
            mine = random.choice(mines)
            mine_gates = [g for g in gates if g.get("mine_id") == mine["_id"]]
            gate = random.choice(mine_gates) if mine_gates else None

            alert_type, message_template, severities = random.choice(alert_templates)

            # Fill in template
            message = message_template.format(
                item=random.choice(items),
                location=gate["name"] if gate else "Main Entrance",
                equipment=random.choice(equipment),
                gas=random.choice(gases),
                zone=random.randint(1, 5),
            )

            severity = random.choice(severities)
            alert_time = day_date + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))

            # Older alerts are mostly resolved
            if day_offset > 7:
                status = "resolved"
            elif day_offset > 2:
                status = random.choice(["resolved", "resolved", "acknowledged"])
            else:
                status = random.choice(["active", "acknowledged", "resolved"])

            alert = {
                "_id": ObjectId(),
                "alert_type": alert_type,
                "severity": severity,
                "status": status,
                "message": message,
                "mine_id": mine["_id"],
                "mine_name": mine["name"],
                "gate_id": str(gate["_id"]) if gate else None,
                "created_at": alert_time,
                "created_by": "yearly_seed",
            }

            if status in ["acknowledged", "resolved"]:
                alert["acknowledged_at"] = alert_time + timedelta(minutes=random.randint(5, 60))
            if status == "resolved":
                alert["resolved_at"] = alert_time + timedelta(hours=random.randint(1, 24))
                alert["resolution_notes"] = "Issue addressed by safety team"

            alerts.append(alert)

    if alerts:
        await db.alerts.insert_many(alerts)

    active = len([a for a in alerts if a["status"] == "active"])
    print(f"  [OK] Created {len(alerts)} alerts ({active} active)")


async def print_summary(db):
    """Print summary of database contents."""
    print("\n" + "=" * 70)
    print("                    DATABASE SUMMARY")
    print("=" * 70)

    # Core counts
    mines = await db.mines.count_documents({})
    workers = await db.workers.count_documents({"is_active": True})
    gates = await db.gates.count_documents({})

    # Entry stats
    total_entries = await db.gate_entries.count_documents({})
    violations = await db.gate_entries.count_documents({"violations": {"$ne": []}})

    # Time-based stats
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    year_ago = today - timedelta(days=365)

    entries_today = await db.gate_entries.count_documents({"timestamp": {"$gte": today}})
    entries_week = await db.gate_entries.count_documents({"timestamp": {"$gte": week_ago}})
    entries_month = await db.gate_entries.count_documents({"timestamp": {"$gte": month_ago}})
    entries_year = await db.gate_entries.count_documents({"timestamp": {"$gte": year_ago}})

    # Other collections
    health_readings = await db.health_readings.count_documents({})
    predictions = await db.predictions.count_documents({})
    alerts = await db.alerts.count_documents({})
    active_alerts = await db.alerts.count_documents({"status": "active"})

    print(f"\n{'Core Data:':<25}")
    print(f"  {'Mines:':<20} {mines}")
    print(f"  {'Workers:':<20} {workers}")
    print(f"  {'Gates:':<20} {gates}")

    print(f"\n{'Gate Entries:':<25}")
    print(f"  {'Total:':<20} {total_entries:,}")
    print(f"  {'With Violations:':<20} {violations:,} ({violations/max(1,total_entries)*100:.1f}%)")
    print(f"  {'Today:':<20} {entries_today:,}")
    print(f"  {'This Week:':<20} {entries_week:,}")
    print(f"  {'This Month:':<20} {entries_month:,}")
    print(f"  {'This Year:':<20} {entries_year:,}")

    print(f"\n{'Other Data:':<25}")
    print(f"  {'Health Readings:':<20} {health_readings:,}")
    print(f"  {'Predictions:':<20} {predictions}")
    print(f"  {'Alerts:':<20} {alerts} ({active_alerts} active)")

    # Monthly breakdown
    print(f"\n{'Monthly Entry Distribution:':<25}")
    for i in range(12):
        month_start = today.replace(day=1) - timedelta(days=30*i)
        month_end = (month_start + timedelta(days=32)).replace(day=1)
        month_entries = await db.gate_entries.count_documents({
            "timestamp": {"$gte": month_start, "$lt": month_end}
        })
        if month_entries > 0:
            print(f"  {month_start.strftime('%b %Y'):<15} {month_entries:,} entries")


if __name__ == "__main__":
    print("\nStarting yearly data seeding...")
    print("This may take several minutes...\n")
    asyncio.run(seed_yearly_data())
