"""
Seed script for Gas Monitoring and SOS Alerts mock data.
Populates MongoDB with realistic historical data for dashboards.
"""
import asyncio
import random
from datetime import datetime, timedelta
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGO_ATLAS_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017/sih_safety_system"))

# Mine zone layouts for danger zone mapping
MINE_ZONES = [
    {"name": "Tunnel A - Section 1", "risk_level": "low", "coordinates": {"x": 10, "y": 10, "width": 20, "height": 15}, "depth_m": 150},
    {"name": "Tunnel A - Section 2", "risk_level": "medium", "coordinates": {"x": 35, "y": 10, "width": 25, "height": 15}, "depth_m": 200},
    {"name": "Tunnel B - Main Shaft", "risk_level": "high", "coordinates": {"x": 10, "y": 30, "width": 30, "height": 20}, "depth_m": 350},
    {"name": "Tunnel B - Deep Extraction", "risk_level": "critical", "coordinates": {"x": 45, "y": 30, "width": 20, "height": 20}, "depth_m": 450},
    {"name": "Tunnel C - Ventilation Area", "risk_level": "low", "coordinates": {"x": 70, "y": 10, "width": 20, "height": 40}, "depth_m": 100},
    {"name": "Central Hub", "risk_level": "low", "coordinates": {"x": 35, "y": 55, "width": 30, "height": 15}, "depth_m": 50},
    {"name": "Equipment Storage", "risk_level": "medium", "coordinates": {"x": 10, "y": 55, "width": 20, "height": 15}, "depth_m": 75},
    {"name": "Emergency Shelter", "risk_level": "low", "coordinates": {"x": 70, "y": 55, "width": 20, "height": 15}, "depth_m": 50},
]

# Worker names for realistic data
WORKER_NAMES = [
    "Rajesh Kumar", "Mukesh Yadav", "Suresh Singh", "Ramesh Patel", "Anil Sharma",
    "Vikram Verma", "Sanjay Gupta", "Deepak Mishra", "Manoj Dubey", "Ashok Pandey",
    "Rahul Tiwari", "Pradeep Chauhan", "Gopal Agarwal", "Vijay Srivastava", "Ravi Rastogi",
    "Amit Saxena", "Nitin Tripathi", "Sunil Joshi", "Pawan Mehta", "Dinesh Rao"
]

# SOS reason types
SOS_REASONS = [
    "Gas leak detected nearby",
    "Tunnel collapse warning",
    "Worker injured - requires medical assistance",
    "Equipment malfunction - trapped",
    "Breathing difficulty - low oxygen",
    "Lost communication with team",
    "Flooding in section",
    "Fire/smoke detected",
    "Structural instability observed",
    "Heat exhaustion symptoms"
]

# Response actions for SOS
RESPONSE_ACTIONS = [
    "Emergency evacuation initiated",
    "Rescue team dispatched",
    "Medical team on standby",
    "Ventilation increased",
    "Section sealed off",
    "All clear - false alarm",
    "Worker rescued successfully",
    "Situation contained"
]


async def seed_data():
    """Main seeding function."""
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client.get_default_database()

    print("Connected to MongoDB. Starting data seeding...")

    # Get or create a mine
    mine = await db.mines.find_one({"is_active": True})
    if not mine:
        mine_result = await db.mines.insert_one({
            "name": "Jharia Coal Mine - Main Site",
            "location": "Jharia, Jharkhand, India",
            "description": "Primary extraction site with deep mining operations",
            "is_active": True,
            "created_at": datetime.utcnow()
        })
        mine_id = mine_result.inserted_id
        print(f"Created mine with ID: {mine_id}")
    else:
        mine_id = mine["_id"]
        print(f"Using existing mine: {mine['name']}")

    # Create zones if not exist
    existing_zones = await db.zones.count_documents({"mine_id": mine_id})
    zone_ids = []

    if existing_zones < len(MINE_ZONES):
        await db.zones.delete_many({"mine_id": mine_id})
        for zone_data in MINE_ZONES:
            result = await db.zones.insert_one({
                "mine_id": mine_id,
                "name": zone_data["name"],
                "risk_level": zone_data["risk_level"],
                "coordinates": zone_data["coordinates"],
                "depth_m": zone_data["depth_m"],
                "worker_count": random.randint(5, 25),
                "created_at": datetime.utcnow()
            })
            zone_ids.append(result.inserted_id)
        print(f"Created {len(zone_ids)} zones")
    else:
        async for zone in db.zones.find({"mine_id": mine_id}):
            zone_ids.append(zone["_id"])
        print(f"Using existing {len(zone_ids)} zones")

    # Seed gas readings (last 7 days)
    print("\nSeeding gas readings...")
    await seed_gas_readings(db, mine_id, zone_ids)

    # Seed danger zone history
    print("\nSeeding danger zone history...")
    await seed_danger_zones(db, mine_id, zone_ids)

    # Seed SOS alerts
    print("\nSeeding SOS alerts...")
    await seed_sos_alerts(db, mine_id, zone_ids)

    # Create gas readings index
    await db.gas_readings.create_index([("mine_id", 1), ("timestamp", -1)])
    await db.gas_readings.create_index("severity")
    await db.gas_readings.create_index("zone_id")

    # Create danger zones index
    await db.danger_zones.create_index([("mine_id", 1), ("detected_at", -1)])
    await db.danger_zones.create_index("status")

    # Create SOS alerts index
    await db.sos_alerts.create_index([("mine_id", 1), ("created_at", -1)])
    await db.sos_alerts.create_index("status")
    await db.sos_alerts.create_index("worker_id")

    print("\nData seeding completed successfully!")
    client.close()


async def seed_gas_readings(db, mine_id, zone_ids):
    """Seed historical gas readings."""
    readings = []
    now = datetime.utcnow()

    # Generate readings for last 7 days, every 15 minutes
    for days_ago in range(7, -1, -1):
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                timestamp = now - timedelta(days=days_ago, hours=hour, minutes=minute)

                for zone_id in zone_ids:
                    zone = await db.zones.find_one({"_id": zone_id})
                    risk_level = zone.get("risk_level", "low") if zone else "low"
                    depth = zone.get("depth_m", 100) if zone else 100

                    # Base levels increase with depth and risk
                    base_methane = 500 + (depth * 5) + {"low": 0, "medium": 1000, "high": 2500, "critical": 5000}.get(risk_level, 0)
                    base_co = 5 + (depth * 0.02) + {"low": 0, "medium": 5, "high": 15, "critical": 25}.get(risk_level, 0)

                    # Add realistic variation
                    methane_variation = random.gauss(0, base_methane * 0.3)
                    co_variation = random.gauss(0, base_co * 0.3)

                    # Occasional spikes (simulate incidents)
                    if random.random() < 0.02:  # 2% chance of spike
                        methane_variation += random.uniform(3000, 8000)
                        co_variation += random.uniform(15, 35)

                    methane_ppm = max(100, base_methane + methane_variation)
                    co_ppm = max(1, base_co + co_variation)

                    # Determine severity
                    severity = "normal"
                    if methane_ppm > 12500 or co_ppm > 50:
                        severity = "critical"
                    elif methane_ppm > 10000 or co_ppm > 35:
                        severity = "high"
                    elif methane_ppm > 5000 or co_ppm > 25:
                        severity = "medium"

                    readings.append({
                        "mine_id": mine_id,
                        "zone_id": zone_id,
                        "sensor_id": f"GS-{str(zone_id)[-6:]}",
                        "methane_ppm": round(methane_ppm, 2),
                        "co_ppm": round(co_ppm, 2),
                        "pressure_hpa": round(random.uniform(990, 1020), 1),
                        "altitude_m": round(random.uniform(-depth - 50, -depth + 50), 1),
                        "temperature_c": round(random.uniform(25, 45), 1),
                        "humidity": round(random.uniform(60, 95), 1),
                        "severity": severity,
                        "timestamp": timestamp
                    })

    # Insert in batches
    if readings:
        # Clear old readings first
        await db.gas_readings.delete_many({"mine_id": mine_id})

        batch_size = 1000
        for i in range(0, len(readings), batch_size):
            batch = readings[i:i + batch_size]
            await db.gas_readings.insert_many(batch)
            print(f"  Inserted {min(i + batch_size, len(readings))}/{len(readings)} gas readings")

    print(f"  Total gas readings: {len(readings)}")


async def seed_danger_zones(db, mine_id, zone_ids):
    """Seed danger zone history with incidents."""
    danger_zones = []
    now = datetime.utcnow()

    # Create danger zone incidents over the past 30 days
    for days_ago in range(30, -1, -1):
        # Random number of incidents per day (0-3)
        num_incidents = random.randint(0, 3)

        for _ in range(num_incidents):
            zone_id = random.choice(zone_ids)
            zone = await db.zones.find_one({"_id": zone_id})

            incident_time = now - timedelta(
                days=days_ago,
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            # Duration of danger status (15 mins to 4 hours)
            duration_mins = random.randint(15, 240)

            # Determine if resolved
            resolved = days_ago > 0 or random.random() > 0.3

            danger_type = random.choice([
                "high_methane", "high_co", "combined_gas",
                "ventilation_failure", "structural_risk"
            ])

            danger_zones.append({
                "mine_id": mine_id,
                "zone_id": zone_id,
                "zone_name": zone["name"] if zone else "Unknown Zone",
                "coordinates": zone.get("coordinates", {"x": 50, "y": 50, "width": 20, "height": 20}) if zone else None,
                "danger_type": danger_type,
                "severity": random.choice(["medium", "high", "critical"]),
                "detected_at": incident_time,
                "resolved_at": incident_time + timedelta(minutes=duration_mins) if resolved else None,
                "status": "resolved" if resolved else "active",
                "peak_methane_ppm": round(random.uniform(8000, 18000), 2) if "methane" in danger_type or danger_type == "combined_gas" else None,
                "peak_co_ppm": round(random.uniform(30, 80), 2) if "co" in danger_type or danger_type == "combined_gas" else None,
                "affected_workers": random.randint(2, 15),
                "evacuation_ordered": random.random() > 0.5,
                "notes": f"Automated detection triggered by sensor readings. {random.choice(['Ventilation adjusted.', 'Workers evacuated.', 'Monitoring continued.', 'Section sealed temporarily.'])}"
            })

    if danger_zones:
        await db.danger_zones.delete_many({"mine_id": mine_id})
        await db.danger_zones.insert_many(danger_zones)

    print(f"  Total danger zone incidents: {len(danger_zones)}")


async def seed_sos_alerts(db, mine_id, zone_ids):
    """Seed SOS alert history."""
    sos_alerts = []
    now = datetime.utcnow()

    # Get or create some workers
    workers = []
    async for worker in db.workers.find({"mine_id": mine_id}).limit(20):
        workers.append(worker)

    if not workers:
        # Create mock workers if none exist
        for i, name in enumerate(WORKER_NAMES):
            worker_result = await db.workers.insert_one({
                "employee_id": f"EMP{1001 + i}",
                "name": name,
                "department": random.choice(["Extraction", "Drilling", "Transport", "Maintenance", "Safety"]),
                "mine_id": mine_id,
                "zone_id": random.choice(zone_ids),
                "assigned_shift": random.choice(["day", "afternoon", "night"]),
                "phone": f"+91 98765{43210 + i}",
                "emergency_contact": f"+91 98765{98765 - i}",
                "face_registered": True,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "compliance_score": random.uniform(70, 100),
                "total_violations": random.randint(0, 10),
                "badges": []
            })
            workers.append({"_id": worker_result.inserted_id, "name": name, "employee_id": f"EMP{1001 + i}"})
        print(f"  Created {len(workers)} workers")

    # Create SOS alerts over the past 60 days
    for days_ago in range(60, -1, -1):
        # Fewer SOS alerts (0-2 per day on average)
        num_alerts = random.choices([0, 1, 2, 3], weights=[0.4, 0.35, 0.2, 0.05])[0]

        for _ in range(num_alerts):
            worker = random.choice(workers)
            zone_id = random.choice(zone_ids)
            zone = await db.zones.find_one({"_id": zone_id})

            alert_time = now - timedelta(
                days=days_ago,
                hours=random.randint(6, 22),  # Most alerts during work hours
                minutes=random.randint(0, 59)
            )

            # Geolocation within the zone
            zone_coords = zone.get("coordinates", {"x": 50, "y": 50, "width": 20, "height": 20}) if zone else {"x": 50, "y": 50, "width": 20, "height": 20}
            location = {
                "x": zone_coords["x"] + random.uniform(0, zone_coords["width"]),
                "y": zone_coords["y"] + random.uniform(0, zone_coords["height"]),
                "depth_m": zone.get("depth_m", 100) + random.uniform(-20, 20) if zone else 100,
                "section": zone["name"] if zone else "Unknown Section"
            }

            # Determine status based on age
            if days_ago > 0:
                status = "resolved"
                response_time_mins = random.randint(2, 30)
                resolution_time_mins = random.randint(15, 180)
            elif random.random() > 0.2:
                status = "acknowledged"
                response_time_mins = random.randint(2, 15)
                resolution_time_mins = None
            else:
                status = "active"
                response_time_mins = None
                resolution_time_mins = None

            severity = random.choices(
                ["critical", "high", "medium"],
                weights=[0.3, 0.5, 0.2]
            )[0]

            sos_alerts.append({
                "mine_id": mine_id,
                "zone_id": zone_id,
                "zone_name": zone["name"] if zone else "Unknown Zone",
                "worker_id": worker["_id"],
                "worker_name": worker.get("name", "Unknown Worker"),
                "employee_id": worker.get("employee_id", "Unknown"),
                "reason": random.choice(SOS_REASONS),
                "severity": severity,
                "status": status,
                "location": location,
                "created_at": alert_time,
                "acknowledged_at": alert_time + timedelta(minutes=response_time_mins) if response_time_mins else None,
                "acknowledged_by": random.choice(["Safety Officer", "Shift Incharge", "Control Room"]) if response_time_mins else None,
                "resolved_at": alert_time + timedelta(minutes=resolution_time_mins) if resolution_time_mins else None,
                "resolved_by": random.choice(["Rescue Team", "Medical Team", "Safety Officer"]) if resolution_time_mins else None,
                "resolution_notes": random.choice(RESPONSE_ACTIONS) if status == "resolved" else None,
                "nearby_workers_notified": random.randint(3, 15),
                "evacuation_triggered": severity == "critical" and random.random() > 0.3,
                "audio_broadcast_sent": True,
                "response_actions": [
                    {"action": "Alert received", "timestamp": alert_time.isoformat(), "by": "System"},
                    {"action": "Nearby workers notified via audio", "timestamp": (alert_time + timedelta(seconds=5)).isoformat(), "by": "System"},
                ] + ([
                    {"action": "Alert acknowledged", "timestamp": (alert_time + timedelta(minutes=response_time_mins)).isoformat(), "by": random.choice(["Safety Officer", "Shift Incharge"])}
                ] if response_time_mins else []) + ([
                    {"action": random.choice(RESPONSE_ACTIONS), "timestamp": (alert_time + timedelta(minutes=resolution_time_mins)).isoformat(), "by": "Rescue Team"}
                ] if resolution_time_mins else [])
            })

    if sos_alerts:
        await db.sos_alerts.delete_many({"mine_id": mine_id})
        await db.sos_alerts.insert_many(sos_alerts)

    print(f"  Total SOS alerts: {len(sos_alerts)}")


if __name__ == "__main__":
    asyncio.run(seed_data())
