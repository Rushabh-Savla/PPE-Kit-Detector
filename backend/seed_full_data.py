"""
Comprehensive seed script that populates the database with:
- All base data (mines, zones, gates, users, workers)
- TODAY's gate entries (so dashboard shows current data)
- Recent alerts
- Historical data for analytics

Run this script to ensure the super admin dashboard shows correct data.
"""
import asyncio
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
import bcrypt
from bson import ObjectId
import os
import random
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGO_ATLAS_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017/sih_safety_system"))


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


async def create_indexes(db):
    """Create indexes for better query performance."""
    print("\nCreating database indexes...")

    # Gate entries indexes
    await db.gate_entries.create_index([("timestamp", -1)])
    await db.gate_entries.create_index([("mine_id", 1), ("timestamp", -1)])
    await db.gate_entries.create_index([("worker_id", 1), ("timestamp", -1)])
    await db.gate_entries.create_index([("entry_type", 1), ("timestamp", -1)])
    await db.gate_entries.create_index([("violations", 1)])

    # Workers indexes
    await db.workers.create_index([("mine_id", 1)])
    await db.workers.create_index([("is_active", 1)])
    await db.workers.create_index([("employee_id", 1)], unique=True)

    # Users indexes
    await db.users.create_index([("username", 1)], unique=True)
    await db.users.create_index([("role", 1)])

    # Alerts indexes
    await db.alerts.create_index([("status", 1), ("created_at", -1)])
    await db.alerts.create_index([("mine_id", 1), ("status", 1)])

    # PPE violations indexes
    await db.ppe_violations.create_index([("timestamp", -1)])

    # Attendance indexes
    await db.attendance.create_index([("date", 1)])
    await db.attendance.create_index([("employee_id", 1), ("date", 1)])

    print("  Indexes created successfully")


async def seed_database():
    """Main function to seed the database with comprehensive data."""
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client.get_default_database()

    print("=" * 60)
    print("COMPREHENSIVE DATABASE SEEDING")
    print("=" * 60)

    # Create indexes first
    await create_indexes(db)

    # ==================== Check existing data ====================
    existing_mines = await db.mines.count_documents({})
    existing_workers = await db.workers.count_documents({})

    if existing_mines > 0 and existing_workers > 0:
        print(f"\nFound existing data: {existing_mines} mines, {existing_workers} workers")
        print("Adding today's data without clearing existing data...")

        # Get existing data
        mines = await db.mines.find({"is_active": True}).to_list(length=None)
        workers = await db.workers.find({"is_active": True}).to_list(length=None)
        gates = await db.gates.find({"is_active": True}).to_list(length=None)
        users = await db.users.find({}).to_list(length=None)

        if not gates:
            print("No gates found, creating gates...")
            gates = await create_gates(db, mines)
    else:
        print("\nNo existing data found. Creating full dataset...")
        # Clear and recreate
        await db.users.delete_many({})
        await db.workers.delete_many({})
        await db.mines.delete_many({})
        await db.zones.delete_many({})
        await db.gates.delete_many({})
        await db.alerts.delete_many({})
        await db.warnings.delete_many({})
        await db.ppe_configs.delete_many({})

        mines = await create_mines(db)
        zones = await create_zones(db, mines)
        gates = await create_gates(db, mines, zones)
        users = await create_users(db, mines, gates)
        workers = await create_workers(db, mines, zones)

    # Always create fresh today's data
    await create_todays_entries(db, workers, gates)
    await create_recent_alerts(db, mines, gates)
    await create_historical_data(db, workers, gates, days=30)

    # Print summary
    await print_summary(db)

    client.close()
    print("\nDone!")


async def create_mines(db):
    """Create sample mines."""
    print("\nCreating mines...")

    mines = [
        {
            "_id": ObjectId(),
            "name": "Jharia Coal Mine",
            "location": "Jharkhand, India",
            "description": "Main coal mining facility in Jharia region",
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
        {
            "_id": ObjectId(),
            "name": "Singareni Collieries",
            "location": "Telangana, India",
            "description": "Coal mining operations in Telangana",
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
        {
            "_id": ObjectId(),
            "name": "Korba Coalfield",
            "location": "Chhattisgarh, India",
            "description": "Open cast and underground mining",
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
    ]

    await db.mines.insert_many(mines)
    print(f"  Created {len(mines)} mines")
    return mines


async def create_zones(db, mines):
    """Create zones for each mine."""
    print("\nCreating zones...")

    zones = []
    risk_levels = ["low", "normal", "high", "critical"]
    zone_templates = [
        ("Extraction Zone", "Primary extraction area", "high"),
        ("Processing Zone", "Coal processing and sorting", "normal"),
        ("Storage Zone", "Equipment and material storage", "low"),
        ("Deep Mining Section", "Underground operations", "critical"),
        ("Surface Operations", "Surface level operations", "normal"),
    ]

    for mine in mines:
        for i, (name, desc, risk) in enumerate(zone_templates[:3 + mines.index(mine) % 2]):
            zones.append({
                "_id": ObjectId(),
                "mine_id": mine["_id"],
                "name": f"{name} - {mine['name'][:10]}",
                "description": desc,
                "risk_level": risk,
                "coordinates": {"x": i * 100, "y": 0, "width": 150, "height": 100},
                "created_at": datetime.utcnow(),
            })

    await db.zones.insert_many(zones)
    print(f"  Created {len(zones)} zones")
    return zones


async def create_gates(db, mines, zones=None):
    """Create gates for each mine."""
    print("\nCreating gates...")

    gates = []
    for mine in mines:
        mine_zones = [z for z in (zones or []) if z.get("mine_id") == mine["_id"]]

        for i in range(2):  # 2 gates per mine
            zone_id = mine_zones[i % len(mine_zones)]["_id"] if mine_zones else None
            gates.append({
                "_id": ObjectId(),
                "mine_id": mine["_id"],
                "zone_id": zone_id,
                "name": f"Gate {i+1} - {mine['name'][:15]}",
                "gate_type": "both",
                "location": f"Entrance {i+1}",
                "has_camera": True,
                "is_active": True,
                "position": {"x": 100 + i * 200, "y": -20},
                "created_at": datetime.utcnow(),
            })

    await db.gates.insert_many(gates)
    print(f"  Created {len(gates)} gates")
    return gates


async def create_users(db, mines, gates):
    """Create staff users."""
    print("\nCreating users...")

    users = [
        # Super Admin
        {
            "_id": ObjectId(),
            "username": "superadmin",
            "password_hash": get_password_hash("admin123"),
            "full_name": "System Administrator",
            "email": "admin@minesafety.com",
            "phone": "+91-9999999999",
            "role": "super_admin",
            "mine_id": None,
            "mine_ids": [],
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
        # General Manager
        {
            "_id": ObjectId(),
            "username": "gm",
            "password_hash": get_password_hash("gm123"),
            "full_name": "Rajesh Kumar",
            "email": "gm@minesafety.com",
            "phone": "+91-9888888888",
            "role": "general_manager",
            "mine_id": None,
            "mine_ids": [],
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
        # Area Safety Officer
        {
            "_id": ObjectId(),
            "username": "aso",
            "password_hash": get_password_hash("aso123"),
            "full_name": "Priya Sharma",
            "email": "aso@minesafety.com",
            "phone": "+91-9777777777",
            "role": "area_safety_officer",
            "mine_id": None,
            "mine_ids": [str(m["_id"]) for m in mines],
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
    ]

    # Add managers and safety officers for each mine
    for i, mine in enumerate(mines):
        mine_gates = [g for g in gates if g["mine_id"] == mine["_id"]]

        users.append({
            "_id": ObjectId(),
            "username": f"manager{i+1}",
            "password_hash": get_password_hash("manager123"),
            "full_name": f"Manager {mine['name'][:15]}",
            "email": f"manager{i+1}@minesafety.com",
            "phone": f"+91-966666666{i}",
            "role": "manager",
            "mine_id": str(mine["_id"]),
            "mine_ids": [],
            "is_active": True,
            "created_at": datetime.utcnow(),
        })

        users.append({
            "_id": ObjectId(),
            "username": f"safety{i+1}",
            "password_hash": get_password_hash("safety123"),
            "full_name": f"Safety Officer {mine['name'][:10]}",
            "email": f"safety{i+1}@minesafety.com",
            "phone": f"+91-944444444{i}",
            "role": "safety_officer",
            "mine_id": str(mine["_id"]),
            "mine_ids": [],
            "is_active": True,
            "created_at": datetime.utcnow(),
        })

        # Shift incharges
        for shift in ["day", "afternoon", "night"]:
            gate_id = str(mine_gates[0]["_id"]) if mine_gates else None
            users.append({
                "_id": ObjectId(),
                "username": f"shift_{shift}_{i+1}",
                "password_hash": get_password_hash("shift123"),
                "full_name": f"Shift Incharge {shift.title()} - Mine {i+1}",
                "email": f"shift_{shift}_{i+1}@minesafety.com",
                "phone": f"+91-922222{i}{shift[0]}",
                "role": "shift_incharge",
                "mine_id": str(mine["_id"]),
                "mine_ids": [],
                "assigned_shift": shift,
                "assigned_gate_id": gate_id,
                "is_active": True,
                "created_at": datetime.utcnow(),
            })

    await db.users.insert_many(users)
    print(f"  Created {len(users)} users")
    return users


async def create_workers(db, mines, zones):
    """Create workers for each mine."""
    print("\nCreating workers...")

    workers = []
    departments = ["Extraction", "Processing", "Maintenance", "Transport", "Safety"]
    shifts = ["day", "afternoon", "night"]

    worker_id_counter = 1
    for mine in mines:
        mine_zones = [z for z in zones if z.get("mine_id") == mine["_id"]]
        num_workers = 20 + mines.index(mine) * 5  # 20-30 workers per mine

        for i in range(num_workers):
            zone = random.choice(mine_zones) if mine_zones else None
            emp_id = f"W{worker_id_counter:03d}"
            worker_id_counter += 1

            workers.append({
                "_id": ObjectId(),
                "employee_id": emp_id,
                "name": f"Worker {emp_id}",
                "password_hash": get_password_hash("worker123"),
                "department": random.choice(departments),
                "mine_id": mine["_id"],
                "zone_id": zone["_id"] if zone else None,
                "assigned_shift": random.choice(shifts),
                "phone": f"+91-98{emp_id[1:]}00000",
                "emergency_contact": "+91-1234567890",
                "face_registered": random.random() > 0.3,  # 70% have face registered
                "is_active": True,
                "created_at": datetime.utcnow(),
                "compliance_score": round(random.uniform(70, 100), 1),
                "total_violations": random.randint(0, 10),
                "badges": ["safety_star"] if random.random() > 0.7 else [],
            })

    await db.workers.insert_many(workers)
    print(f"  Created {len(workers)} workers")
    return workers


async def create_todays_entries(db, workers, gates):
    """Create gate entries for TODAY so dashboard shows current data."""
    print("\nCreating today's gate entries...")

    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Clear today's entries to avoid duplicates
    await db.gate_entries.delete_many({
        "timestamp": {"$gte": today_start},
        "recorded_by": "seed_script"
    })

    entries = []
    violations_count = 0

    # Create entries for 60-80% of workers
    active_workers = random.sample(workers, k=int(len(workers) * random.uniform(0.6, 0.8)))

    for worker in active_workers:
        # Find appropriate gate
        worker_gates = [g for g in gates if g.get("mine_id") == worker.get("mine_id")]
        gate = random.choice(worker_gates) if worker_gates else random.choice(gates)

        # Entry time - random time today
        hours_ago = random.randint(0, min(now.hour, 10))
        entry_time = now - timedelta(hours=hours_ago, minutes=random.randint(0, 59))

        # PPE status - 85% compliance rate
        has_violation = random.random() < 0.15
        ppe_items = ["helmet", "vest", "mask", "boots"]
        violations = []
        ppe_status = {}

        for item in ppe_items:
            if has_violation and random.random() < 0.3:
                ppe_status[item] = False
                violations.append(f"NO-{item.title()}")
            else:
                ppe_status[item] = True

        if violations:
            violations_count += 1

        entries.append({
            "_id": ObjectId(),
            "gate_id": str(gate["_id"]),
            "gate_name": gate.get("name", "Unknown Gate"),
            "mine_id": worker.get("mine_id"),
            "worker_id": str(worker["_id"]),
            "employee_id": worker.get("employee_id", ""),
            "worker_name": worker.get("name", ""),
            "entry_type": "entry",
            "status": "denied" if violations else "approved",
            "ppe_status": ppe_status,
            "violations": violations,
            "timestamp": entry_time,
            "shift": worker.get("assigned_shift", "day"),
            "recorded_by": "seed_script",
            "created_at": entry_time,
        })

    if entries:
        await db.gate_entries.insert_many(entries)

    print(f"  Created {len(entries)} entries for today ({violations_count} with violations)")
    return entries


async def create_recent_alerts(db, mines, gates):
    """Create recent active alerts."""
    print("\nCreating recent alerts...")

    # Clear old seeded alerts
    await db.alerts.delete_many({"created_by": "seed_script"})

    now = datetime.utcnow()
    alerts = []

    alert_types = [
        ("ppe_violation", "PPE violation detected at gate", "medium"),
        ("unauthorized_access", "Unauthorized access attempt", "high"),
        ("equipment_malfunction", "Equipment malfunction reported", "low"),
        ("gas_leak", "Elevated gas levels detected", "critical"),
        ("worker_distress", "Worker distress signal received", "high"),
    ]

    for mine in mines:
        mine_gates = [g for g in gates if g.get("mine_id") == mine["_id"]]
        gate = mine_gates[0] if mine_gates else None

        # 2-4 alerts per mine
        num_alerts = random.randint(2, 4)
        for i in range(num_alerts):
            alert_type, message, severity = random.choice(alert_types)
            hours_ago = random.randint(0, 48)

            alert = {
                "_id": ObjectId(),
                "alert_type": alert_type,
                "severity": severity,
                "status": random.choice(["active", "active", "acknowledged"]),  # More active
                "message": f"{message} - {mine['name']}",
                "mine_id": mine["_id"],
                "mine_name": mine["name"],
                "gate_id": str(gate["_id"]) if gate else None,
                "created_at": now - timedelta(hours=hours_ago),
                "created_by": "seed_script",
            }

            if alert["status"] == "acknowledged":
                alert["acknowledged_at"] = now - timedelta(hours=hours_ago - 1)

            alerts.append(alert)

    if alerts:
        await db.alerts.insert_many(alerts)

    active_count = len([a for a in alerts if a["status"] == "active"])
    print(f"  Created {len(alerts)} alerts ({active_count} active)")
    return alerts


async def create_historical_data(db, workers, gates, days=30):
    """Create historical gate entries for the past N days."""
    print(f"\nCreating {days} days of historical data...")

    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    total_entries = 0

    for day_offset in range(1, days + 1):
        day_date = today_start - timedelta(days=day_offset)

        # Show progress
        if day_offset % 10 == 0:
            print(f"  Processing day {day_offset}/{days}...")

        # 50-70% worker attendance per day
        daily_workers = random.sample(workers, k=int(len(workers) * random.uniform(0.5, 0.7)))

        entries = []
        for worker in daily_workers:
            worker_gates = [g for g in gates if g.get("mine_id") == worker.get("mine_id")]
            gate = random.choice(worker_gates) if worker_gates else random.choice(gates)

            # Entry time
            entry_hour = random.randint(5, 10)
            entry_time = day_date.replace(hour=entry_hour, minute=random.randint(0, 59))

            # PPE status
            has_violation = random.random() < 0.1  # 10% violation rate for historical
            violations = []
            ppe_status = {"helmet": True, "vest": True, "mask": True, "boots": True}

            if has_violation:
                item = random.choice(["helmet", "vest", "mask"])
                ppe_status[item] = False
                violations.append(f"NO-{item.title()}")

            entries.append({
                "gate_id": str(gate["_id"]),
                "gate_name": gate.get("name", ""),
                "mine_id": worker.get("mine_id"),
                "worker_id": str(worker["_id"]),
                "employee_id": worker.get("employee_id", ""),
                "worker_name": worker.get("name", ""),
                "entry_type": "entry",
                "status": "denied" if violations else "approved",
                "ppe_status": ppe_status,
                "violations": violations,
                "timestamp": entry_time,
                "shift": worker.get("assigned_shift", "day"),
                "recorded_by": "historical_seed",
                "created_at": entry_time,
            })

        if entries:
            await db.gate_entries.insert_many(entries)
            total_entries += len(entries)

    print(f"  Created {total_entries} historical entries")


async def print_summary(db):
    """Print summary of database contents."""
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)

    mines = await db.mines.count_documents({})
    zones = await db.zones.count_documents({})
    gates = await db.gates.count_documents({})
    users = await db.users.count_documents({})
    workers = await db.workers.count_documents({"is_active": True})

    # Today's stats
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)

    today_entries = await db.gate_entries.count_documents({
        "timestamp": {"$gte": today, "$lt": tomorrow},
        "entry_type": "entry"
    })

    today_violations = await db.gate_entries.count_documents({
        "timestamp": {"$gte": today, "$lt": tomorrow},
        "violations": {"$ne": []}
    })

    active_alerts = await db.alerts.count_documents({"status": "active"})
    total_entries = await db.gate_entries.count_documents({})

    print(f"\nCore Data:")
    print(f"  Mines: {mines}")
    print(f"  Zones: {zones}")
    print(f"  Gates: {gates}")
    print(f"  Staff Users: {users}")
    print(f"  Workers: {workers}")

    print(f"\nActivity Data:")
    print(f"  Total Gate Entries: {total_entries}")
    print(f"  Today's Entries: {today_entries}")
    print(f"  Today's Violations: {today_violations}")
    print(f"  Active Alerts: {active_alerts}")

    compliance_rate = 100.0
    if today_entries > 0:
        compliance_rate = round((1 - today_violations / today_entries) * 100, 1)
    print(f"  Today's Compliance Rate: {compliance_rate}%")

    print("\n" + "=" * 60)
    print("LOGIN CREDENTIALS")
    print("=" * 60)
    print("\nStaff (use /auth/login):")
    print("  Super Admin: superadmin / admin123")
    print("  General Manager: gm / gm123")
    print("  Area Safety Officer: aso / aso123")
    print("  Manager: manager1, manager2, manager3 / manager123")
    print("  Safety Officer: safety1, safety2, safety3 / safety123")
    print("  Shift Incharge: shift_day_1, shift_afternoon_1, etc / shift123")
    print("\nWorkers (use /auth/worker/login):")
    print("  Employee IDs: W001-W060+ / worker123")


if __name__ == "__main__":
    print("Starting comprehensive database seeding...")
    asyncio.run(seed_database())
