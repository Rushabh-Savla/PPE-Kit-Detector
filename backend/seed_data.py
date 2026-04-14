"""
Seed data script to populate the database with test data.
Run this script to create:
- Test users for all roles
- Sample mines with zones and gates
- Sample workers
"""
import asyncio
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
import bcrypt
from bson import ObjectId
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGO_ATLAS_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017/sih_safety_system"))


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


async def seed_database():
    """Main function to seed the database."""
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client.get_default_database()

    print("Starting database seeding...")

    # ==================== Clear existing data (optional) ====================
    print("\nClearing existing data...")
    await db.users.delete_many({})
    await db.workers.delete_many({})
    await db.mines.delete_many({})
    await db.zones.delete_many({})
    await db.gates.delete_many({})
    await db.gate_entries.delete_many({})
    await db.alerts.delete_many({})
    await db.warnings.delete_many({})
    await db.ppe_configs.delete_many({})

    # ==================== Create Mines ====================
    print("\nCreating mines...")

    mine1 = {
        "_id": ObjectId(),
        "name": "Jharia Coal Mine",
        "location": "Jharkhand, India",
        "description": "Main coal mining facility in Jharia region",
        "is_active": True,
        "created_at": datetime.utcnow(),
    }

    mine2 = {
        "_id": ObjectId(),
        "name": "Singareni Collieries",
        "location": "Telangana, India",
        "description": "Coal mining operations in Telangana",
        "is_active": True,
        "created_at": datetime.utcnow(),
    }

    await db.mines.insert_many([mine1, mine2])
    print(f"  Created mines: {mine1['name']}, {mine2['name']}")

    # ==================== Create Zones ====================
    print("\nCreating zones...")

    # Mine 1 Zones
    zone1_1 = {
        "_id": ObjectId(),
        "mine_id": mine1["_id"],
        "name": "Extraction Zone A",
        "description": "Primary extraction area",
        "risk_level": "high",
        "coordinates": {"x": 0, "y": 0, "width": 200, "height": 150},
        "created_at": datetime.utcnow(),
    }

    zone1_2 = {
        "_id": ObjectId(),
        "mine_id": mine1["_id"],
        "name": "Processing Zone",
        "description": "Coal processing and sorting area",
        "risk_level": "normal",
        "coordinates": {"x": 220, "y": 0, "width": 180, "height": 150},
        "created_at": datetime.utcnow(),
    }

    zone1_3 = {
        "_id": ObjectId(),
        "mine_id": mine1["_id"],
        "name": "Storage Zone",
        "description": "Equipment and material storage",
        "risk_level": "low",
        "coordinates": {"x": 0, "y": 170, "width": 150, "height": 100},
        "created_at": datetime.utcnow(),
    }

    # Mine 2 Zones
    zone2_1 = {
        "_id": ObjectId(),
        "mine_id": mine2["_id"],
        "name": "Deep Mining Section",
        "description": "Underground mining operations",
        "risk_level": "critical",
        "coordinates": {"x": 0, "y": 0, "width": 250, "height": 200},
        "created_at": datetime.utcnow(),
    }

    zone2_2 = {
        "_id": ObjectId(),
        "mine_id": mine2["_id"],
        "name": "Surface Operations",
        "description": "Surface level operations",
        "risk_level": "normal",
        "coordinates": {"x": 270, "y": 0, "width": 200, "height": 200},
        "created_at": datetime.utcnow(),
    }

    await db.zones.insert_many([zone1_1, zone1_2, zone1_3, zone2_1, zone2_2])
    print(f"  Created {5} zones")

    # ==================== Create Gates ====================
    print("\nCreating gates...")

    # Mine 1 Gates
    gate1_1 = {
        "_id": ObjectId(),
        "mine_id": mine1["_id"],
        "zone_id": zone1_1["_id"],
        "name": "Main Entry Gate",
        "gate_type": "both",
        "location": "North Entrance",
        "has_camera": True,
        "is_active": True,
        "position": {"x": 100, "y": -20},
        "created_at": datetime.utcnow(),
    }

    gate1_2 = {
        "_id": ObjectId(),
        "mine_id": mine1["_id"],
        "zone_id": zone1_2["_id"],
        "name": "Processing Area Gate",
        "gate_type": "both",
        "location": "East Side",
        "has_camera": True,
        "is_active": True,
        "position": {"x": 400, "y": 75},
        "created_at": datetime.utcnow(),
    }

    # Mine 2 Gates
    gate2_1 = {
        "_id": ObjectId(),
        "mine_id": mine2["_id"],
        "zone_id": zone2_1["_id"],
        "name": "Primary Gate",
        "gate_type": "both",
        "location": "Main Entrance",
        "has_camera": True,
        "is_active": True,
        "position": {"x": 125, "y": -20},
        "created_at": datetime.utcnow(),
    }

    await db.gates.insert_many([gate1_1, gate1_2, gate2_1])
    print(f"  Created {3} gates")

    # ==================== Create Users (Staff) ====================
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
            "assigned_shift": None,
            "assigned_gate_id": None,
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
            "assigned_shift": None,
            "assigned_gate_id": None,
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
        # Area Safety Officer (oversees both mines)
        {
            "_id": ObjectId(),
            "username": "aso",
            "password_hash": get_password_hash("aso123"),
            "full_name": "Priya Sharma",
            "email": "aso@minesafety.com",
            "phone": "+91-9777777777",
            "role": "area_safety_officer",
            "mine_id": None,
            "mine_ids": [str(mine1["_id"]), str(mine2["_id"])],
            "assigned_shift": None,
            "assigned_gate_id": None,
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
        # Manager - Mine 1
        {
            "_id": ObjectId(),
            "username": "manager1",
            "password_hash": get_password_hash("manager123"),
            "full_name": "Amit Patel",
            "email": "manager1@minesafety.com",
            "phone": "+91-9666666666",
            "role": "manager",
            "mine_id": str(mine1["_id"]),
            "mine_ids": [],
            "assigned_shift": None,
            "assigned_gate_id": None,
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
        # Manager - Mine 2
        {
            "_id": ObjectId(),
            "username": "manager2",
            "password_hash": get_password_hash("manager123"),
            "full_name": "Sunita Reddy",
            "email": "manager2@minesafety.com",
            "phone": "+91-9555555555",
            "role": "manager",
            "mine_id": str(mine2["_id"]),
            "mine_ids": [],
            "assigned_shift": None,
            "assigned_gate_id": None,
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
        # Safety Officer - Mine 1
        {
            "_id": ObjectId(),
            "username": "safety1",
            "password_hash": get_password_hash("safety123"),
            "full_name": "Vikram Singh",
            "email": "safety1@minesafety.com",
            "phone": "+91-9444444444",
            "role": "safety_officer",
            "mine_id": str(mine1["_id"]),
            "mine_ids": [],
            "assigned_shift": None,
            "assigned_gate_id": None,
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
        # Safety Officer - Mine 2
        {
            "_id": ObjectId(),
            "username": "safety2",
            "password_hash": get_password_hash("safety123"),
            "full_name": "Meera Das",
            "email": "safety2@minesafety.com",
            "phone": "+91-9333333333",
            "role": "safety_officer",
            "mine_id": str(mine2["_id"]),
            "mine_ids": [],
            "assigned_shift": None,
            "assigned_gate_id": None,
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
        # Shift Incharge - Mine 1, Day Shift
        {
            "_id": ObjectId(),
            "username": "shift_day1",
            "password_hash": get_password_hash("shift123"),
            "full_name": "Ramesh Verma",
            "email": "shift_day1@minesafety.com",
            "phone": "+91-9222222222",
            "role": "shift_incharge",
            "mine_id": str(mine1["_id"]),
            "mine_ids": [],
            "assigned_shift": "day",
            "assigned_gate_id": str(gate1_1["_id"]),
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
        # Shift Incharge - Mine 1, Afternoon Shift
        {
            "_id": ObjectId(),
            "username": "shift_afternoon1",
            "password_hash": get_password_hash("shift123"),
            "full_name": "Sanjay Kumar",
            "email": "shift_afternoon1@minesafety.com",
            "phone": "+91-9111111111",
            "role": "shift_incharge",
            "mine_id": str(mine1["_id"]),
            "mine_ids": [],
            "assigned_shift": "afternoon",
            "assigned_gate_id": str(gate1_1["_id"]),
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
        # Shift Incharge - Mine 1, Night Shift
        {
            "_id": ObjectId(),
            "username": "shift_night1",
            "password_hash": get_password_hash("shift123"),
            "full_name": "Deepak Yadav",
            "email": "shift_night1@minesafety.com",
            "phone": "+91-9000000001",
            "role": "shift_incharge",
            "mine_id": str(mine1["_id"]),
            "mine_ids": [],
            "assigned_shift": "night",
            "assigned_gate_id": str(gate1_1["_id"]),
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
        # Shift Incharge - Mine 2, Day Shift
        {
            "_id": ObjectId(),
            "username": "shift_day2",
            "password_hash": get_password_hash("shift123"),
            "full_name": "Lakshmi Nair",
            "email": "shift_day2@minesafety.com",
            "phone": "+91-9000000002",
            "role": "shift_incharge",
            "mine_id": str(mine2["_id"]),
            "mine_ids": [],
            "assigned_shift": "day",
            "assigned_gate_id": str(gate2_1["_id"]),
            "is_active": True,
            "created_at": datetime.utcnow(),
        },
    ]

    await db.users.insert_many(users)
    print(f"  Created {len(users)} users")

    # ==================== Create Workers ====================
    print("\nCreating workers...")

    workers = []
    worker_names = [
        ("W001", "Rushabh Savla", "Extraction", "day", zone1_1["_id"]),
        ("W002", "Tanush Maloo", "Extraction", "day", zone1_1["_id"]),
        ("W003", "Mahesh Rao", "Extraction", "afternoon", zone1_1["_id"]),
        ("W004", "Ganesh Sharma", "Processing", "day", zone1_2["_id"]),
        ("W005", "Prakash Verma", "Processing", "afternoon", zone1_2["_id"]),
        ("W006", "Anil Singh", "Storage", "day", zone1_3["_id"]),
        ("W007", "Vijay Kumar", "Extraction", "night", zone1_1["_id"]),
        ("W008", "Santosh Das", "Processing", "night", zone1_2["_id"]),
    ]

    for emp_id, name, dept, shift, zone_id in worker_names:
        workers.append({
            "_id": ObjectId(),
            "employee_id": emp_id,
            "name": name,
            "password_hash": get_password_hash("worker123"),
            "department": dept,
            "mine_id": mine1["_id"],
            "zone_id": zone_id,
            "assigned_shift": shift,
            "phone": f"+91-98{emp_id[1:]}00000",
            "emergency_contact": "+91-1234567890",
            "face_registered": False,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "compliance_score": 95.0 + (hash(name) % 10) / 2,
            "total_violations": hash(name) % 3,
            "badges": ["safety_star"] if hash(name) % 2 == 0 else [],
        })

    # Mine 2 workers
    worker_names_2 = [
        ("W101", "Krishna Reddy", "Deep Mining", "day", zone2_1["_id"]),
        ("W102", "Venkat Rao", "Deep Mining", "afternoon", zone2_1["_id"]),
        ("W103", "Nagesh Kumar", "Surface Ops", "day", zone2_2["_id"]),
        ("W104", "Ramana Murthy", "Surface Ops", "afternoon", zone2_2["_id"]),
        ("W105", "Srinivas Rao", "Deep Mining", "night", zone2_1["_id"]),
    ]

    for emp_id, name, dept, shift, zone_id in worker_names_2:
        workers.append({
            "_id": ObjectId(),
            "employee_id": emp_id,
            "name": name,
            "password_hash": get_password_hash("worker123"),
            "department": dept,
            "mine_id": mine2["_id"],
            "zone_id": zone_id,
            "assigned_shift": shift,
            "phone": f"+91-98{emp_id[1:]}00000",
            "emergency_contact": "+91-1234567890",
            "face_registered": False,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "compliance_score": 90.0 + (hash(name) % 20) / 2,
            "total_violations": hash(name) % 5,
            "badges": [],
        })

    await db.workers.insert_many(workers)
    print(f"  Created {len(workers)} workers")

    # ==================== Create Sample Gate Entries ====================
    print("\nCreating sample gate entries...")

    entries = []
    now = datetime.utcnow()

    for worker in workers[:5]:  # First 5 workers
        # Create entry from yesterday
        entry_time = now - timedelta(hours=hash(worker["name"]) % 24)
        entries.append({
            "_id": ObjectId(),
            "gate_id": str(gate1_1["_id"]),
            "mine_id": mine1["_id"],
            "worker_id": str(worker["_id"]),
            "employee_id": worker["employee_id"],
            "worker_name": worker["name"],
            "entry_type": "entry",
            "status": "approved",
            "ppe_status": {"helmet": True, "vest": True, "mask": True},
            "violations": [],
            "timestamp": entry_time,
            "shift": worker["assigned_shift"],
        })

    # Add some entries with violations
    for worker in workers[5:8]:
        entry_time = now - timedelta(hours=hash(worker["name"]) % 12)
        violations = ["NO-Hardhat"] if hash(worker["name"]) % 2 == 0 else ["NO-Safety Vest"]
        entries.append({
            "_id": ObjectId(),
            "gate_id": str(gate1_1["_id"]),
            "mine_id": mine1["_id"],
            "worker_id": str(worker["_id"]),
            "employee_id": worker["employee_id"],
            "worker_name": worker["name"],
            "entry_type": "entry",
            "status": "denied",
            "ppe_status": {
                "helmet": "NO-Hardhat" not in violations,
                "vest": "NO-Safety Vest" not in violations,
                "mask": True
            },
            "violations": violations,
            "timestamp": entry_time,
            "shift": worker["assigned_shift"],
        })

    await db.gate_entries.insert_many(entries)
    print(f"  Created {len(entries)} sample gate entries")

    # ==================== Create Sample Alerts ====================
    print("\nCreating sample alerts...")

    alerts = [
        {
            "_id": ObjectId(),
            "alert_type": "ppe_violation",
            "severity": "medium",
            "status": "active",
            "message": "Multiple helmet violations detected at Main Entry Gate",
            "mine_id": mine1["_id"],
            "gate_id": str(gate1_1["_id"]),
            "created_at": now - timedelta(hours=2),
        },
        {
            "_id": ObjectId(),
            "alert_type": "ppe_violation",
            "severity": "low",
            "status": "acknowledged",
            "message": "Safety vest violation detected",
            "mine_id": mine1["_id"],
            "gate_id": str(gate1_1["_id"]),
            "created_at": now - timedelta(hours=5),
            "acknowledged_at": now - timedelta(hours=4),
            "acknowledged_by": str(users[7]["_id"]),
        },
    ]

    await db.alerts.insert_many(alerts)
    print(f"  Created {len(alerts)} sample alerts")

    # ==================== Create PPE Configurations ====================
    print("\nCreating PPE configurations...")

    ppe_configs = [
        {
            "_id": ObjectId(),
            "mine_id": mine1["_id"],
            "zone_id": None,  # Mine-wide
            "required_items": [
                {"name": "helmet", "required": True, "description": "Safety hard hat"},
                {"name": "vest", "required": True, "description": "High-visibility safety vest"},
                {"name": "mask", "required": True, "description": "Dust mask/respirator"},
                {"name": "boots", "required": True, "description": "Safety boots"},
                {"name": "gloves", "required": False, "description": "Work gloves"},
            ],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        },
        {
            "_id": ObjectId(),
            "mine_id": mine2["_id"],
            "zone_id": None,
            "required_items": [
                {"name": "helmet", "required": True, "description": "Safety hard hat"},
                {"name": "vest", "required": True, "description": "High-visibility safety vest"},
                {"name": "mask", "required": True, "description": "Dust mask/respirator"},
                {"name": "boots", "required": True, "description": "Safety boots"},
            ],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        },
    ]

    await db.ppe_configs.insert_many(ppe_configs)
    print(f"  Created {len(ppe_configs)} PPE configurations")

    # ==================== Print Summary ====================
    print("\n" + "=" * 60)
    print("DATABASE SEEDING COMPLETE!")
    print("=" * 60)

    print("\n=== TEST LOGIN CREDENTIALS ===")
    print("\nStaff Users:")
    print("-" * 40)
    print(f"{'Role':<25} {'Username':<15} Password")
    print("-" * 40)
    print(f"{'Super Admin':<25} {'superadmin':<15} admin123")
    print(f"{'General Manager':<25} {'gm':<15} gm123")
    print(f"{'Area Safety Officer':<25} {'aso':<15} aso123")
    print(f"{'Manager (Mine 1)':<25} {'manager1':<15} manager123")
    print(f"{'Manager (Mine 2)':<25} {'manager2':<15} manager123")
    print(f"{'Safety Officer (Mine 1)':<25} {'safety1':<15} safety123")
    print(f"{'Safety Officer (Mine 2)':<25} {'safety2':<15} safety123")
    print(f"{'Shift Incharge (Day)':<25} {'shift_day1':<15} shift123")
    print(f"{'Shift Incharge (Afternoon)':<25} {'shift_afternoon1':<15} shift123")
    print(f"{'Shift Incharge (Night)':<25} {'shift_night1':<15} shift123")

    print("\nWorkers (login via /auth/worker/login):")
    print("-" * 40)
    print(f"{'Employee ID':<15} Password")
    print("-" * 40)
    print(f"{'W001-W008':<15} worker123  (Mine 1)")
    print(f"{'W101-W105':<15} worker123  (Mine 2)")

    print("\n=== CREATED DATA SUMMARY ===")
    print(f"Mines: 2")
    print(f"Zones: 5")
    print(f"Gates: 3")
    print(f"Staff Users: {len(users)}")
    print(f"Workers: {len(workers)}")
    print(f"Gate Entries: {len(entries)}")
    print(f"Alerts: {len(alerts)}")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(seed_database())
