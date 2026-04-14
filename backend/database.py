"""
MongoDB database connection and models.
"""
import os
import asyncio
from datetime import datetime
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load .env file from the backend directory
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

MONGODB_URI = os.getenv("MONGO_ATLAS_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017/sih_safety_system"))

client: Optional[AsyncIOMotorClient] = None
db = None


async def connect_to_mongodb(max_retries: int = 3):
    """Connect to MongoDB with retry logic."""
    global client, db

    # Connection settings for better reliability
    client = AsyncIOMotorClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=30000,  # 30 seconds
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
        retryWrites=True,
        retryReads=True,
    )
    db = client.get_default_database()

    # Test connection with retry
    for attempt in range(max_retries):
        try:
            await client.admin.command('ping')
            print(f"MongoDB connection successful")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"MongoDB connection attempt {attempt + 1} failed, retrying in 2s...")
                await asyncio.sleep(2)
            else:
                print(f"Warning: Could not ping MongoDB after {max_retries} attempts: {e}")
                # Continue anyway - indexes will be created on first use

    # Create indexes for all collections (wrapped in try-except for resilience)
    try:
        # Users collection (staff/management)
        await db.users.create_index("username", unique=True)
        await db.users.create_index("role")
        await db.users.create_index("mine_id")
        await db.users.create_index("is_active")

        # Workers collection (extended employees)
        await db.workers.create_index("employee_id", unique=True)
        await db.workers.create_index("name")
        await db.workers.create_index("mine_id")
        await db.workers.create_index("zone_id")
        await db.workers.create_index("assigned_shift")
        await db.workers.create_index("is_active")

        # Mines collection
        await db.mines.create_index("name")
        await db.mines.create_index("is_active")

        # Zones collection
        await db.zones.create_index("mine_id")
        await db.zones.create_index("name")

        # Gates collection
        await db.gates.create_index("mine_id")
        await db.gates.create_index("zone_id")
        await db.gates.create_index("is_active")

        # Gate entries collection
        await db.gate_entries.create_index([("gate_id", 1), ("timestamp", -1)])
        await db.gate_entries.create_index([("worker_id", 1), ("timestamp", -1)])
        await db.gate_entries.create_index("timestamp")
        await db.gate_entries.create_index("shift")
        await db.gate_entries.create_index("status")

        # Alerts collection
        await db.alerts.create_index([("mine_id", 1), ("created_at", -1)])
        await db.alerts.create_index("status")
        await db.alerts.create_index("severity")
        await db.alerts.create_index("alert_type")

        # PPE configurations collection
        await db.ppe_configs.create_index([("mine_id", 1), ("zone_id", 1)], unique=True)

        # Warnings collection
        await db.warnings.create_index([("worker_id", 1), ("issued_at", -1)])
        await db.warnings.create_index("issued_by")

        # Incidents collection
        await db.incidents.create_index([("mine_id", 1), ("reported_at", -1)])
        await db.incidents.create_index("status")
        await db.incidents.create_index("severity")

        # Predictions collection (ML-based worker risk predictions)
        await db.predictions.create_index([("worker_id", 1), ("prediction_date", -1)])
        await db.predictions.create_index("overall_risk_score")
        await db.predictions.create_index("risk_category")
        await db.predictions.create_index("expires_at")
        await db.predictions.create_index("requires_intervention")

        # Health readings collection (from helmet/ear sensors)
        await db.health_readings.create_index([("worker_id", 1), ("timestamp", -1)])
        await db.health_readings.create_index([("mine_id", 1), ("timestamp", -1)])
        await db.health_readings.create_index("timestamp")
        await db.health_readings.create_index("status")
        await db.health_readings.create_index("spo2")
        await db.health_readings.create_index("heart_rate")

        # Legacy collections (for backward compatibility)
        await db.employees.create_index("employee_id", unique=True)
        await db.employees.create_index("name")
        await db.attendance.create_index([("employee_id", 1), ("timestamp", -1)])
        await db.attendance.create_index("date")
        await db.ppe_violations.create_index([("employee_id", 1), ("timestamp", -1)])
        await db.ppe_violations.create_index("timestamp")
        await db.admins.create_index("username", unique=True)

        # Report schedules collection
        await db.report_schedules.create_index("report_type")
        await db.report_schedules.create_index("created_by")
        await db.report_schedules.create_index("mine_id")
        await db.report_schedules.create_index("is_active")
        await db.report_schedules.create_index("next_run")

        # Report history collection
        await db.report_history.create_index([("schedule_id", 1), ("generated_at", -1)])
        await db.report_history.create_index("report_type")
        await db.report_history.create_index("generated_at")

        print("Database indexes created successfully")
    except Exception as e:
        print(f"Warning: Could not create indexes (will be created on first use): {e}")

    print("Connected to MongoDB")


async def close_mongodb_connection():
    """Close MongoDB connection."""
    global client
    if client:
        client.close()
        print("MongoDB connection closed")


def get_database():
    """Get database instance."""
    return db


# ==================== Collection Helpers ====================

# User/Role collections
async def get_users_collection():
    return db.users


async def get_workers_collection():
    return db.workers


# Mine/Site collections
async def get_mines_collection():
    return db.mines


async def get_zones_collection():
    return db.zones


async def get_gates_collection():
    return db.gates


# Operations collections
async def get_gate_entries_collection():
    return db.gate_entries


async def get_alerts_collection():
    return db.alerts


async def get_ppe_configs_collection():
    return db.ppe_configs


async def get_warnings_collection():
    return db.warnings


async def get_incidents_collection():
    return db.incidents


async def get_predictions_collection():
    return db.predictions


async def get_health_readings_collection():
    return db.health_readings


# Legacy collections
async def get_employees_collection():
    return db.employees


async def get_attendance_collection():
    return db.attendance


async def get_ppe_violations_collection():
    return db.ppe_violations


async def get_admins_collection():
    return db.admins
