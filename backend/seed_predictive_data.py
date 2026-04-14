"""
Generate 6 months of realistic historical data for ML training.

Creates diverse worker patterns:
- Model workers (25%): High compliance
- Average workers (50%): Mostly compliant
- At-risk workers (15%): Frequent violations
- High-risk workers (10%): Critical issues
"""

import asyncio
import random
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGO_ATLAS_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017/sih_safety_system"))

# Worker Archetypes (for realistic patterns)
ARCHETYPES = {
    "model_worker": {
        "attendance_rate": 0.95,
        "violation_rate": 0.02,
        "compliance_score_target": 95
    },
    "average_worker": {
        "attendance_rate": 0.85,
        "violation_rate": 0.08,
        "compliance_score_target": 82
    },
    "at_risk_worker": {
        "attendance_rate": 0.70,
        "violation_rate": 0.20,
        "compliance_score_target": 65
    },
    "high_risk_worker": {
        "attendance_rate": 0.55,
        "violation_rate": 0.35,
        "compliance_score_target": 45
    }
}


async def seed_predictive_data():
    """
    Generate 6 months of historical gate entries with realistic patterns.
    """
    print("="*60)
    print("Seeding Predictive Analysis Data")
    print("="*60)

    client = AsyncIOMotorClient(MONGODB_URI)
    db = client.get_default_database()

    # Get all workers
    workers = await db.workers.find({"is_active": True}).to_list(length=None)

    if not workers:
        print("No workers found! Please create workers first.")
        client.close()
        return

    print(f"Found {len(workers)} workers")

    # Get gates for realistic gate assignments
    gates = await db.gates.find({"is_active": True}).to_list(length=None)

    if not gates:
        print("No gates found! Creating default gate...")
        # Create a default gate
        mines = await db.mines.find({"is_active": True}).to_list(length=1)
        if mines:
            default_gate = {
                "name": "Main Gate",
                "gate_id": "GATE001",
                "mine_id": mines[0]["_id"],
                "gate_type": "both",
                "is_active": True,
                "has_camera": True,
                "created_at": datetime.utcnow()
            }
            result = await db.gates.insert_one(default_gate)
            default_gate["_id"] = result.inserted_id
            gates = [default_gate]
        else:
            print("No mines found! Cannot create gate.")
            client.close()
            return

    # Assign archetypes to workers
    print("\nAssigning worker archetypes...")
    worker_archetypes = {}
    archetype_counts = {"model_worker": 0, "average_worker": 0, "at_risk_worker": 0, "high_risk_worker": 0}

    for worker in workers:
        archetype = random.choices(
            list(ARCHETYPES.keys()),
            weights=[0.25, 0.50, 0.15, 0.10],  # Weights for each archetype
            k=1
        )[0]
        worker_archetypes[str(worker["_id"])] = archetype
        archetype_counts[archetype] += 1

    print(f"Archetype distribution: {archetype_counts}")

    # Generate 6 months (180 days) of data
    print("\nGenerating 180 days of historical data...")
    start_date = datetime.utcnow() - timedelta(days=180)
    end_date = datetime.utcnow()

    entries_created = 0
    violations_created = 0

    for days_offset in range((end_date - start_date).days):
        current_date = start_date + timedelta(days=days_offset)

        # Show progress every 30 days
        if days_offset % 30 == 0:
            print(f"Processing day {days_offset}/180...")

        for worker in workers:
            worker_id = str(worker["_id"])
            archetype_name = worker_archetypes[worker_id]
            archetype = ARCHETYPES[archetype_name]
            assigned_shift = worker.get("assigned_shift", "day")

            # Determine if worker shows up today (based on archetype attendance rate)
            if random.random() > archetype["attendance_rate"]:
                continue  # Worker absent today

            # Shift times
            shift_times = {
                "day": (6, 14),
                "afternoon": (14, 22),
                "night": (22, 6)
            }
            start_hour, end_hour = shift_times[assigned_shift]

            # Entry time (with some variability)
            entry_hour = start_hour + random.randint(-1, 1)  # +/- 1 hour variability
            entry_minute = random.randint(0, 59)

            entry_time = current_date.replace(
                hour=max(0, min(23, entry_hour)),
                minute=entry_minute,
                second=random.randint(0, 59)
            )

            # Determine PPE violations (based on archetype violation rate)
            ppe_items = ["helmet", "vest", "goggles", "gloves", "mask", "safety_shoes"]
            ppe_status = {}
            violations = []

            for item in ppe_items:
                # Each item has independent violation probability
                if random.random() < archetype["violation_rate"]:
                    ppe_status[item] = False
                    violations.append(f"Missing {item}")
                else:
                    ppe_status[item] = True

            # Select a random gate
            gate = random.choice(gates)

            # Create gate entry
            entry_doc = {
                "gate_id": str(gate["_id"]),
                "gate_name": gate.get("name", "Unknown Gate"),
                "worker_id": worker_id,
                "employee_id": worker.get("employee_id", ""),
                "worker_name": worker.get("name", ""),
                "mine_id": worker.get("mine_id"),
                "zone_id": worker.get("zone_id"),
                "entry_type": "entry",
                "shift": assigned_shift,
                "timestamp": entry_time,
                "ppe_status": ppe_status,
                "violations": violations,
                "status": "denied" if violations else "approved",
                "recorded_by": "system_seed",
                "created_at": entry_time
            }

            await db.gate_entries.insert_one(entry_doc)
            entries_created += 1

            if violations:
                violations_created += 1

            # Create corresponding exit entry (8 hours later with some variability)
            exit_hours_later = 8 + random.randint(-1, 1)
            exit_time = entry_time + timedelta(hours=exit_hours_later)

            exit_doc = {
                "gate_id": str(gate["_id"]),
                "gate_name": gate.get("name", "Unknown Gate"),
                "worker_id": worker_id,
                "employee_id": worker.get("employee_id", ""),
                "worker_name": worker.get("name", ""),
                "mine_id": worker.get("mine_id"),
                "zone_id": worker.get("zone_id"),
                "entry_type": "exit",
                "shift": assigned_shift,
                "timestamp": exit_time,
                "ppe_status": ppe_status,  # Same status
                "violations": [],  # No PPE check on exit
                "status": "approved",
                "recorded_by": "system_seed",
                "created_at": exit_time
            }

            await db.gate_entries.insert_one(exit_doc)
            entries_created += 1

    print(f"\nGenerated {entries_created} gate entries")
    print(f"Created {violations_created} violation entries")

    # Update worker compliance scores based on generated data
    print("\nUpdating worker compliance scores...")
    for worker in workers:
        worker_id = str(worker["_id"])

        # Count total entries and violations
        total_entries = await db.gate_entries.count_documents({
            "worker_id": worker_id,
            "entry_type": "entry"
        })

        total_violations = await db.gate_entries.count_documents({
            "worker_id": worker_id,
            "entry_type": "entry",
            "violations": {"$ne": []}
        })

        # Calculate compliance score
        if total_entries > 0:
            violation_rate = total_violations / total_entries
            compliance_score = max(0, 100 - (violation_rate * 200))  # 2x penalty
        else:
            compliance_score = 100.0

        # Assign badges
        badges = []
        if compliance_score >= 95:
            badges.append("safety_star")
        if total_violations == 0 and total_entries > 30:
            badges.append("perfect_record")

        # Get last entry
        last_entry = await db.gate_entries.find_one(
            {"worker_id": worker_id, "entry_type": "entry"},
            sort=[("timestamp", -1)]
        )

        last_entry_at = last_entry["timestamp"] if last_entry else datetime.utcnow()

        # Update worker document
        await db.workers.update_one(
            {"_id": worker["_id"]},
            {"$set": {
                "compliance_score": round(compliance_score, 1),
                "total_violations": total_violations,
                "badges": badges,
                "last_entry_at": last_entry_at
            }}
        )

    print("Worker compliance scores updated")

    # Summary stats
    print("\n" + "="*60)
    print("Data Generation Complete!")
    print("="*60)
    print(f"Total gate entries created: {entries_created}")
    print(f"Total violation entries: {violations_created}")
    print(f"Violation rate: {violations_created/entries_created*100:.1f}%")
    print(f"Workers processed: {len(workers)}")
    print(f"Time period: {start_date.date()} to {end_date.date()} (180 days)")

    client.close()


if __name__ == "__main__":
    print("Starting mock data generation for predictive analysis...")
    asyncio.run(seed_predictive_data())
    print("\nDone! You can now train the ML models.")
