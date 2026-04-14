
import asyncio, os, random
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/sih_safety_system")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

PPE_ITEMS = ["helmet", "vest", "gloves", "boots", "goggles", "mask"]
DEPARTMENTS = ["Mining", "Excavation", "Transport", "Maintenance", "Safety", "Operations"]
SHIFTS = ["day", "afternoon", "night"]

async def seed():
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client.get_default_database()
    
    print("Clearing data...")
    for coll in ["users", "workers", "mines", "zones", "gates", "gate_entries", "alerts", "warnings", "gas_readings"]:
        await db[coll].delete_many({})
    
    print("Creating 3 mines...")
    mines, zones_by_mine, gates_by_mine = [], {}, {}
    
    for i in range(1, 4):
        mine = await db.mines.insert_one({
            "name": f"Mine {i}", "location": f"Region {i}",
            "is_active": True, "created_at": datetime.utcnow()
        })
        mines.append(mine.inserted_id)
        
        zones = []
        for j in range(1, 5):
            zone = await db.zones.insert_one({
                "mine_id": mine.inserted_id, "name": f"Zone {chr(64+j)}",
                "risk_level": "high" if j <= 2 else "medium", "created_at": datetime.utcnow()
            })
            zones.append(zone.inserted_id)
        zones_by_mine[mine.inserted_id] = zones
        
        gates = []
        for k in range(1, 7):
            gate = await db.gates.insert_one({
                "mine_id": mine.inserted_id, "zone_id": zones[(k-1)//2],
                "name": f"Gate {k}", "gate_type": "entry" if k%2==1 else "exit",
                "has_camera": True, "is_active": True, "created_at": datetime.utcnow()
            })
            gates.append(gate.inserted_id)
        gates_by_mine[mine.inserted_id] = gates
    
    print("Creating users...")
    users = {}
    
    await db.users.insert_one({
        "username": "superadmin", "password_hash": pwd_context.hash("admin123"),
        "full_name": "System Administrator", "role": "super_admin",
        "is_active": True, "created_at": datetime.utcnow()
    })
    
    await db.users.insert_one({
        "username": "gm", "password_hash": pwd_context.hash("gm123"),
        "full_name": "General Manager", "role": "general_manager",
        "mine_ids": [str(m) for m in mines], "is_active": True, "created_at": datetime.utcnow()
    })
    
    for i, mine_id in enumerate(mines, 1):
        await db.users.insert_one({
            "username": f"manager{i}", "password_hash": pwd_context.hash("manager123"),
            "full_name": f"Manager {i}", "role": "manager",
            "mine_id": str(mine_id), "is_active": True, "created_at": datetime.utcnow()
        })
        
        await db.users.insert_one({
            "username": f"safety{i}", "password_hash": pwd_context.hash("safety123"),
            "full_name": f"Safety Officer {i}", "role": "safety_officer",
            "mine_id": str(mine_id), "is_active": True, "created_at": datetime.utcnow()
        })
        
        for shift in SHIFTS:
            await db.users.insert_one({
                "username": f"shift_{shift}{i}", "password_hash": pwd_context.hash("shift123"),
                "full_name": f"Shift {shift} {i}", "role": "shift_incharge",
                "mine_id": str(mine_id), "assigned_shift": shift,
                "is_active": True, "created_at": datetime.utcnow()
            })
    
    print("Creating 60 workers...")
    workers = []
    wid = 1
    for mine_id in mines:
        zones = zones_by_mine[mine_id]
        for j in range(20):
            w = await db.workers.insert_one({
                "employee_id": f"W{wid:03d}", "password_hash": pwd_context.hash("worker123"),
                "name": f"Worker {wid}", "department": DEPARTMENTS[j%6],
                "mine_id": mine_id, "zone_id": zones[j%4],
                "assigned_shift": SHIFTS[j%3], "is_active": True,
                "compliance_score": round(random.uniform(75,100),1),
                "total_violations": random.randint(0,15),
                "created_at": datetime.utcnow()
            })
            workers.append(w.inserted_id)
            wid += 1
    
    print("Creating gate entries, alerts, gas readings...")
    entries, alerts, gas = 0, 0, 0
    
    for days in range(30):
        date = datetime.utcnow() - timedelta(days=days)
        for _ in range(random.randint(10,30)):
            mine_id = random.choice(mines)
            await db.gate_entries.insert_one({
                "gate_id": random.choice(gates_by_mine[mine_id]),
                "worker_id": random.choice(workers),
                "mine_id": mine_id,
                "entry_type": random.choice(["entry","exit"]),
                "shift": random.choice(SHIFTS),
                "timestamp": date.replace(hour=random.randint(6,22)),
                "status": "approved",
                "violations": [],
                "detected_ppe": {i: True for i in PPE_ITEMS},
                "created_at": date
            })
            entries += 1
        
        for _ in range(3):
            mine_id = random.choice(mines)
            await db.alerts.insert_one({
                "mine_id": mine_id,
                "zone_id": random.choice(zones_by_mine[mine_id]),
                "alert_type": random.choice(["ppe_violation","safety_concern"]),
                "severity": random.choice(["low","medium","high"]),
                "message": "Safety alert",
                "status": "resolved" if random.random()<0.7 else "active",
                "created_at": date
            })
            alerts += 1
    
    for days in range(7):
        date = datetime.utcnow() - timedelta(days=days)
        for hour in range(24):
            for mine_id in mines:
                await db.gas_readings.insert_one({
                    "mine_id": mine_id,
                    "zone_id": random.choice(zones_by_mine[mine_id]),
                    "methane_ppm": round(random.uniform(0,10),2),
                    "co_ppm": round(random.uniform(0,30),2),
                    "severity": "normal",
                    "timestamp": date.replace(hour=hour),
                    "created_at": date.replace(hour=hour)
                })
                gas += 1
    
    print(f"DONE! Mines:3 Users:15 Workers:60 Entries:{entries} Alerts:{alerts} Gas:{gas}")
    client.close()

if __name__ == "__main__":
    asyncio.run(seed())
