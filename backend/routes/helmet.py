"""
Helmet Sensor Routes - Smart Helmet Monitoring
Handles real-time helmet sensor data for worker safety monitoring.
"""
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from bson import ObjectId

from database import get_database
from auth import get_current_user
from services.helmet_service import get_latest_helmet_data, get_all_helmet_data, DEMO_WORKER_NAMES

router = APIRouter(prefix="/api/helmet", tags=["Helmet Sensors"])


# ==================== Raw IMU Data for 3D Tracking ====================

@router.get("/imu-data")
async def get_imu_data():
    """
    Get raw IMU data for 3D position tracking.
    Returns accelerometer, gyroscope, and orientation data.
    No authentication required for real-time tracking.
    """
    helmet_data = get_all_helmet_data()

    if not helmet_data:
        return {}

    # Get first helmet's data (worker_1)
    data = helmet_data[0] if helmet_data else {}

    if not data:
        return {}

    # Return in format expected by useHelmetTracking hook
    return {
        "Accel X (mg)": data.get("accel_x", 0),
        "Accel Y (mg)": data.get("accel_y", 0),
        "Accel Z (mg)": data.get("accel_z", 1000),
        "Gyro X (mdps)": data.get("gyro_x", 0),
        "Gyro Y (mdps)": data.get("gyro_y", 0),
        "Gyro Z (mdps)": data.get("gyro_z", 0),
        "Roll (Deg)": data.get("roll", 0),
        "Pitch (Deg)": data.get("pitch", 0),
        "Yaw (Deg)": data.get("yaw", 0),
        "Timestamp (ms)": data.get("timestamp_ms", 0),
    }


# ==================== Latest Data ====================

@router.get("/latest")
async def get_latest_readings(
    mine_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Get the latest readings from all active helmets.
    Returns real-time data from the serial reader.
    """
    db = get_database()
    helmet_data = get_all_helmet_data()

    if not helmet_data:
        return {"readings": [], "message": "No helmet data available"}

    # Enrich with worker information
    enriched_data = []
    for data in helmet_data:
        worker_id = data.get("worker_id")
        worker = None

        if worker_id:
            # Try to find worker by employee_id
            worker = await db.workers.find_one({"employee_id": worker_id})
            if not worker:
                worker = await db.workers.find_one({"_id": ObjectId(worker_id)} if ObjectId.is_valid(worker_id) else {})

        # Use demo worker name as fallback
        worker_name = "Unknown"
        if worker:
            worker_name = worker.get("name", "Unknown")
        elif worker_id in DEMO_WORKER_NAMES:
            worker_name = DEMO_WORKER_NAMES[worker_id]

        reading = {
            **data,
            "worker_name": worker_name,
            "mine_id": str(worker.get("mine_id", "")) if worker else None,
            "zone_id": str(worker.get("zone_id", "")) if worker else None,
            "timestamp": data["timestamp"].isoformat() if isinstance(data.get("timestamp"), datetime) else data.get("timestamp"),
        }
        enriched_data.append(reading)

    return {"readings": enriched_data}


@router.get("/latest/{worker_id}")
async def get_latest_by_worker(
    worker_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get the latest helmet reading for a specific worker.
    """
    db = get_database()
    data = get_latest_helmet_data(worker_id)

    if not data:
        raise HTTPException(status_code=404, detail=f"No helmet data for worker {worker_id}")

    # Get worker info
    worker = await db.workers.find_one({"employee_id": worker_id})
    if not worker:
        worker = await db.workers.find_one({"_id": ObjectId(worker_id)} if ObjectId.is_valid(worker_id) else {})

    # Use demo worker name as fallback
    worker_name = "Unknown"
    if worker:
        worker_name = worker.get("name", "Unknown")
    elif worker_id in DEMO_WORKER_NAMES:
        worker_name = DEMO_WORKER_NAMES[worker_id]

    return {
        **data,
        "worker_name": worker_name,
        "mine_id": str(worker.get("mine_id", "")) if worker else None,
        "zone_id": str(worker.get("zone_id", "")) if worker else None,
        "timestamp": data["timestamp"].isoformat() if isinstance(data.get("timestamp"), datetime) else data.get("timestamp"),
    }


# ==================== Historical Readings ====================

@router.get("/readings")
async def get_helmet_readings(
    worker_id: Optional[str] = None,
    mine_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(100, le=1000),
    skip: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """
    Get historical helmet sensor readings from database.
    Note: Only readings that exceeded thresholds are stored.
    """
    db = get_database()

    query = {}

    if worker_id:
        query["worker_id"] = worker_id

    if mine_id:
        query["mine_id"] = ObjectId(mine_id)

    if severity:
        query["severity"] = severity

    if start_date or end_date:
        time_query = {}
        if start_date:
            time_query["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if end_date:
            time_query["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        query["timestamp"] = time_query

    cursor = db.helmet_readings.find(query).sort("timestamp", -1).skip(skip).limit(limit)
    readings = []

    async for reading in cursor:
        # Get worker info
        w_id = reading.get("worker_id")
        worker = await db.workers.find_one({"employee_id": w_id})

        # Use demo worker name as fallback
        worker_name = "Unknown"
        if worker:
            worker_name = worker.get("name", "Unknown")
        elif w_id in DEMO_WORKER_NAMES:
            worker_name = DEMO_WORKER_NAMES[w_id]

        readings.append({
            "id": str(reading["_id"]),
            "worker_id": w_id,
            "worker_name": worker_name,
            "methane_ppm": reading.get("methane_ppm"),
            "co_raw": reading.get("co_raw"),
            "battery_voltage": reading.get("battery_voltage"),
            "battery_low": reading.get("battery_low"),
            "heart_rate": reading.get("heart_rate"),
            "spo2": reading.get("spo2"),
            "sos_active": reading.get("sos_active"),
            "severity": reading.get("severity"),
            "timestamp": reading["timestamp"].isoformat() if reading.get("timestamp") else None,
        })

    total = await db.helmet_readings.count_documents(query)

    return {
        "readings": readings,
        "total": total,
        "limit": limit,
        "skip": skip
    }


# ==================== Statistics ====================

@router.get("/stats")
async def get_helmet_stats(
    mine_id: Optional[str] = None,
    hours: int = Query(24, ge=1, le=168),
    current_user: dict = Depends(get_current_user)
):
    """
    Get helmet monitoring statistics.
    """
    db = get_database()

    # Get real-time stats from current helmet data
    helmet_data = get_all_helmet_data()

    active_helmets = len(helmet_data)
    total_workers = await db.workers.count_documents({"is_active": True})

    # Calculate averages from live data
    avg_battery = 0
    avg_heart_rate = 0
    avg_spo2 = 0
    low_battery_count = 0
    sos_active_count = 0
    critical_count = 0
    high_count = 0

    if helmet_data:
        batteries = [d.get("battery_voltage", 0) for d in helmet_data]
        heart_rates = [d.get("heart_rate", 0) for d in helmet_data]
        spo2s = [d.get("spo2", 0) for d in helmet_data]

        avg_battery = round(sum(batteries) / len(batteries), 2) if batteries else 0
        avg_heart_rate = round(sum(heart_rates) / len(heart_rates)) if heart_rates else 0
        avg_spo2 = round(sum(spo2s) / len(spo2s), 1) if spo2s else 0

        for d in helmet_data:
            if d.get("battery_low"):
                low_battery_count += 1
            if d.get("sos_active"):
                sos_active_count += 1
            if d.get("severity") == "critical":
                critical_count += 1
            elif d.get("severity") == "high":
                high_count += 1

    # Get historical alert counts
    start_time = datetime.utcnow() - timedelta(hours=hours)
    query = {"timestamp": {"$gte": start_time}}

    alerts_in_period = await db.helmet_readings.count_documents({
        **query,
        "severity": {"$in": ["medium", "high", "critical"]}
    })

    # Severity distribution from historical data
    severity_pipeline = [
        {"$match": query},
        {"$group": {"_id": "$severity", "count": {"$sum": 1}}}
    ]
    severity_counts = {"normal": 0, "medium": 0, "high": 0, "critical": 0}
    async for doc in db.helmet_readings.aggregate(severity_pipeline):
        severity_counts[doc["_id"]] = doc["count"]

    return {
        "active_helmets": active_helmets,
        "total_workers": total_workers,
        "coverage_percent": round((active_helmets / total_workers * 100) if total_workers > 0 else 0, 1),
        "avg_battery_voltage": avg_battery,
        "avg_heart_rate": avg_heart_rate,
        "avg_spo2": avg_spo2,
        "low_battery_count": low_battery_count,
        "sos_active_count": sos_active_count,
        "critical_alerts": critical_count,
        "high_alerts": high_count,
        "alerts_in_period": alerts_in_period,
        "time_range_hours": hours,
        "severity_distribution": severity_counts,
        "is_safe": critical_count == 0 and sos_active_count == 0
    }


# ==================== Alerts ====================

@router.get("/alerts")
async def get_helmet_alerts(
    worker_id: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(50, le=200),
    current_user: dict = Depends(get_current_user)
):
    """
    Get helmet-related alerts.
    """
    db = get_database()

    query = {"alert_type": "helmet_sensor"}

    if worker_id:
        query["worker_id"] = worker_id

    if status:
        query["status"] = status

    if severity:
        query["severity"] = severity

    cursor = db.alerts.find(query).sort("created_at", -1).limit(limit)
    alerts = []

    async for alert in cursor:
        # Get worker info
        w_id = alert.get("worker_id")
        worker = None
        if w_id:
            worker = await db.workers.find_one({"employee_id": w_id})

        # Use demo worker name as fallback
        worker_name = "Unknown"
        if worker:
            worker_name = worker.get("name", "Unknown")
        elif w_id in DEMO_WORKER_NAMES:
            worker_name = DEMO_WORKER_NAMES[w_id]

        alerts.append({
            "id": str(alert["_id"]),
            "severity": alert["severity"],
            "status": alert["status"],
            "message": alert["message"],
            "worker_id": w_id,
            "worker_name": worker_name,
            "details": alert.get("details", {}),
            "created_at": alert["created_at"].isoformat(),
            "acknowledged_at": alert.get("acknowledged_at").isoformat() if alert.get("acknowledged_at") else None,
            "acknowledged_by": alert.get("acknowledged_by"),
        })

    return {"alerts": alerts}


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_helmet_alert(
    alert_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Acknowledge a helmet alert.
    """
    db = get_database()

    result = await db.alerts.update_one(
        {"_id": ObjectId(alert_id), "alert_type": "helmet_sensor"},
        {
            "$set": {
                "status": "acknowledged",
                "acknowledged_at": datetime.utcnow(),
                "acknowledged_by": current_user.get("username", current_user.get("employee_id", "unknown"))
            }
        }
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"success": True, "message": "Alert acknowledged"}


@router.post("/alerts/{alert_id}/resolve")
async def resolve_helmet_alert(
    alert_id: str,
    resolution_note: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Resolve a helmet alert.
    """
    db = get_database()

    update_data = {
        "status": "resolved",
        "resolved_at": datetime.utcnow(),
        "resolved_by": current_user.get("username", current_user.get("employee_id", "unknown"))
    }

    if resolution_note:
        update_data["resolution_note"] = resolution_note

    result = await db.alerts.update_one(
        {"_id": ObjectId(alert_id), "alert_type": "helmet_sensor"},
        {"$set": update_data}
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"success": True, "message": "Alert resolved"}
