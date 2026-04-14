"""
Gas Sensor Routes - Methane & CO Monitoring
Handles real-time gas sensor data and alerts.
"""
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from bson import ObjectId

from database import get_database
from auth import get_current_user, UserRole

router = APIRouter(prefix="/api/gas-sensors", tags=["Gas Sensors"])


# ==================== Gas Sensor Data ====================

@router.post("/readings")
async def create_gas_reading(
    mine_id: str,
    zone_id: Optional[str] = None,
    gate_id: Optional[str] = None,
    methane_ppm: float = 0.0,
    co_ppm: float = 0.0,
    pressure_hpa: float = 0.0,
    altitude_m: float = 0.0,
    temperature_c: Optional[float] = None,
    humidity: Optional[float] = None,
    sensor_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Create a new gas sensor reading (from IoT device)."""
    db = get_database()

    # Validate mine exists
    mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
    if not mine:
        raise HTTPException(status_code=404, detail="Mine not found")

    # Determine alert severity based on gas levels
    # OSHA standards: Methane >1% (10,000 PPM) = warning, >1.25% (12,500 PPM) = danger
    # CO: >35 PPM = warning, >50 PPM = danger
    severity = "normal"
    alert_needed = False
    alert_messages = []

    if methane_ppm > 12500:
        severity = "critical"
        alert_needed = True
        alert_messages.append(f"CRITICAL: Methane level {methane_ppm} PPM exceeds danger threshold (12,500 PPM)")
    elif methane_ppm > 10000:
        severity = "high"
        alert_needed = True
        alert_messages.append(f"WARNING: Methane level {methane_ppm} PPM exceeds safe threshold (10,000 PPM)")
    elif methane_ppm > 5000:
        severity = "medium"
        alert_needed = True
        alert_messages.append(f"CAUTION: Elevated methane detected ({methane_ppm} PPM)")

    if co_ppm > 50:
        severity = "critical" if severity != "critical" else severity
        alert_needed = True
        alert_messages.append(f"CRITICAL: CO level {co_ppm} PPM exceeds danger threshold (50 PPM)")
    elif co_ppm > 35:
        if severity == "normal":
            severity = "high"
        alert_needed = True
        alert_messages.append(f"WARNING: CO level {co_ppm} PPM exceeds safe threshold (35 PPM)")
    elif co_ppm > 25:
        if severity == "normal":
            severity = "medium"
        alert_needed = True
        alert_messages.append(f"CAUTION: Elevated CO detected ({co_ppm} PPM)")

    # Create gas reading record
    reading = {
        "mine_id": ObjectId(mine_id),
        "zone_id": ObjectId(zone_id) if zone_id else None,
        "gate_id": ObjectId(gate_id) if gate_id else None,
        "sensor_id": sensor_id,
        "methane_ppm": methane_ppm,
        "co_ppm": co_ppm,
        "pressure_hpa": pressure_hpa,
        "altitude_m": altitude_m,
        "temperature_c": temperature_c,
        "humidity": humidity,
        "severity": severity,
        "timestamp": datetime.utcnow(),
    }

    result = await db.gas_readings.insert_one(reading)

    # Create alert if gas levels are dangerous
    if alert_needed:
        alert = {
            "alert_type": "gas_level",
            "severity": severity,
            "status": "active",
            "message": " | ".join(alert_messages),
            "mine_id": ObjectId(mine_id),
            "zone_id": ObjectId(zone_id) if zone_id else None,
            "gate_id": ObjectId(gate_id) if gate_id else None,
            "gas_reading_id": result.inserted_id,
            "created_at": datetime.utcnow(),
            "details": {
                "methane_ppm": methane_ppm,
                "co_ppm": co_ppm,
                "sensor_id": sensor_id
            }
        }
        await db.alerts.insert_one(alert)

    return {
        "success": True,
        "reading_id": str(result.inserted_id),
        "severity": severity,
        "alert_created": alert_needed
    }


@router.get("/readings")
async def get_gas_readings(
    mine_id: Optional[str] = None,
    zone_id: Optional[str] = None,
    gate_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(100, le=1000),
    skip: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """Get gas sensor readings with filters."""
    db = get_database()

    # Build query
    query = {}

    if mine_id:
        query["mine_id"] = ObjectId(mine_id)

    if zone_id:
        query["zone_id"] = ObjectId(zone_id)

    if gate_id:
        query["gate_id"] = ObjectId(gate_id)

    if severity:
        query["severity"] = severity

    if start_date or end_date:
        time_query = {}
        if start_date:
            time_query["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if end_date:
            time_query["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        query["timestamp"] = time_query

    # Get readings
    cursor = db.gas_readings.find(query).sort("timestamp", -1).skip(skip).limit(limit)
    readings = []

    async for reading in cursor:
        readings.append({
            "id": str(reading["_id"]),
            "mine_id": str(reading["mine_id"]),
            "zone_id": str(reading["zone_id"]) if reading.get("zone_id") else None,
            "gate_id": str(reading["gate_id"]) if reading.get("gate_id") else None,
            "sensor_id": reading.get("sensor_id"),
            "methane_ppm": reading["methane_ppm"],
            "co_ppm": reading["co_ppm"],
            "pressure_hpa": reading["pressure_hpa"],
            "altitude_m": reading["altitude_m"],
            "temperature_c": reading.get("temperature_c"),
            "humidity": reading.get("humidity"),
            "severity": reading["severity"],
            "timestamp": reading["timestamp"].isoformat()
        })

    total = await db.gas_readings.count_documents(query)

    return {
        "readings": readings,
        "total": total,
        "limit": limit,
        "skip": skip
    }


@router.get("/latest")
async def get_latest_readings(
    mine_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get the latest gas reading for each zone/gate."""
    db = get_database()

    query = {}
    if mine_id:
        query["mine_id"] = ObjectId(mine_id)

    # Get latest readings per location
    pipeline = [
        {"$match": query} if query else {"$match": {}},
        {"$sort": {"timestamp": -1}},
        {
            "$group": {
                "_id": {
                    "mine_id": "$mine_id",
                    "zone_id": "$zone_id",
                    "gate_id": "$gate_id",
                    "sensor_id": "$sensor_id"
                },
                "latest": {"$first": "$$ROOT"}
            }
        },
        {"$replaceRoot": {"newRoot": "$latest"}}
    ]

    readings = []
    async for reading in db.gas_readings.aggregate(pipeline):
        # Get location names
        mine = await db.mines.find_one({"_id": reading["mine_id"]})
        zone = None
        gate = None

        if reading.get("zone_id"):
            zone = await db.zones.find_one({"_id": reading["zone_id"]})
        if reading.get("gate_id"):
            gate = await db.gates.find_one({"_id": reading["gate_id"]})

        readings.append({
            "id": str(reading["_id"]),
            "mine_id": str(reading["mine_id"]),
            "mine_name": mine["name"] if mine else "Unknown",
            "zone_id": str(reading["zone_id"]) if reading.get("zone_id") else None,
            "zone_name": zone["name"] if zone else None,
            "gate_id": str(reading["gate_id"]) if reading.get("gate_id") else None,
            "gate_name": gate["name"] if gate else None,
            "sensor_id": reading.get("sensor_id"),
            "methane_ppm": reading["methane_ppm"],
            "co_ppm": reading["co_ppm"],
            "pressure_hpa": reading["pressure_hpa"],
            "altitude_m": reading["altitude_m"],
            "temperature_c": reading.get("temperature_c"),
            "humidity": reading.get("humidity"),
            "severity": reading["severity"],
            "timestamp": reading["timestamp"].isoformat()
        })

    return {"readings": readings}


@router.get("/stats")
async def get_gas_stats(
    mine_id: Optional[str] = None,
    hours: int = Query(24, ge=1, le=168),  # Last N hours (max 1 week)
    current_user: dict = Depends(get_current_user)
):
    """Get gas monitoring statistics."""
    db = get_database()

    start_time = datetime.utcnow() - timedelta(hours=hours)

    query = {"timestamp": {"$gte": start_time}}
    if mine_id:
        query["mine_id"] = ObjectId(mine_id)

    # Get all readings in time range
    cursor = db.gas_readings.find(query)

    methane_values = []
    co_values = []
    severity_counts = {"normal": 0, "medium": 0, "high": 0, "critical": 0}
    total_alerts = 0

    async for reading in cursor:
        methane_values.append(reading["methane_ppm"])
        co_values.append(reading["co_ppm"])
        severity = reading.get("severity", "normal")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        if severity in ["medium", "high", "critical"]:
            total_alerts += 1

    total_readings = len(methane_values)

    stats = {
        "total_readings": total_readings,
        "time_range_hours": hours,
        "methane": {
            "current": methane_values[0] if methane_values else 0,
            "avg": sum(methane_values) / len(methane_values) if methane_values else 0,
            "max": max(methane_values) if methane_values else 0,
            "min": min(methane_values) if methane_values else 0,
        },
        "co": {
            "current": co_values[0] if co_values else 0,
            "avg": sum(co_values) / len(co_values) if co_values else 0,
            "max": max(co_values) if co_values else 0,
            "min": min(co_values) if co_values else 0,
        },
        "severity_distribution": severity_counts,
        "total_alerts": total_alerts,
        "is_safe": severity_counts.get("critical", 0) == 0 and severity_counts.get("high", 0) == 0
    }

    return stats


@router.get("/alerts")
async def get_gas_alerts(
    mine_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(50, le=200),
    current_user: dict = Depends(get_current_user)
):
    """Get gas-related alerts."""
    db = get_database()

    query = {"alert_type": "gas_level"}
    if mine_id:
        query["mine_id"] = ObjectId(mine_id)
    if status:
        query["status"] = status

    cursor = db.alerts.find(query).sort("created_at", -1).limit(limit)
    alerts = []

    async for alert in cursor:
        # Get location names
        mine = await db.mines.find_one({"_id": alert["mine_id"]})
        zone = None
        gate = None

        if alert.get("zone_id"):
            zone = await db.zones.find_one({"_id": alert["zone_id"]})
        if alert.get("gate_id"):
            gate = await db.gates.find_one({"_id": alert["gate_id"]})

        alerts.append({
            "id": str(alert["_id"]),
            "severity": alert["severity"],
            "status": alert["status"],
            "message": alert["message"],
            "mine_name": mine["name"] if mine else "Unknown",
            "zone_name": zone["name"] if zone else None,
            "gate_name": gate["name"] if gate else None,
            "details": alert.get("details", {}),
            "created_at": alert["created_at"].isoformat(),
            "acknowledged_at": alert.get("acknowledged_at").isoformat() if alert.get("acknowledged_at") else None,
            "acknowledged_by": alert.get("acknowledged_by"),
        })

    return {"alerts": alerts}
