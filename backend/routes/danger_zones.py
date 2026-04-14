"""
Danger Zones Routes - Gas Hazard Zone Management
Handles danger zone tracking, history, and visualization data.
"""
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from bson import ObjectId

from database import get_database
from auth import get_current_user

router = APIRouter(prefix="/api/danger-zones", tags=["Danger Zones"])


@router.post("")
async def create_danger_zone(
    zone_id: str,
    zone_name: str,
    danger_type: str,
    severity: str,
    mine_id: Optional[str] = None,
    peak_methane_ppm: Optional[float] = None,
    peak_co_ppm: Optional[float] = None,
    affected_workers: int = 0,
    evacuation_ordered: bool = False,
    notes: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Create a new danger zone incident."""
    db = get_database()

    # Get zone details
    zone = await db.zones.find_one({"_id": ObjectId(zone_id)})
    coordinates = zone.get("coordinates") if zone else {"x": 50, "y": 50, "width": 20, "height": 20}

    danger_zone = {
        "mine_id": ObjectId(mine_id) if mine_id else (zone.get("mine_id") if zone else None),
        "zone_id": ObjectId(zone_id),
        "zone_name": zone_name or (zone.get("name") if zone else "Unknown Zone"),
        "coordinates": coordinates,
        "danger_type": danger_type,
        "severity": severity,
        "detected_at": datetime.utcnow(),
        "resolved_at": None,
        "status": "active",
        "peak_methane_ppm": peak_methane_ppm,
        "peak_co_ppm": peak_co_ppm,
        "affected_workers": affected_workers,
        "evacuation_ordered": evacuation_ordered,
        "notes": notes
    }

    result = await db.danger_zones.insert_one(danger_zone)

    # Create alert
    alert = {
        "alert_type": "danger_zone",
        "severity": severity,
        "status": "active",
        "message": f"Danger zone detected: {zone_name} - {danger_type.replace('_', ' ')}",
        "mine_id": danger_zone["mine_id"],
        "zone_id": ObjectId(zone_id),
        "created_at": datetime.utcnow(),
        "danger_zone_id": result.inserted_id,
        "details": {
            "danger_type": danger_type,
            "peak_methane_ppm": peak_methane_ppm,
            "peak_co_ppm": peak_co_ppm
        }
    }
    await db.alerts.insert_one(alert)

    return {
        "success": True,
        "danger_zone_id": str(result.inserted_id),
        "message": "Danger zone created"
    }


@router.get("")
async def get_danger_zones(
    mine_id: Optional[str] = None,
    zone_id: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    danger_type: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(100, le=500),
    skip: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """Get danger zones with filters."""
    db = get_database()

    start_date = datetime.utcnow() - timedelta(days=days)
    query = {"detected_at": {"$gte": start_date}}

    if mine_id:
        query["mine_id"] = ObjectId(mine_id)
    if zone_id:
        query["zone_id"] = ObjectId(zone_id)
    if status:
        query["status"] = status
    if severity:
        query["severity"] = severity
    if danger_type:
        query["danger_type"] = danger_type

    cursor = db.danger_zones.find(query).sort("detected_at", -1).skip(skip).limit(limit)
    danger_zones = []

    async for dz in cursor:
        danger_zones.append({
            "id": str(dz["_id"]),
            "mine_id": str(dz.get("mine_id")) if dz.get("mine_id") else None,
            "zone_id": str(dz.get("zone_id")) if dz.get("zone_id") else None,
            "zone_name": dz.get("zone_name", "Unknown"),
            "coordinates": dz.get("coordinates"),
            "danger_type": dz.get("danger_type", "unknown"),
            "severity": dz.get("severity", "medium"),
            "detected_at": dz.get("detected_at").isoformat() if dz.get("detected_at") else None,
            "resolved_at": dz.get("resolved_at").isoformat() if dz.get("resolved_at") else None,
            "status": dz.get("status", "active"),
            "peak_methane_ppm": dz.get("peak_methane_ppm"),
            "peak_co_ppm": dz.get("peak_co_ppm"),
            "affected_workers": dz.get("affected_workers", 0),
            "evacuation_ordered": dz.get("evacuation_ordered", False),
            "notes": dz.get("notes")
        })

    total = await db.danger_zones.count_documents(query)

    return {
        "danger_zones": danger_zones,
        "total": total,
        "limit": limit,
        "skip": skip
    }


@router.get("/active")
async def get_active_danger_zones(
    mine_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get only active danger zones for real-time monitoring."""
    db = get_database()

    query = {"status": "active"}
    if mine_id:
        query["mine_id"] = ObjectId(mine_id)

    cursor = db.danger_zones.find(query).sort("detected_at", -1)
    danger_zones = []

    async for dz in cursor:
        danger_zones.append({
            "id": str(dz["_id"]),
            "zone_id": str(dz.get("zone_id")) if dz.get("zone_id") else None,
            "zone_name": dz.get("zone_name", "Unknown"),
            "coordinates": dz.get("coordinates"),
            "danger_type": dz.get("danger_type", "unknown"),
            "severity": dz.get("severity", "medium"),
            "detected_at": dz.get("detected_at").isoformat() if dz.get("detected_at") else None,
            "peak_methane_ppm": dz.get("peak_methane_ppm"),
            "peak_co_ppm": dz.get("peak_co_ppm"),
            "affected_workers": dz.get("affected_workers", 0),
        })

    return {"danger_zones": danger_zones, "count": len(danger_zones)}


@router.post("/{danger_zone_id}/resolve")
async def resolve_danger_zone(
    danger_zone_id: str,
    notes: Optional[str] = "Area cleared and safe",
    current_user: dict = Depends(get_current_user)
):
    """Mark a danger zone as resolved."""
    db = get_database()

    result = await db.danger_zones.update_one(
        {"_id": ObjectId(danger_zone_id)},
        {
            "$set": {
                "status": "resolved",
                "resolved_at": datetime.utcnow(),
                "notes": notes
            }
        }
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Danger zone not found")

    # Update related alert
    await db.alerts.update_one(
        {"danger_zone_id": ObjectId(danger_zone_id)},
        {
            "$set": {
                "status": "resolved",
                "resolved_at": datetime.utcnow(),
                "resolution_notes": notes
            }
        }
    )

    return {"success": True, "message": "Danger zone resolved"}


@router.get("/stats")
async def get_danger_zone_stats(
    mine_id: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(get_current_user)
):
    """Get danger zone statistics."""
    db = get_database()

    start_date = datetime.utcnow() - timedelta(days=days)
    query = {"detected_at": {"$gte": start_date}}

    if mine_id:
        query["mine_id"] = ObjectId(mine_id)

    # Get counts
    total = await db.danger_zones.count_documents(query)
    active = await db.danger_zones.count_documents({**query, "status": "active"})
    resolved = await db.danger_zones.count_documents({**query, "status": "resolved"})

    # Get counts by type
    type_counts = {}
    for dtype in ["high_methane", "high_co", "combined_gas", "ventilation_failure", "structural_risk"]:
        type_counts[dtype] = await db.danger_zones.count_documents({**query, "danger_type": dtype})

    # Get counts by severity
    severity_counts = {}
    for sev in ["medium", "high", "critical"]:
        severity_counts[sev] = await db.danger_zones.count_documents({**query, "severity": sev})

    # Calculate average duration
    resolved_zones = db.danger_zones.find({
        **query,
        "status": "resolved",
        "resolved_at": {"$ne": None}
    })

    durations = []
    async for zone in resolved_zones:
        if zone.get("detected_at") and zone.get("resolved_at"):
            delta = (zone["resolved_at"] - zone["detected_at"]).total_seconds() / 60
            durations.append(delta)

    avg_duration = sum(durations) / len(durations) if durations else 0

    # Count evacuations
    evacuations = await db.danger_zones.count_documents({**query, "evacuation_ordered": True})

    return {
        "total": total,
        "active": active,
        "resolved": resolved,
        "by_type": type_counts,
        "by_severity": severity_counts,
        "avg_duration_minutes": round(avg_duration, 1),
        "total_evacuations": evacuations,
        "time_range_days": days
    }


@router.get("/heatmap")
async def get_danger_zone_heatmap(
    mine_id: str,
    days: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(get_current_user)
):
    """Get danger zone heatmap data for visualization."""
    db = get_database()

    start_date = datetime.utcnow() - timedelta(days=days)

    pipeline = [
        {
            "$match": {
                "mine_id": ObjectId(mine_id),
                "detected_at": {"$gte": start_date}
            }
        },
        {
            "$group": {
                "_id": "$zone_id",
                "zone_name": {"$first": "$zone_name"},
                "coordinates": {"$first": "$coordinates"},
                "incident_count": {"$sum": 1},
                "critical_count": {
                    "$sum": {"$cond": [{"$eq": ["$severity", "critical"]}, 1, 0]}
                },
                "high_count": {
                    "$sum": {"$cond": [{"$eq": ["$severity", "high"]}, 1, 0]}
                },
                "avg_methane": {"$avg": "$peak_methane_ppm"},
                "avg_co": {"$avg": "$peak_co_ppm"},
                "total_affected_workers": {"$sum": "$affected_workers"},
                "evacuations": {
                    "$sum": {"$cond": ["$evacuation_ordered", 1, 0]}
                }
            }
        },
        {"$sort": {"incident_count": -1}}
    ]

    heatmap_data = []
    async for zone in db.danger_zones.aggregate(pipeline):
        risk_score = (
            (zone.get("critical_count", 0) * 3) +
            (zone.get("high_count", 0) * 2) +
            (zone.get("incident_count", 0) - zone.get("critical_count", 0) - zone.get("high_count", 0))
        )

        heatmap_data.append({
            "zone_id": str(zone["_id"]) if zone["_id"] else None,
            "zone_name": zone.get("zone_name", "Unknown"),
            "coordinates": zone.get("coordinates"),
            "incident_count": zone.get("incident_count", 0),
            "critical_count": zone.get("critical_count", 0),
            "high_count": zone.get("high_count", 0),
            "avg_methane_ppm": round(zone.get("avg_methane", 0), 2) if zone.get("avg_methane") else None,
            "avg_co_ppm": round(zone.get("avg_co", 0), 2) if zone.get("avg_co") else None,
            "total_affected_workers": zone.get("total_affected_workers", 0),
            "evacuations": zone.get("evacuations", 0),
            "risk_score": risk_score
        })

    return {"heatmap": heatmap_data, "time_range_days": days}
