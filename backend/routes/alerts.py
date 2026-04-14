"""
Alerts and warnings management routes.
"""
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Depends, Query, Form
from bson import ObjectId
from database import get_database
from auth import (
    get_current_user, get_shift_incharge_or_above, get_safety_officer_or_above,
    UserRole, check_mine_access
)
from schemas import (
    AlertCreate, AlertResponse, AlertList, AlertSeverity, AlertStatus,
    WarningCreate, WarningResponse
)

router = APIRouter(prefix="/alerts", tags=["Alerts & Warnings"])


# ==================== Alert Endpoints ====================

@router.post("", response_model=AlertResponse)
async def create_alert(
    alert_data: AlertCreate,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Create a new alert manually."""
    db = get_database()

    # Check mine access
    if not check_mine_access(current_user, alert_data.mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    # Get mine name
    mine_name = None
    try:
        mine = await db.mines.find_one({"_id": ObjectId(alert_data.mine_id)})
        if mine:
            mine_name = mine["name"]
    except:
        pass

    # Get worker name if provided
    worker_name = None
    if alert_data.worker_id:
        worker = await db.workers.find_one({"_id": ObjectId(alert_data.worker_id)})
        if worker:
            worker_name = worker["name"]

    alert_doc = {
        "alert_type": alert_data.alert_type,
        "severity": alert_data.severity.value,
        "status": AlertStatus.ACTIVE.value,
        "message": alert_data.message,
        "mine_id": ObjectId(alert_data.mine_id),
        "zone_id": ObjectId(alert_data.zone_id) if alert_data.zone_id else None,
        "gate_id": alert_data.gate_id,
        "worker_id": alert_data.worker_id,
        "metadata": alert_data.metadata or {},
        "created_at": datetime.utcnow(),
        "created_by": current_user.get("user_id") or current_user.get("sub"),
    }

    result = await db.alerts.insert_one(alert_doc)

    return AlertResponse(
        id=str(result.inserted_id),
        alert_type=alert_doc["alert_type"],
        severity=alert_data.severity,
        status=AlertStatus.ACTIVE,
        message=alert_doc["message"],
        mine_id=alert_data.mine_id,
        mine_name=mine_name,
        zone_id=alert_data.zone_id,
        gate_id=alert_data.gate_id,
        worker_id=alert_data.worker_id,
        worker_name=worker_name,
        created_at=alert_doc["created_at"]
    )


@router.get("", response_model=AlertList)
async def list_alerts(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    mine_id: Optional[str] = None,
    status: Optional[AlertStatus] = None,
    severity: Optional[AlertSeverity] = None,
    alert_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """List alerts with filters."""
    db = get_database()

    query = {}

    # Filter by mine access
    user_role = UserRole(current_user.get("role"))

    if user_role in [UserRole.SUPER_ADMIN, UserRole.GENERAL_MANAGER]:
        pass
    elif user_role == UserRole.AREA_SAFETY_OFFICER:
        mine_ids = current_user.get("mine_ids", [])
        query["mine_id"] = {"$in": [ObjectId(mid) for mid in mine_ids]}
    else:
        user_mine_id = current_user.get("mine_id")
        if user_mine_id:
            query["mine_id"] = ObjectId(user_mine_id)

    # Apply filters
    if mine_id:
        if not check_mine_access(current_user, mine_id):
            raise HTTPException(status_code=403, detail="No access to this mine")
        query["mine_id"] = ObjectId(mine_id)

    if status:
        query["status"] = status.value

    if severity:
        query["severity"] = severity.value

    if alert_type:
        query["alert_type"] = alert_type

    if start_date or end_date:
        query["created_at"] = {}
        if start_date:
            query["created_at"]["$gte"] = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            query["created_at"]["$lt"] = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    cursor = db.alerts.find(query).skip(skip).limit(limit).sort("created_at", -1)
    alerts = []

    async for alert in cursor:
        # Get mine name
        mine_name = None
        if alert.get("mine_id"):
            mine = await db.mines.find_one({"_id": alert["mine_id"]})
            if mine:
                mine_name = mine["name"]

        # Get worker name
        worker_name = None
        if alert.get("worker_id"):
            worker = await db.workers.find_one({"_id": ObjectId(alert["worker_id"])})
            if worker:
                worker_name = worker["name"]

        alerts.append(AlertResponse(
            id=str(alert["_id"]),
            alert_type=alert["alert_type"],
            severity=AlertSeverity(alert["severity"]),
            status=AlertStatus(alert["status"]),
            message=alert["message"],
            mine_id=str(alert["mine_id"]),
            mine_name=mine_name,
            zone_id=str(alert["zone_id"]) if alert.get("zone_id") else None,
            gate_id=alert.get("gate_id"),
            worker_id=alert.get("worker_id"),
            worker_name=worker_name,
            created_at=alert["created_at"],
            acknowledged_by=alert.get("acknowledged_by"),
            acknowledged_at=alert.get("acknowledged_at"),
            resolved_by=alert.get("resolved_by"),
            resolved_at=alert.get("resolved_at"),
            resolution_notes=alert.get("resolution_notes")
        ))

    total = await db.alerts.count_documents(query)

    return AlertList(alerts=alerts, total=total)


@router.get("/active")
async def get_active_alerts(
    mine_id: Optional[str] = None,
    severity: Optional[AlertSeverity] = None,
    limit: int = Query(20, ge=1, le=50),
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Get active (unresolved) alerts for dashboard."""
    db = get_database()

    query = {"status": {"$in": [AlertStatus.ACTIVE.value, AlertStatus.ACKNOWLEDGED.value]}}

    # Filter by mine access
    user_role = UserRole(current_user.get("role"))

    if user_role in [UserRole.SUPER_ADMIN, UserRole.GENERAL_MANAGER]:
        pass
    elif user_role == UserRole.AREA_SAFETY_OFFICER:
        mine_ids = current_user.get("mine_ids", [])
        query["mine_id"] = {"$in": [ObjectId(mid) for mid in mine_ids]}
    else:
        user_mine_id = current_user.get("mine_id")
        if user_mine_id:
            query["mine_id"] = ObjectId(user_mine_id)

    if mine_id:
        if not check_mine_access(current_user, mine_id):
            raise HTTPException(status_code=403, detail="No access to this mine")
        query["mine_id"] = ObjectId(mine_id)

    if severity:
        query["severity"] = severity.value

    cursor = db.alerts.find(query).limit(limit).sort([
        ("severity", -1),  # Critical first
        ("created_at", -1)
    ])

    alerts = []
    async for alert in cursor:
        mine_name = None
        if alert.get("mine_id"):
            mine = await db.mines.find_one({"_id": alert["mine_id"]})
            if mine:
                mine_name = mine["name"]

        worker_name = None
        if alert.get("worker_id"):
            worker = await db.workers.find_one({"_id": ObjectId(alert["worker_id"])})
            if worker:
                worker_name = worker["name"]

        alerts.append({
            "id": str(alert["_id"]),
            "alert_type": alert["alert_type"],
            "severity": alert["severity"],
            "status": alert["status"],
            "message": alert["message"],
            "mine_id": str(alert["mine_id"]),
            "mine_name": mine_name,
            "worker_id": alert.get("worker_id"),
            "worker_name": worker_name,
            "created_at": alert["created_at"].isoformat(),
            "metadata": alert.get("metadata", {}),
        })

    # Get counts by severity
    severity_counts = {}
    for sev in AlertSeverity:
        count = await db.alerts.count_documents({
            **query,
            "severity": sev.value
        })
        severity_counts[sev.value] = count

    return {
        "alerts": alerts,
        "total": len(alerts),
        "by_severity": severity_counts
    }


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Acknowledge an alert."""
    db = get_database()

    try:
        alert = await db.alerts.find_one({"_id": ObjectId(alert_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid alert ID")

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    # Check mine access
    mine_id = str(alert["mine_id"])
    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this alert")

    await db.alerts.update_one(
        {"_id": ObjectId(alert_id)},
        {"$set": {
            "status": AlertStatus.ACKNOWLEDGED.value,
            "acknowledged_by": current_user.get("user_id") or current_user.get("sub"),
            "acknowledged_at": datetime.utcnow()
        }}
    )

    return {"success": True, "message": "Alert acknowledged"}


@router.post("/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolution_notes: str = Form(...),
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Resolve an alert with notes."""
    db = get_database()

    try:
        alert = await db.alerts.find_one({"_id": ObjectId(alert_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid alert ID")

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    # Check mine access
    mine_id = str(alert["mine_id"])
    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this alert")

    await db.alerts.update_one(
        {"_id": ObjectId(alert_id)},
        {"$set": {
            "status": AlertStatus.RESOLVED.value,
            "resolved_by": current_user.get("user_id") or current_user.get("sub"),
            "resolved_at": datetime.utcnow(),
            "resolution_notes": resolution_notes
        }}
    )

    return {"success": True, "message": "Alert resolved"}


# ==================== Emergency Endpoints ====================

@router.post("/emergency")
async def trigger_emergency(
    mine_id: str = Form(...),
    emergency_type: str = Form(...),
    description: str = Form(...),
    zone_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Trigger an emergency alert (evacuation, accident, etc.)."""
    db = get_database()

    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    # Get mine info
    mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
    if not mine:
        raise HTTPException(status_code=404, detail="Mine not found")

    alert_doc = {
        "alert_type": f"emergency_{emergency_type}",
        "severity": AlertSeverity.CRITICAL.value,
        "status": AlertStatus.ACTIVE.value,
        "message": f"EMERGENCY: {emergency_type.upper()} - {description}",
        "mine_id": ObjectId(mine_id),
        "zone_id": ObjectId(zone_id) if zone_id else None,
        "metadata": {
            "emergency_type": emergency_type,
            "triggered_by": current_user.get("sub"),
            "triggered_at": datetime.utcnow().isoformat()
        },
        "created_at": datetime.utcnow(),
        "created_by": current_user.get("user_id") or current_user.get("sub"),
    }

    result = await db.alerts.insert_one(alert_doc)

    return {
        "success": True,
        "message": "Emergency alert triggered",
        "alert_id": str(result.inserted_id),
        "mine_name": mine["name"]
    }


# ==================== Warning Endpoints ====================

@router.post("/warnings", response_model=WarningResponse)
async def issue_warning(
    warning_data: WarningCreate,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Issue a warning to a worker."""
    db = get_database()

    # Get worker
    try:
        worker = await db.workers.find_one({"_id": ObjectId(warning_data.worker_id)})
    except:
        worker = await db.workers.find_one({"employee_id": warning_data.worker_id})

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Check mine access
    worker_mine_id = str(worker["mine_id"]) if worker.get("mine_id") else None
    if worker_mine_id and not check_mine_access(current_user, worker_mine_id):
        raise HTTPException(status_code=403, detail="No access to this worker")

    # Get issuer name
    issuer_name = current_user.get("sub")
    if current_user.get("user_type") != "worker":
        user = await db.users.find_one({"username": current_user.get("sub")})
        if user:
            issuer_name = user.get("full_name", current_user.get("sub"))

    warning_doc = {
        "worker_id": str(worker["_id"]),
        "employee_id": worker["employee_id"],
        "worker_name": worker["name"],
        "warning_type": warning_data.warning_type,
        "description": warning_data.description,
        "severity": warning_data.severity,
        "issued_by": current_user.get("user_id") or current_user.get("sub"),
        "issued_by_name": issuer_name,
        "issued_at": datetime.utcnow(),
        "acknowledged": False,
    }

    result = await db.warnings.insert_one(warning_doc)

    return WarningResponse(
        id=str(result.inserted_id),
        worker_id=warning_doc["worker_id"],
        worker_name=warning_doc["worker_name"],
        employee_id=warning_doc["employee_id"],
        warning_type=warning_doc["warning_type"],
        description=warning_doc["description"],
        severity=warning_doc["severity"],
        issued_by=warning_doc["issued_by"],
        issued_by_name=warning_doc["issued_by_name"],
        issued_at=warning_doc["issued_at"]
    )


@router.get("/warnings")
async def list_warnings(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    worker_id: Optional[str] = None,
    warning_type: Optional[str] = None,
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """List warnings with filters."""
    db = get_database()

    query = {}

    if worker_id:
        query["worker_id"] = worker_id

    if warning_type:
        query["warning_type"] = warning_type

    if severity:
        query["severity"] = severity

    if acknowledged is not None:
        query["acknowledged"] = acknowledged

    cursor = db.warnings.find(query).skip(skip).limit(limit).sort("issued_at", -1)
    warnings = []

    async for warning in cursor:
        warnings.append({
            "id": str(warning["_id"]),
            "worker_id": warning["worker_id"],
            "worker_name": warning["worker_name"],
            "employee_id": warning["employee_id"],
            "warning_type": warning["warning_type"],
            "description": warning["description"],
            "severity": warning["severity"],
            "issued_by": warning["issued_by"],
            "issued_by_name": warning["issued_by_name"],
            "issued_at": warning["issued_at"].isoformat(),
            "acknowledged": warning.get("acknowledged", False),
            "acknowledged_at": warning.get("acknowledged_at").isoformat() if warning.get("acknowledged_at") else None,
        })

    total = await db.warnings.count_documents(query)

    return {"warnings": warnings, "total": total}


@router.post("/warnings/{warning_id}/acknowledge")
async def acknowledge_warning(
    warning_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Acknowledge a warning (by worker or supervisor)."""
    db = get_database()

    try:
        warning = await db.warnings.find_one({"_id": ObjectId(warning_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid warning ID")

    if not warning:
        raise HTTPException(status_code=404, detail="Warning not found")

    # Workers can only acknowledge their own warnings
    if current_user.get("user_type") == "worker":
        if warning["worker_id"] != current_user.get("worker_id"):
            raise HTTPException(status_code=403, detail="Cannot acknowledge other worker's warnings")

    await db.warnings.update_one(
        {"_id": ObjectId(warning_id)},
        {"$set": {
            "acknowledged": True,
            "acknowledged_at": datetime.utcnow(),
            "acknowledged_by": current_user.get("user_id") or current_user.get("worker_id") or current_user.get("sub")
        }}
    )

    return {"success": True, "message": "Warning acknowledged"}
