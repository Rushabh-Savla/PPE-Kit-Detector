"""
Gate entry routes - handles PPE detection at entry gates.
"""
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Depends, Query, UploadFile, File, Form
from fastapi.responses import JSONResponse
from bson import ObjectId
import base64
from io import BytesIO

from database import get_database
from auth import (
    get_current_user, get_shift_incharge_or_above,
    UserRole, check_mine_access
)
from schemas import (
    GateEntryCreate, GateEntryResponse, GateEntryList,
    EntryStatus, ShiftType, AlertSeverity, SHIFT_DEFINITIONS
)

router = APIRouter(prefix="/gate-entries", tags=["Gate Entries"])


def get_current_shift() -> ShiftType:
    """Determine current shift based on time."""
    now = datetime.utcnow()
    hour = now.hour

    if 6 <= hour < 14:
        return ShiftType.DAY
    elif 14 <= hour < 22:
        return ShiftType.AFTERNOON
    else:
        return ShiftType.NIGHT


async def create_alert_for_violation(db, entry_doc: dict, worker_doc: dict, gate_doc: dict):
    """Create an alert for PPE violations."""
    violations = entry_doc.get("violations", [])
    if not violations:
        return

    severity = AlertSeverity.LOW
    if len(violations) >= 3:
        severity = AlertSeverity.HIGH
    elif len(violations) >= 2:
        severity = AlertSeverity.MEDIUM

    alert_doc = {
        "alert_type": "ppe_violation",
        "severity": severity.value,
        "status": "active",
        "message": f"PPE violation detected: {', '.join(violations)}",
        "mine_id": entry_doc.get("mine_id"),
        "gate_id": entry_doc.get("gate_id"),
        "worker_id": entry_doc.get("worker_id"),
        "entry_id": str(entry_doc["_id"]),
        "created_at": datetime.utcnow(),
        "metadata": {
            "violations": violations,
            "worker_name": worker_doc.get("name") if worker_doc else "Unknown",
            "employee_id": worker_doc.get("employee_id") if worker_doc else None,
            "gate_name": gate_doc.get("name") if gate_doc else "Unknown",
        }
    }

    await db.alerts.insert_one(alert_doc)


async def update_worker_compliance(db, worker_id: str, has_violation: bool):
    """Update worker's compliance score based on entry."""
    worker = await db.workers.find_one({"_id": ObjectId(worker_id)})
    if not worker:
        return

    current_score = worker.get("compliance_score", 100.0)
    total_violations = worker.get("total_violations", 0)

    if has_violation:
        # Decrease score for violations (min 0)
        new_score = max(0, current_score - 5)
        total_violations += 1
    else:
        # Slight increase for compliant entries (max 100)
        new_score = min(100, current_score + 0.5)

    # Update badges based on score
    badges = worker.get("badges", [])
    if new_score >= 95 and "safety_star" not in badges:
        badges.append("safety_star")
    if total_violations == 0 and "perfect_record" not in badges:
        badges.append("perfect_record")

    await db.workers.update_one(
        {"_id": ObjectId(worker_id)},
        {"$set": {
            "compliance_score": round(new_score, 1),
            "total_violations": total_violations,
            "badges": badges,
            "last_entry_at": datetime.utcnow()
        }}
    )


async def add_attendance_record(db, worker: dict, entry_type: str, violations: list, img_base64: str, location: str = None):
    """Add attendance record to the attendance collection after worker detection."""
    if not worker:
        return None

    employee_id = worker.get("employee_id")
    employee_name = worker.get("name")
    worker_id = str(worker["_id"]) if "_id" in worker else None

    # Determine attendance type based on entry_type
    attendance_type = "check_in" if entry_type == "entry" else "check_out"
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # For check-in, check if already checked in today
    if attendance_type == "check_in":
        existing = await db.attendance.find_one({
            "employee_id": employee_id,
            "date": today,
            "type": "check_in"
        })
        if existing:
            # Already checked in today, skip duplicate
            return {
                "employee_id": employee_id,
                "employee_name": employee_name,
                "timestamp": existing["timestamp"].isoformat(),
                "already_checked_in": True
            }

    # Create attendance record
    attendance_doc = {
        "employee_id": employee_id,
        "employee_name": employee_name,
        "worker_id": worker_id,
        "type": attendance_type,
        "timestamp": datetime.utcnow(),
        "date": today,
        "location": location,
        "ppe_compliant": len(violations) == 0,
        "violations": violations,
        "source": "gate_entry",  # Track that this came from gate entry
    }

    # Only add image for check-in
    if attendance_type == "check_in" and img_base64:
        attendance_doc["image"] = f"data:image/png;base64,{img_base64}"

    await db.attendance.insert_one(attendance_doc)

    return {
        "employee_id": employee_id,
        "employee_name": employee_name,
        "timestamp": attendance_doc["timestamp"].isoformat(),
        "ppe_compliant": attendance_doc["ppe_compliant"],
        "violations": violations
    }


# Global detector instance (loaded once at module import)
_detector = None

def get_detector():
    """Get or create the singleton detector instance."""
    global _detector
    if _detector is None:
        from detector import PersonDetector
        _detector = PersonDetector()
    return _detector


@router.post("/detect")
async def detect_and_record_entry(
    gate_id: str = Form(None),  # Made optional
    file: UploadFile = File(...),
    entry_type: str = Form("entry"),
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """
    Process gate camera feed - detect PPE and faces, record entry.
    This is the main endpoint for gate cameras.
    Gate ID is optional - if not provided, will use the first available gate.
    """
    db = get_database()

    gate = None

    # If gate_id provided, verify it exists
    if gate_id:
        try:
            gate = await db.gates.find_one({"_id": ObjectId(gate_id)})
        except:
            raise HTTPException(status_code=400, detail="Invalid gate ID")

        if not gate:
            raise HTTPException(status_code=404, detail="Gate not found")
    else:
        # No gate_id provided - find the first available gate
        gate = await db.gates.find_one({"is_active": True, "has_camera": True})
        if not gate:
            # Create a default test gate if none exists
            default_mine = await db.mines.find_one({"is_active": True})
            if not default_mine:
                # Create a default mine
                mine_result = await db.mines.insert_one({
                    "name": "Default Mine",
                    "location": "Auto-created",
                    "is_active": True,
                    "created_at": datetime.utcnow(),
                })
                default_mine = {"_id": mine_result.inserted_id}

            # Create a default gate
            gate_result = await db.gates.insert_one({
                "mine_id": default_mine["_id"],
                "name": "Default Gate",
                "gate_type": "both",
                "has_camera": True,
                "is_active": True,
                "created_at": datetime.utcnow(),
            })
            gate = await db.gates.find_one({"_id": gate_result.inserted_id})

        gate_id = str(gate["_id"])

    mine_id = str(gate["mine_id"])

    # Check mine access
    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    # Process image using singleton detector
    contents = await file.read()
    detector = get_detector()
    result_image, detections = detector.process_image(contents)

    # Convert result image to base64
    buffered = BytesIO()
    result_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Get identified persons and violations
    summary = detections.get("summary", {})
    identified_persons = summary.get("identified_persons", [])
    violations = detections.get("violations", [])
    violation_labels = [v.get("label", "Unknown") for v in violations]

    # Determine current shift
    current_shift = get_current_shift()

    # Find worker if identified
    worker = None
    worker_id = None
    if identified_persons:
        employee_id = identified_persons[0]
        worker = await db.workers.find_one({"employee_id": employee_id})
        if worker:
            worker_id = str(worker["_id"])

    # Determine entry status
    entry_status = EntryStatus.APPROVED
    if violation_labels:
        entry_status = EntryStatus.DENIED

    # Create entry record
    entry_doc = {
        "gate_id": gate_id,
        "mine_id": ObjectId(mine_id),
        "worker_id": worker_id,
        "employee_id": worker.get("employee_id") if worker else None,
        "worker_name": worker.get("name") if worker else "Unknown",
        "entry_type": entry_type,
        "status": entry_status.value,
        "ppe_status": {
            "helmet": "NO Helmet" not in violation_labels,
            "vest": "NO Vest" not in violation_labels,
            "goggles": "NO Goggles" not in violation_labels,
            "gloves": "NO Gloves" not in violation_labels,
            "mask": "NO Mask" not in violation_labels,
            "safety_shoes": "NO Safety Shoes" not in violation_labels,
        },
        "violations": violation_labels,
        "timestamp": datetime.utcnow(),
        "shift": current_shift.value,
        "image": f"data:image/png;base64,{img_base64}",
        "recorded_by": current_user.get("user_id") or current_user.get("sub"),
        "detections_raw": detections,
    }

    result = await db.gate_entries.insert_one(entry_doc)
    entry_doc["_id"] = result.inserted_id

    # Create alert if violations
    if violation_labels and worker:
        await create_alert_for_violation(db, entry_doc, worker, gate)

    # Update worker compliance score
    if worker_id:
        await update_worker_compliance(db, worker_id, len(violation_labels) > 0)

    # Add attendance record to attendance collection
    attendance_record = None
    if worker:
        gate_name_for_location = gate.get("name", "Unknown Gate")
        attendance_record = await add_attendance_record(
            db,
            worker,
            entry_type,
            violation_labels,
            img_base64,
            location=gate_name_for_location
        )

    # Get gate name for response
    gate_name = gate.get("name", "Unknown Gate")

    response_entry = GateEntryResponse(
        id=str(result.inserted_id),
        gate_id=gate_id,
        gate_name=gate_name,
        worker_id=worker_id,
        worker_name=worker.get("name") if worker else None,
        employee_id=worker.get("employee_id") if worker else None,
        entry_type=entry_type,
        status=entry_status,
        ppe_status=entry_doc["ppe_status"],
        violations=violation_labels,
        timestamp=entry_doc["timestamp"],
        shift=current_shift
    )

    return JSONResponse({
        "success": True,
        "entry": response_entry.model_dump(mode="json"),
        "image": f"data:image/png;base64,{img_base64}",
        "detections": detections,
        "violations": violation_labels,
        "attendance_marked": attendance_record is not None,
        "attendance": attendance_record,
        "message": f"Entry {'approved' if entry_status == EntryStatus.APPROVED else 'denied'} - {len(violation_labels)} violation(s) detected"
    })


@router.post("/{entry_id}/override")
async def override_entry(
    entry_id: str,
    reason: str = Form(...),
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """
    Override a denied entry (with reason logging).
    Only Shift Incharge and above can override.
    """
    db = get_database()

    try:
        entry = await db.gate_entries.find_one({"_id": ObjectId(entry_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid entry ID")

    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    # Check mine access
    mine_id = str(entry["mine_id"])
    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    # Update entry with override
    await db.gate_entries.update_one(
        {"_id": ObjectId(entry_id)},
        {"$set": {
            "status": EntryStatus.OVERRIDE.value,
            "override_by": current_user.get("user_id") or current_user.get("sub"),
            "override_by_name": current_user.get("sub"),
            "override_reason": reason,
            "override_at": datetime.utcnow()
        }}
    )

    # Resolve any active alerts for this entry
    await db.alerts.update_many(
        {"entry_id": entry_id, "status": "active"},
        {"$set": {
            "status": "resolved",
            "resolved_by": current_user.get("user_id") or current_user.get("sub"),
            "resolved_at": datetime.utcnow(),
            "resolution_notes": f"Entry overridden: {reason}"
        }}
    )

    return {"success": True, "message": "Entry overridden successfully"}


@router.get("", response_model=GateEntryList)
async def list_entries(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    gate_id: Optional[str] = None,
    mine_id: Optional[str] = None,
    worker_id: Optional[str] = None,
    shift: Optional[ShiftType] = None,
    status: Optional[EntryStatus] = None,
    entry_type: Optional[str] = None,
    has_violations: Optional[bool] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """List gate entries with filters."""
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
    if gate_id:
        query["gate_id"] = gate_id

    if mine_id:
        if not check_mine_access(current_user, mine_id):
            raise HTTPException(status_code=403, detail="No access to this mine")
        query["mine_id"] = ObjectId(mine_id)

    if worker_id:
        query["worker_id"] = worker_id

    if shift:
        query["shift"] = shift.value

    if status:
        query["status"] = status.value

    if entry_type:
        query["entry_type"] = entry_type

    if has_violations is not None:
        if has_violations:
            query["violations"] = {"$ne": []}
        else:
            query["violations"] = []

    if start_date or end_date:
        query["timestamp"] = {}
        if start_date:
            query["timestamp"]["$gte"] = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            query["timestamp"]["$lt"] = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    cursor = db.gate_entries.find(query).skip(skip).limit(limit).sort("timestamp", -1)
    entries = []

    async for entry in cursor:
        # Get gate name
        gate_name = None
        if entry.get("gate_id"):
            gate = await db.gates.find_one({"_id": ObjectId(entry["gate_id"])})
            if gate:
                gate_name = gate["name"]

        entries.append(GateEntryResponse(
            id=str(entry["_id"]),
            gate_id=entry.get("gate_id"),
            gate_name=gate_name,
            worker_id=entry.get("worker_id"),
            worker_name=entry.get("worker_name"),
            employee_id=entry.get("employee_id"),
            entry_type=entry.get("entry_type", "entry"),
            status=EntryStatus(entry.get("status", "approved")),
            ppe_status=entry.get("ppe_status", {}),
            violations=entry.get("violations", []),
            timestamp=entry["timestamp"],
            shift=ShiftType(entry.get("shift", "day")),
            override_by=entry.get("override_by"),
            override_reason=entry.get("override_reason")
        ))

    total = await db.gate_entries.count_documents(query)

    return GateEntryList(entries=entries, total=total)


@router.get("/live")
async def get_live_entries(
    gate_id: Optional[str] = None,
    mine_id: Optional[str] = None,
    limit: int = Query(20, ge=1, le=50),
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Get live/recent entries for current shift dashboard."""
    db = get_database()

    current_shift = get_current_shift()

    # Calculate shift start time
    now = datetime.utcnow()
    if current_shift == ShiftType.DAY:
        shift_start = now.replace(hour=6, minute=0, second=0, microsecond=0)
    elif current_shift == ShiftType.AFTERNOON:
        shift_start = now.replace(hour=14, minute=0, second=0, microsecond=0)
    else:  # Night shift
        if now.hour < 6:
            shift_start = (now - timedelta(days=1)).replace(hour=22, minute=0, second=0, microsecond=0)
        else:
            shift_start = now.replace(hour=22, minute=0, second=0, microsecond=0)

    query = {
        "timestamp": {"$gte": shift_start},
        "shift": current_shift.value
    }

    # Filter by mine access
    user_role = UserRole(current_user.get("role"))
    if user_role not in [UserRole.SUPER_ADMIN, UserRole.GENERAL_MANAGER]:
        if user_role == UserRole.AREA_SAFETY_OFFICER:
            mine_ids = current_user.get("mine_ids", [])
            query["mine_id"] = {"$in": [ObjectId(mid) for mid in mine_ids]}
        else:
            user_mine_id = current_user.get("mine_id")
            if user_mine_id:
                query["mine_id"] = ObjectId(user_mine_id)

    if gate_id:
        query["gate_id"] = gate_id

    if mine_id:
        if not check_mine_access(current_user, mine_id):
            raise HTTPException(status_code=403, detail="No access to this mine")
        query["mine_id"] = ObjectId(mine_id)

    cursor = db.gate_entries.find(query).limit(limit).sort("timestamp", -1)
    entries = []

    async for entry in cursor:
        gate_name = None
        if entry.get("gate_id"):
            gate = await db.gates.find_one({"_id": ObjectId(entry["gate_id"])})
            if gate:
                gate_name = gate["name"]

        entries.append({
            "id": str(entry["_id"]),
            "gate_id": entry.get("gate_id"),
            "gate_name": gate_name,
            "worker_id": entry.get("worker_id"),
            "worker_name": entry.get("worker_name"),
            "employee_id": entry.get("employee_id"),
            "entry_type": entry.get("entry_type"),
            "status": entry.get("status"),
            "ppe_status": entry.get("ppe_status", {}),
            "violations": entry.get("violations", []),
            "timestamp": entry["timestamp"].isoformat(),
            "shift": entry.get("shift"),
        })

    # Get summary stats
    total_entries = await db.gate_entries.count_documents(query)
    total_violations = await db.gate_entries.count_documents({
        **query,
        "violations": {"$ne": []}
    })
    total_exits = await db.gate_entries.count_documents({
        **query,
        "entry_type": "exit"
    })
    total_inside = total_entries - total_exits

    shift_info = SHIFT_DEFINITIONS[current_shift]

    return {
        "current_shift": current_shift.value,
        "shift_name": shift_info.name,
        "shift_start": shift_info.start_time,
        "shift_end": shift_info.end_time,
        "entries": entries,
        "summary": {
            "total_entries": total_entries,
            "total_exits": total_exits,
            "currently_inside": total_inside,
            "total_violations": total_violations,
            "compliance_rate": round(
                ((total_entries - total_violations) / total_entries * 100) if total_entries > 0 else 100,
                1
            )
        }
    }


@router.get("/{entry_id}")
async def get_entry(
    entry_id: str,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Get entry details by ID."""
    db = get_database()

    try:
        entry = await db.gate_entries.find_one({"_id": ObjectId(entry_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid entry ID")

    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    # Check mine access
    mine_id = str(entry["mine_id"])
    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this entry")

    # Get gate name
    gate_name = None
    if entry.get("gate_id"):
        gate = await db.gates.find_one({"_id": ObjectId(entry["gate_id"])})
        if gate:
            gate_name = gate["name"]

    return {
        "id": str(entry["_id"]),
        "gate_id": entry.get("gate_id"),
        "gate_name": gate_name,
        "worker_id": entry.get("worker_id"),
        "worker_name": entry.get("worker_name"),
        "employee_id": entry.get("employee_id"),
        "entry_type": entry.get("entry_type"),
        "status": entry.get("status"),
        "ppe_status": entry.get("ppe_status", {}),
        "violations": entry.get("violations", []),
        "timestamp": entry["timestamp"].isoformat(),
        "shift": entry.get("shift"),
        "image": entry.get("image"),
        "override_by": entry.get("override_by"),
        "override_reason": entry.get("override_reason"),
        "override_at": entry.get("override_at").isoformat() if entry.get("override_at") else None,
    }
