"""
Worker management routes.
"""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Depends, Query, UploadFile, File, Form
from bson import ObjectId
from database import get_database
from auth import (
    get_password_hash, get_current_user, get_shift_incharge_or_above,
    get_manager_or_above, UserRole, check_mine_access
)
from schemas import (
    WorkerCreate, WorkerUpdate, WorkerResponse, WorkerList, ShiftType
)

router = APIRouter(prefix="/workers", tags=["Worker Management"])


async def get_worker_with_details(db, worker_doc: dict) -> WorkerResponse:
    """Helper to build WorkerResponse with mine/zone names."""
    mine_name = None
    zone_name = None

    if worker_doc.get("mine_id"):
        try:
            mine = await db.mines.find_one({"_id": ObjectId(worker_doc["mine_id"])})
            if mine:
                mine_name = mine["name"]
        except:
            pass

    if worker_doc.get("zone_id"):
        try:
            zone = await db.zones.find_one({"_id": ObjectId(worker_doc["zone_id"])})
            if zone:
                zone_name = zone["name"]
        except:
            pass

    return WorkerResponse(
        id=str(worker_doc["_id"]),
        employee_id=worker_doc["employee_id"],
        name=worker_doc["name"],
        department=worker_doc.get("department"),
        mine_id=str(worker_doc["mine_id"]) if worker_doc.get("mine_id") else "",
        mine_name=mine_name,
        zone_id=str(worker_doc["zone_id"]) if worker_doc.get("zone_id") else None,
        zone_name=zone_name,
        assigned_shift=ShiftType(worker_doc.get("assigned_shift", "day")),
        phone=worker_doc.get("phone"),
        emergency_contact=worker_doc.get("emergency_contact"),
        face_registered=worker_doc.get("face_registered", False),
        is_active=worker_doc.get("is_active", True),
        created_at=worker_doc["created_at"],
        compliance_score=worker_doc.get("compliance_score", 100.0),
        total_violations=worker_doc.get("total_violations", 0),
        badges=worker_doc.get("badges", [])
    )


@router.post("", response_model=WorkerResponse)
async def create_worker(
    worker_data: WorkerCreate,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """
    Create a new worker. Shift Incharge and above can create workers.
    """
    db = get_database()

    # Check mine access
    if not check_mine_access(current_user, worker_data.mine_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this mine"
        )

    # Check if employee_id already exists
    existing = await db.workers.find_one({"employee_id": worker_data.employee_id})
    if existing:
        raise HTTPException(status_code=400, detail="Employee ID already exists")

    # Verify mine exists
    try:
        mine = await db.mines.find_one({"_id": ObjectId(worker_data.mine_id)})
        if not mine:
            raise HTTPException(status_code=400, detail="Mine not found")
    except:
        raise HTTPException(status_code=400, detail="Invalid mine ID")

    # Verify zone if provided
    if worker_data.zone_id:
        try:
            zone = await db.zones.find_one({"_id": ObjectId(worker_data.zone_id)})
            if not zone:
                raise HTTPException(status_code=400, detail="Zone not found")
        except:
            raise HTTPException(status_code=400, detail="Invalid zone ID")

    # Create worker document
    worker_doc = {
        "employee_id": worker_data.employee_id,
        "name": worker_data.name,
        "password_hash": get_password_hash(worker_data.password),
        "department": worker_data.department,
        "mine_id": ObjectId(worker_data.mine_id),
        "zone_id": ObjectId(worker_data.zone_id) if worker_data.zone_id else None,
        "assigned_shift": worker_data.assigned_shift.value,
        "phone": worker_data.phone,
        "emergency_contact": worker_data.emergency_contact,
        "face_registered": False,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "created_by": current_user.get("user_id") or current_user.get("sub"),
        "compliance_score": 100.0,
        "total_violations": 0,
        "badges": [],
    }

    result = await db.workers.insert_one(worker_doc)
    worker_doc["_id"] = result.inserted_id

    return await get_worker_with_details(db, worker_doc)


@router.get("", response_model=WorkerList)
async def list_workers(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    mine_id: Optional[str] = None,
    zone_id: Optional[str] = None,
    shift: Optional[ShiftType] = None,
    department: Optional[str] = None,
    search: Optional[str] = None,
    is_active: Optional[bool] = None,
    face_registered: Optional[bool] = None,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """
    List workers with filters. Access based on mine assignment.
    """
    db = get_database()

    query = {}

    # Filter by mine access
    user_role = UserRole(current_user.get("role"))

    if user_role in [UserRole.SUPER_ADMIN, UserRole.GENERAL_MANAGER]:
        pass  # Can see all workers
    elif user_role == UserRole.AREA_SAFETY_OFFICER:
        mine_ids = current_user.get("mine_ids", [])
        query["mine_id"] = {"$in": [ObjectId(mid) for mid in mine_ids]}
    else:
        # Manager, Safety Officer, Shift Incharge - only their mine
        user_mine_id = current_user.get("mine_id")
        if user_mine_id:
            query["mine_id"] = ObjectId(user_mine_id)

    # Apply additional filters
    if mine_id:
        if not check_mine_access(current_user, mine_id):
            raise HTTPException(status_code=403, detail="No access to this mine")
        query["mine_id"] = ObjectId(mine_id)

    if zone_id:
        query["zone_id"] = ObjectId(zone_id)

    if shift:
        query["assigned_shift"] = shift.value

    if department:
        query["department"] = {"$regex": department, "$options": "i"}

    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"employee_id": {"$regex": search, "$options": "i"}},
            {"department": {"$regex": search, "$options": "i"}}
        ]

    if is_active is not None:
        query["is_active"] = is_active

    if face_registered is not None:
        query["face_registered"] = face_registered

    cursor = db.workers.find(query).skip(skip).limit(limit).sort("name", 1)
    workers = []

    async for worker in cursor:
        workers.append(await get_worker_with_details(db, worker))

    total = await db.workers.count_documents(query)

    return WorkerList(workers=workers, total=total)


@router.get("/{worker_id}", response_model=WorkerResponse)
async def get_worker(
    worker_id: str,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Get worker by ID or employee_id."""
    db = get_database()

    # Try to find by ObjectId first, then by employee_id
    worker = None
    try:
        worker = await db.workers.find_one({"_id": ObjectId(worker_id)})
    except:
        worker = await db.workers.find_one({"employee_id": worker_id})

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Check mine access
    worker_mine_id = str(worker["mine_id"]) if worker.get("mine_id") else None
    if worker_mine_id and not check_mine_access(current_user, worker_mine_id):
        raise HTTPException(status_code=403, detail="No access to this worker")

    return await get_worker_with_details(db, worker)


@router.put("/{worker_id}", response_model=WorkerResponse)
async def update_worker(
    worker_id: str,
    worker_data: WorkerUpdate,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Update worker details."""
    db = get_database()

    # Find worker
    worker = None
    try:
        worker = await db.workers.find_one({"_id": ObjectId(worker_id)})
    except:
        worker = await db.workers.find_one({"employee_id": worker_id})

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Check mine access
    worker_mine_id = str(worker["mine_id"]) if worker.get("mine_id") else None
    if worker_mine_id and not check_mine_access(current_user, worker_mine_id):
        raise HTTPException(status_code=403, detail="No access to this worker")

    # Build update document
    update_doc = {}
    if worker_data.name:
        update_doc["name"] = worker_data.name
    if worker_data.department is not None:
        update_doc["department"] = worker_data.department
    if worker_data.mine_id:
        if not check_mine_access(current_user, worker_data.mine_id):
            raise HTTPException(status_code=403, detail="No access to target mine")
        update_doc["mine_id"] = ObjectId(worker_data.mine_id)
    if worker_data.zone_id is not None:
        update_doc["zone_id"] = ObjectId(worker_data.zone_id) if worker_data.zone_id else None
    if worker_data.assigned_shift:
        update_doc["assigned_shift"] = worker_data.assigned_shift.value
    if worker_data.phone is not None:
        update_doc["phone"] = worker_data.phone
    if worker_data.emergency_contact is not None:
        update_doc["emergency_contact"] = worker_data.emergency_contact
    if worker_data.is_active is not None:
        update_doc["is_active"] = worker_data.is_active

    if not update_doc:
        raise HTTPException(status_code=400, detail="No update data provided")

    update_doc["updated_at"] = datetime.utcnow()
    update_doc["updated_by"] = current_user.get("user_id") or current_user.get("sub")

    await db.workers.update_one(
        {"_id": worker["_id"]},
        {"$set": update_doc}
    )

    # Fetch updated worker
    worker = await db.workers.find_one({"_id": worker["_id"]})
    return await get_worker_with_details(db, worker)


@router.delete("/{worker_id}")
async def delete_worker(
    worker_id: str,
    current_user: dict = Depends(get_manager_or_above)
):
    """Delete worker (soft delete). Manager and above only."""
    db = get_database()

    # Find worker
    worker = None
    try:
        worker = await db.workers.find_one({"_id": ObjectId(worker_id)})
    except:
        worker = await db.workers.find_one({"employee_id": worker_id})

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Check mine access
    worker_mine_id = str(worker["mine_id"]) if worker.get("mine_id") else None
    if worker_mine_id and not check_mine_access(current_user, worker_mine_id):
        raise HTTPException(status_code=403, detail="No access to this worker")

    # Soft delete
    await db.workers.update_one(
        {"_id": worker["_id"]},
        {"$set": {
            "is_active": False,
            "deleted_at": datetime.utcnow(),
            "deleted_by": current_user.get("user_id") or current_user.get("sub")
        }}
    )

    return {"success": True, "message": "Worker deactivated successfully"}


@router.post("/{worker_id}/reset-password")
async def reset_worker_password(
    worker_id: str,
    new_password: str = Form(...),
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Reset worker password."""
    db = get_database()

    # Find worker
    worker = None
    try:
        worker = await db.workers.find_one({"_id": ObjectId(worker_id)})
    except:
        worker = await db.workers.find_one({"employee_id": worker_id})

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Check mine access
    worker_mine_id = str(worker["mine_id"]) if worker.get("mine_id") else None
    if worker_mine_id and not check_mine_access(current_user, worker_mine_id):
        raise HTTPException(status_code=403, detail="No access to this worker")

    await db.workers.update_one(
        {"_id": worker["_id"]},
        {"$set": {
            "password_hash": get_password_hash(new_password),
            "password_reset_at": datetime.utcnow(),
            "password_reset_by": current_user.get("user_id") or current_user.get("sub")
        }}
    )

    return {"success": True, "message": "Password reset successfully"}


@router.post("/{worker_id}/register-face")
async def register_worker_face(
    worker_id: str,
    file: UploadFile = File(...),
    angle: str = Form(None),
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Register face for worker.

    Supports multi-angle face registration for improved recognition:
    - No angle: Primary face registration (center view)
    - angle=angle_1: Left view
    - angle=angle_2: Right view

    Multiple angles improve face recognition accuracy.
    """
    db = get_database()

    # Find worker
    worker = None
    try:
        worker = await db.workers.find_one({"_id": ObjectId(worker_id)})
    except:
        worker = await db.workers.find_one({"employee_id": worker_id})

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Check mine access
    worker_mine_id = str(worker["mine_id"]) if worker.get("mine_id") else None
    if worker_mine_id and not check_mine_access(current_user, worker_mine_id):
        raise HTTPException(status_code=403, detail="No access to this worker")

    contents = await file.read()

    # Import and use the global detector from main to ensure faces are in sync
    try:
        from main import detector as main_detector
        local_detector = main_detector
    except ImportError:
        # Fallback: create local detector if main import fails
        from detector import PersonDetector
        local_detector = PersonDetector()

    # Determine the face registration key
    # Primary registration uses employee_id, angles use employee_id_angle_N
    if angle:
        face_key = f"{worker['employee_id']}_{angle}"
        display_name = f"{worker['name']} ({angle})"
    else:
        face_key = worker["employee_id"]
        display_name = worker["name"]

    success = local_detector.register_face(face_key, contents, display_name)

    if not success:
        raise HTTPException(status_code=400, detail="No face detected in image")

    # Reload faces in the main detector to ensure new face is available for detection
    try:
        from main import detector as main_detector
        main_detector.reload_faces()
        print(f"Main detector now has {len(main_detector.known_faces)} faces registered")
    except Exception as e:
        print(f"Warning: Could not reload main detector faces: {e}")

    # Update worker's face registration status
    update_data = {"face_registered": True}

    # Track registered angles in the worker document
    if angle:
        registered_angles = worker.get("face_angles", [])
        if angle not in registered_angles:
            registered_angles.append(angle)
        update_data["face_angles"] = registered_angles
    else:
        # Primary face registered
        update_data["primary_face_registered"] = True

    await db.workers.update_one(
        {"_id": worker["_id"]},
        {"$set": update_data}
    )

    angle_msg = f" ({angle})" if angle else ""
    return {"success": True, "message": f"Face registered for {worker['name']}{angle_msg}"}


@router.get("/{worker_id}/violations")
async def get_worker_violations(
    worker_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Get worker's violation history."""
    db = get_database()

    # Find worker
    worker = None
    try:
        worker = await db.workers.find_one({"_id": ObjectId(worker_id)})
    except:
        worker = await db.workers.find_one({"employee_id": worker_id})

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Get violations from gate_entries
    cursor = db.gate_entries.find({
        "worker_id": str(worker["_id"]),
        "violations": {"$ne": []}
    }).skip(skip).limit(limit).sort("timestamp", -1)

    violations = []
    async for entry in cursor:
        violations.append({
            "id": str(entry["_id"]),
            "timestamp": entry["timestamp"].isoformat(),
            "violations": entry["violations"],
            "gate_id": entry.get("gate_id"),
            "shift": entry.get("shift"),
        })

    total = await db.gate_entries.count_documents({
        "worker_id": str(worker["_id"]),
        "violations": {"$ne": []}
    })

    return {
        "worker_id": worker_id,
        "worker_name": worker["name"],
        "violations": violations,
        "total": total
    }


@router.get("/{worker_id}/attendance")
async def get_worker_attendance(
    worker_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Get worker's attendance/entry history."""
    db = get_database()

    # Find worker
    worker = None
    try:
        worker = await db.workers.find_one({"_id": ObjectId(worker_id)})
    except:
        worker = await db.workers.find_one({"employee_id": worker_id})

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    query = {"worker_id": str(worker["_id"])}

    if start_date or end_date:
        query["timestamp"] = {}
        if start_date:
            query["timestamp"]["$gte"] = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            from datetime import timedelta
            query["timestamp"]["$lt"] = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    cursor = db.gate_entries.find(query).skip(skip).limit(limit).sort("timestamp", -1)

    entries = []
    async for entry in cursor:
        entries.append({
            "id": str(entry["_id"]),
            "entry_type": entry["entry_type"],
            "timestamp": entry["timestamp"].isoformat(),
            "gate_id": entry.get("gate_id"),
            "shift": entry.get("shift"),
            "ppe_status": entry.get("ppe_status", {}),
            "violations": entry.get("violations", []),
            "status": entry.get("status"),
        })

    total = await db.gate_entries.count_documents(query)

    return {
        "worker_id": worker_id,
        "worker_name": worker["name"],
        "entries": entries,
        "total": total
    }


@router.get("/{worker_id}/prediction")
async def get_worker_prediction_detail(
    worker_id: str,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """
    Get prediction details and recommendations for worker profile page.
    """
    db = get_database()

    # Verify worker exists
    try:
        worker = await db.workers.find_one({"_id": ObjectId(worker_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid worker ID")

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Check mine access
    mine_id = worker.get("mine_id")
    if mine_id and not check_mine_access(current_user, str(mine_id)):
        raise HTTPException(status_code=403, detail="No access to this worker's mine")

    # Get latest prediction
    prediction = await db.predictions.find_one(
        {"worker_id": worker_id},
        sort=[("prediction_date", -1)]
    )

    # If no prediction or expired, generate new one
    if not prediction or prediction.get("expires_at", datetime.min) < datetime.utcnow():
        # Import here to avoid circular dependency
        from ml.prediction_service import PredictionService

        service = PredictionService(db)
        prediction = await service.predict_worker_risk(worker_id)

        # Save to database
        await db.predictions.insert_one(prediction)

    # Generate recommendations based on risk factors
    recommendations = _generate_recommendations(prediction)

    return {
        "worker_id": worker_id,
        "employee_id": worker.get("employee_id", ""),
        "name": worker.get("name", ""),
        "prediction": {
            "overall_risk_score": round(prediction["overall_risk_score"], 1),
            "risk_category": prediction["risk_category"],
            "violation_risk_score": round(prediction["violation_risk_score"], 1),
            "attendance_risk_score": round(prediction["attendance_risk_score"], 1),
            "predicted_violations": prediction["predicted_violations_count"],
            "predicted_absent_days": prediction["predicted_absent_days"],
            "attendance_rate_30d": round(prediction["attendance_rate_30d"], 1),
            "high_risk_ppe_items": prediction["high_risk_ppe_items"],
            "requires_intervention": prediction["requires_intervention"],
            "risk_factors": prediction.get("risk_factors", []),
            "confidence": prediction["confidence"],
            "prediction_date": prediction["prediction_date"].isoformat(),
            "expires_at": prediction["expires_at"].isoformat()
        },
        "recommendations": recommendations
    }


def _generate_recommendations(prediction: dict) -> list:
    """Generate actionable recommendations based on prediction"""
    recommendations = []

    risk_factors = prediction.get("risk_factors", [])
    risk_category = prediction.get("risk_category", "low")

    # Extract factor types
    factor_types = {rf["factor"] for rf in risk_factors}

    # Violation-related recommendations
    if any(f in factor_types for f in ["high_violation_count", "moderate_violation_count"]):
        recommendations.append("Schedule mandatory PPE safety training session")

        high_risk_items = prediction.get("high_risk_ppe_items", [])
        if high_risk_items:
            items_str = ", ".join(high_risk_items)
            recommendations.append(f"Focus training on proper use of: {items_str}")

    # Attendance-related recommendations
    if "critical_low_attendance" in factor_types or "low_attendance" in factor_types:
        recommendations.append("Conduct attendance counseling session")
        recommendations.append("Review work-life balance and personal issues")

    # Consecutive absence recommendations
    if "extended_absence" in factor_types or "consecutive_absence" in factor_types:
        recommendations.append("Immediate check-in required - verify worker wellbeing")
        recommendations.append("Consider temporary reassignment to less critical zones")

    # Compliance recommendations
    if "very_low_compliance" in factor_types:
        recommendations.append("IMMEDIATE ACTION: Assign safety buddy for supervision")
        recommendations.append("Restrict access to high-risk zones until improvement")
        recommendations.append("Weekly safety compliance reviews with supervisor")

    # Warning-related recommendations
    if "multiple_warnings" in factor_types:
        recommendations.append("Escalate to formal performance improvement plan")
        recommendations.append("Require sign-off from Safety Officer for zone access")

    # New worker recommendations
    if "new_worker_violations" in factor_types:
        recommendations.append("Provide additional onboarding and safety orientation")
        recommendations.append("Assign experienced mentor for first 60 days")

    # General recommendations if critical
    if risk_category == "critical" and not recommendations:
        recommendations.append("CRITICAL: Immediate intervention required")
        recommendations.append("Schedule meeting with worker and Safety Officer")
        recommendations.append("Develop personalized safety improvement plan")

    return recommendations[:5]  # Limit to top 5
