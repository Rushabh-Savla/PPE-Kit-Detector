"""
Authentication routes for role-based login.
"""
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Depends
from database import get_database
from auth import (
    get_password_hash, verify_password, create_access_token,
    get_current_user, UserRole
)
from schemas import (
    LoginRequest, TokenResponse, UserResponse,
    WorkerLoginRequest, WorkerTokenResponse, WorkerResponse,
    PasswordChange, ShiftType
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest):
    """
    Login endpoint for staff/management users.
    Returns JWT token with role information.
    """
    db = get_database()
    user = await db.users.find_one({"username": credentials.username})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is deactivated"
        )

    if not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    # Update last login
    await db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )

    # Create token with role and access info
    token_data = {
        "sub": credentials.username,
        "user_id": str(user["_id"]),
        "role": user["role"],
        "user_type": "staff",
        "mine_id": user.get("mine_id"),
        "mine_ids": user.get("mine_ids", []),
        "assigned_shift": user.get("assigned_shift"),
        "assigned_gate_id": user.get("assigned_gate_id"),
    }

    access_token = create_access_token(data=token_data)

    user_response = UserResponse(
        id=str(user["_id"]),
        username=user["username"],
        full_name=user["full_name"],
        email=user.get("email"),
        phone=user.get("phone"),
        role=UserRole(user["role"]),
        mine_id=user.get("mine_id"),
        mine_ids=user.get("mine_ids"),
        assigned_shift=ShiftType(user["assigned_shift"]) if user.get("assigned_shift") else None,
        assigned_gate_id=user.get("assigned_gate_id"),
        is_active=user.get("is_active", True),
        created_at=user["created_at"],
        last_login=datetime.utcnow()
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=user_response
    )


@router.post("/worker/login", response_model=WorkerTokenResponse)
async def worker_login(credentials: WorkerLoginRequest):
    """
    Login endpoint for workers (via employee_id + password).
    Returns JWT token with worker information.
    """
    db = get_database()
    worker = await db.workers.find_one({"employee_id": credentials.employee_id})

    if not worker:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid employee ID or password"
        )

    if not worker.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is deactivated"
        )

    if not verify_password(credentials.password, worker["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid employee ID or password"
        )

    # Update last login
    await db.workers.update_one(
        {"_id": worker["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )

    # Get mine and zone names
    mine_name = None
    zone_name = None
    if worker.get("mine_id"):
        mine = await db.mines.find_one({"_id": worker["mine_id"]})
        if mine:
            mine_name = mine["name"]
    if worker.get("zone_id"):
        zone = await db.zones.find_one({"_id": worker["zone_id"]})
        if zone:
            zone_name = zone["name"]

    # Create token
    token_data = {
        "sub": credentials.employee_id,
        "worker_id": str(worker["_id"]),
        "role": UserRole.WORKER.value,
        "user_type": "worker",
        "mine_id": str(worker["mine_id"]) if worker.get("mine_id") else None,
        "zone_id": str(worker["zone_id"]) if worker.get("zone_id") else None,
        "assigned_shift": worker.get("assigned_shift"),
    }

    access_token = create_access_token(data=token_data)

    worker_response = WorkerResponse(
        id=str(worker["_id"]),
        employee_id=worker["employee_id"],
        name=worker["name"],
        department=worker.get("department"),
        mine_id=str(worker["mine_id"]) if worker.get("mine_id") else "",
        mine_name=mine_name,
        zone_id=str(worker["zone_id"]) if worker.get("zone_id") else None,
        zone_name=zone_name,
        assigned_shift=ShiftType(worker.get("assigned_shift", "day")),
        phone=worker.get("phone"),
        emergency_contact=worker.get("emergency_contact"),
        face_registered=worker.get("face_registered", False),
        is_active=worker.get("is_active", True),
        created_at=worker["created_at"],
        compliance_score=worker.get("compliance_score", 100.0),
        total_violations=worker.get("total_violations", 0),
        badges=worker.get("badges", [])
    )

    return WorkerTokenResponse(
        access_token=access_token,
        token_type="bearer",
        worker=worker_response
    )


@router.get("/verify")
async def verify_auth(current_user: dict = Depends(get_current_user)):
    """Verify current token is valid and return user info."""
    return {
        "valid": True,
        "user_type": current_user.get("user_type"),
        "username": current_user.get("sub"),
        "role": current_user.get("role"),
        "mine_id": current_user.get("mine_id"),
    }


@router.get("/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get full current user information."""
    db = get_database()

    user_type = current_user.get("user_type")

    if user_type == "worker":
        worker = await db.workers.find_one({"employee_id": current_user.get("sub")})
        if not worker:
            raise HTTPException(status_code=404, detail="Worker not found")

        mine_name = None
        zone_name = None
        if worker.get("mine_id"):
            mine = await db.mines.find_one({"_id": worker["mine_id"]})
            if mine:
                mine_name = mine["name"]
        if worker.get("zone_id"):
            zone = await db.zones.find_one({"_id": worker["zone_id"]})
            if zone:
                zone_name = zone["name"]

        return WorkerResponse(
            id=str(worker["_id"]),
            employee_id=worker["employee_id"],
            name=worker["name"],
            department=worker.get("department"),
            mine_id=str(worker["mine_id"]) if worker.get("mine_id") else "",
            mine_name=mine_name,
            zone_id=str(worker["zone_id"]) if worker.get("zone_id") else None,
            zone_name=zone_name,
            assigned_shift=ShiftType(worker.get("assigned_shift", "day")),
            phone=worker.get("phone"),
            emergency_contact=worker.get("emergency_contact"),
            face_registered=worker.get("face_registered", False),
            is_active=worker.get("is_active", True),
            created_at=worker["created_at"],
            compliance_score=worker.get("compliance_score", 100.0),
            total_violations=worker.get("total_violations", 0),
            badges=worker.get("badges", [])
        )
    else:
        user = await db.users.find_one({"username": current_user.get("sub")})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return UserResponse(
            id=str(user["_id"]),
            username=user["username"],
            full_name=user["full_name"],
            email=user.get("email"),
            phone=user.get("phone"),
            role=UserRole(user["role"]),
            mine_id=user.get("mine_id"),
            mine_ids=user.get("mine_ids"),
            assigned_shift=ShiftType(user["assigned_shift"]) if user.get("assigned_shift") else None,
            assigned_gate_id=user.get("assigned_gate_id"),
            is_active=user.get("is_active", True),
            created_at=user["created_at"],
            last_login=user.get("last_login")
        )


@router.post("/change-password")
async def change_password(
    data: PasswordChange,
    current_user: dict = Depends(get_current_user)
):
    """Change password for current user."""
    db = get_database()

    user_type = current_user.get("user_type")

    if user_type == "worker":
        worker = await db.workers.find_one({"employee_id": current_user.get("sub")})
        if not worker:
            raise HTTPException(status_code=404, detail="Worker not found")

        if not verify_password(data.current_password, worker["password_hash"]):
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        await db.workers.update_one(
            {"_id": worker["_id"]},
            {"$set": {"password_hash": get_password_hash(data.new_password)}}
        )
    else:
        user = await db.users.find_one({"username": current_user.get("sub")})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if not verify_password(data.current_password, user["password_hash"]):
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        await db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"password_hash": get_password_hash(data.new_password)}}
        )

    return {"success": True, "message": "Password changed successfully"}
