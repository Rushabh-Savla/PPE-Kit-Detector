"""
User management routes (staff/management users).
"""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, status, Depends, Query
from bson import ObjectId
from database import get_database
from auth import (
    get_password_hash, get_current_user, get_super_admin,
    get_manager_or_above, UserRole, has_higher_or_equal_role
)
from schemas import (
    UserCreate, UserUpdate, UserResponse, UserList, ShiftType
)

router = APIRouter(prefix="/users", tags=["User Management"])


def can_manage_role(manager_role: UserRole, target_role: UserRole) -> bool:
    """Check if manager can create/edit users with target role."""
    # Super admin can manage everyone
    if manager_role == UserRole.SUPER_ADMIN:
        return True

    # General Manager can manage Area Safety Officers and below
    if manager_role == UserRole.GENERAL_MANAGER:
        return target_role in [
            UserRole.AREA_SAFETY_OFFICER,
            UserRole.MANAGER,
            UserRole.SAFETY_OFFICER,
            UserRole.SHIFT_INCHARGE
        ]

    # Area Safety Officer can manage Managers and below in their mines
    if manager_role == UserRole.AREA_SAFETY_OFFICER:
        return target_role in [
            UserRole.MANAGER,
            UserRole.SAFETY_OFFICER,
            UserRole.SHIFT_INCHARGE
        ]

    # Manager can manage Safety Officers and Shift Incharges in their mine
    if manager_role == UserRole.MANAGER:
        return target_role in [
            UserRole.SAFETY_OFFICER,
            UserRole.SHIFT_INCHARGE
        ]

    return False


@router.post("", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    current_user: dict = Depends(get_manager_or_above)
):
    """
    Create a new user. Access based on role hierarchy:
    - Super Admin: Can create any role
    - General Manager: Can create Area Safety Officer and below
    - Area Safety Officer: Can create Manager and below
    - Manager: Can create Safety Officer and Shift Incharge
    """
    db = get_database()

    # Check role management permission
    try:
        manager_role = UserRole(current_user.get("role"))
        target_role = user_data.role
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid role")

    if not can_manage_role(manager_role, target_role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You cannot create users with role: {target_role.value}"
        )

    # Check if username already exists
    existing = await db.users.find_one({"username": user_data.username})
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Validate mine_id for mine-specific roles
    mine_specific_roles = [
        UserRole.MANAGER, UserRole.SAFETY_OFFICER, UserRole.SHIFT_INCHARGE
    ]
    if target_role in mine_specific_roles and not user_data.mine_id:
        raise HTTPException(
            status_code=400,
            detail=f"mine_id is required for {target_role.value} role"
        )

    # Validate mine_ids for Area Safety Officer
    if target_role == UserRole.AREA_SAFETY_OFFICER and not user_data.mine_ids:
        raise HTTPException(
            status_code=400,
            detail="mine_ids is required for Area Safety Officer role"
        )

    # Validate shift and gate for Shift Incharge
    if target_role == UserRole.SHIFT_INCHARGE:
        if not user_data.assigned_shift:
            raise HTTPException(
                status_code=400,
                detail="assigned_shift is required for Shift Incharge role"
            )

    # Create user document
    user_doc = {
        "username": user_data.username,
        "password_hash": get_password_hash(user_data.password),
        "full_name": user_data.full_name,
        "email": user_data.email,
        "phone": user_data.phone,
        "role": user_data.role.value,
        "mine_id": user_data.mine_id,
        "mine_ids": user_data.mine_ids or [],
        "assigned_shift": user_data.assigned_shift.value if user_data.assigned_shift else None,
        "assigned_gate_id": user_data.assigned_gate_id,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "created_by": current_user.get("user_id"),
    }

    result = await db.users.insert_one(user_doc)

    return UserResponse(
        id=str(result.inserted_id),
        username=user_doc["username"],
        full_name=user_doc["full_name"],
        email=user_doc["email"],
        phone=user_doc["phone"],
        role=target_role,
        mine_id=user_doc["mine_id"],
        mine_ids=user_doc["mine_ids"],
        assigned_shift=ShiftType(user_doc["assigned_shift"]) if user_doc["assigned_shift"] else None,
        assigned_gate_id=user_doc["assigned_gate_id"],
        is_active=True,
        created_at=user_doc["created_at"]
    )


@router.get("", response_model=UserList)
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    role: Optional[UserRole] = None,
    mine_id: Optional[str] = None,
    search: Optional[str] = None,
    is_active: Optional[bool] = None,
    current_user: dict = Depends(get_manager_or_above)
):
    """
    List users with filters. Access based on role and mine assignment.
    """
    db = get_database()

    query = {}

    # Filter by role hierarchy - users can only see roles below them
    user_role = UserRole(current_user.get("role"))

    if user_role == UserRole.SUPER_ADMIN:
        pass  # Can see all
    elif user_role == UserRole.GENERAL_MANAGER:
        query["role"] = {"$in": [
            UserRole.AREA_SAFETY_OFFICER.value,
            UserRole.MANAGER.value,
            UserRole.SAFETY_OFFICER.value,
            UserRole.SHIFT_INCHARGE.value
        ]}
    elif user_role == UserRole.AREA_SAFETY_OFFICER:
        # Can only see users in their assigned mines
        mine_ids = current_user.get("mine_ids", [])
        query["$or"] = [
            {"mine_id": {"$in": mine_ids}},
            {"mine_ids": {"$elemMatch": {"$in": mine_ids}}}
        ]
        query["role"] = {"$in": [
            UserRole.MANAGER.value,
            UserRole.SAFETY_OFFICER.value,
            UserRole.SHIFT_INCHARGE.value
        ]}
    elif user_role == UserRole.MANAGER:
        # Can only see users in their mine
        user_mine_id = current_user.get("mine_id")
        if user_mine_id:
            query["mine_id"] = user_mine_id
        query["role"] = {"$in": [
            UserRole.SAFETY_OFFICER.value,
            UserRole.SHIFT_INCHARGE.value
        ]}

    # Apply additional filters
    if role:
        query["role"] = role.value

    if mine_id:
        query["$or"] = [
            {"mine_id": mine_id},
            {"mine_ids": mine_id}
        ]

    if search:
        query["$or"] = [
            {"username": {"$regex": search, "$options": "i"}},
            {"full_name": {"$regex": search, "$options": "i"}},
            {"email": {"$regex": search, "$options": "i"}}
        ]

    if is_active is not None:
        query["is_active"] = is_active

    cursor = db.users.find(query).skip(skip).limit(limit).sort("full_name", 1)
    users = []

    async for user in cursor:
        users.append(UserResponse(
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
        ))

    total = await db.users.count_documents(query)

    return UserList(users=users, total=total)


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: dict = Depends(get_manager_or_above)
):
    """Get user by ID."""
    db = get_database()

    try:
        user = await db.users.find_one({"_id": ObjectId(user_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID format")

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


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    current_user: dict = Depends(get_manager_or_above)
):
    """Update user details."""
    db = get_database()

    try:
        user = await db.users.find_one({"_id": ObjectId(user_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check permission to edit this user
    manager_role = UserRole(current_user.get("role"))
    target_role = UserRole(user["role"])

    if not can_manage_role(manager_role, target_role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to edit this user"
        )

    # Build update document
    update_doc = {}
    if user_data.full_name:
        update_doc["full_name"] = user_data.full_name
    if user_data.email is not None:
        update_doc["email"] = user_data.email
    if user_data.phone is not None:
        update_doc["phone"] = user_data.phone
    if user_data.mine_id is not None:
        update_doc["mine_id"] = user_data.mine_id
    if user_data.mine_ids is not None:
        update_doc["mine_ids"] = user_data.mine_ids
    if user_data.assigned_shift is not None:
        update_doc["assigned_shift"] = user_data.assigned_shift.value
    if user_data.assigned_gate_id is not None:
        update_doc["assigned_gate_id"] = user_data.assigned_gate_id
    if user_data.is_active is not None:
        update_doc["is_active"] = user_data.is_active

    if not update_doc:
        raise HTTPException(status_code=400, detail="No update data provided")

    update_doc["updated_at"] = datetime.utcnow()
    update_doc["updated_by"] = current_user.get("user_id")

    await db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": update_doc}
    )

    # Fetch updated user
    user = await db.users.find_one({"_id": ObjectId(user_id)})

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


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    current_user: dict = Depends(get_super_admin)
):
    """Delete user (Super Admin only). Soft delete by deactivating."""
    db = get_database()

    try:
        user = await db.users.find_one({"_id": ObjectId(user_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Soft delete
    await db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {
            "is_active": False,
            "deleted_at": datetime.utcnow(),
            "deleted_by": current_user.get("user_id")
        }}
    )

    return {"success": True, "message": "User deactivated successfully"}


@router.post("/{user_id}/reset-password")
async def reset_user_password(
    user_id: str,
    new_password: str,
    current_user: dict = Depends(get_manager_or_above)
):
    """Reset user password (by manager or higher)."""
    db = get_database()

    try:
        user = await db.users.find_one({"_id": ObjectId(user_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check permission
    manager_role = UserRole(current_user.get("role"))
    target_role = UserRole(user["role"])

    if not can_manage_role(manager_role, target_role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to reset this user's password"
        )

    await db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {
            "password_hash": get_password_hash(new_password),
            "password_reset_at": datetime.utcnow(),
            "password_reset_by": current_user.get("user_id")
        }}
    )

    return {"success": True, "message": "Password reset successfully"}


@router.get("/roles/available")
async def get_available_roles(current_user: dict = Depends(get_current_user)):
    """Get list of roles the current user can create."""
    user_role = UserRole(current_user.get("role"))

    available_roles = []
    for role in UserRole:
        if role == UserRole.WORKER:
            continue  # Workers are managed separately
        if can_manage_role(user_role, role):
            available_roles.append({
                "value": role.value,
                "label": role.value.replace("_", " ").title()
            })

    return {"roles": available_roles}
