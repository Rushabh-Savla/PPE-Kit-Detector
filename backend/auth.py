"""
Authentication utilities for JWT handling with Role-Based Access Control.
"""
import os
from datetime import datetime, timedelta
from typing import Optional, List
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from functools import wraps

from schemas import UserRole

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


# ==================== Password Utilities ====================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    # Truncate password to 72 bytes (bcrypt limit)
    plain_password = plain_password[:72]
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    # Truncate password to 72 bytes (bcrypt limit)
    password = password[:72]
    return pwd_context.hash(password)


# ==================== Token Utilities ====================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# ==================== Role Hierarchy ====================

# Define role hierarchy (higher number = higher authority)
ROLE_HIERARCHY = {
    UserRole.WORKER: 1,
    UserRole.SHIFT_INCHARGE: 2,
    UserRole.SAFETY_OFFICER: 3,
    UserRole.MANAGER: 4,
    UserRole.AREA_SAFETY_OFFICER: 5,
    UserRole.GENERAL_MANAGER: 6,
    UserRole.SUPER_ADMIN: 7,
}


def get_role_level(role: UserRole) -> int:
    """Get the hierarchy level of a role."""
    return ROLE_HIERARCHY.get(role, 0)


def has_higher_or_equal_role(user_role: UserRole, required_role: UserRole) -> bool:
    """Check if user has higher or equal role than required."""
    return get_role_level(user_role) >= get_role_level(required_role)


# ==================== Permission Definitions ====================

# Define what each role can do
ROLE_PERMISSIONS = {
    UserRole.SUPER_ADMIN: [
        "all",  # Super admin can do everything
    ],
    UserRole.GENERAL_MANAGER: [
        "view_all_mines",
        "view_all_reports",
        "view_kpis",
        "manage_role_access",
        "view_regulatory_reports",
        "view_financial_analytics",
        "approve_safety_rules",
    ],
    UserRole.AREA_SAFETY_OFFICER: [
        "view_assigned_mines",
        "view_comparison_dashboard",
        "view_risk_heatmaps",
        "view_trend_analytics",
        "manage_compliance_leaderboard",
        "approve_multi_site_rules",
        "view_critical_alerts",
    ],
    UserRole.MANAGER: [
        "view_mine_overview",
        "view_shift_performance",
        "view_worker_history",
        "view_escalations",
        "view_incident_summary",
        "approve_workers",
        "approve_ppe_config",
        "manage_maintenance_requests",
    ],
    UserRole.SAFETY_OFFICER: [
        "view_compliance_analytics",
        "view_trend_detection",
        "view_environment_alerts",
        "manage_ppe_rules",
        "view_worker_risk_scores",
        "trigger_disciplinary_actions",
        "view_audit_trail",
        "export_compliance_reports",
    ],
    UserRole.SHIFT_INCHARGE: [
        "view_shift_dashboard",
        "view_live_entries",
        "override_entry",
        "resolve_alarms",
        "download_shift_report",
        "mark_emergency",
        "send_alerts",
        "issue_warnings",
    ],
    UserRole.WORKER: [
        "view_own_profile",
        "view_own_compliance",
        "view_own_violations",
        "view_training_modules",
        "send_sos",
        "view_shift_timing",
    ],
}


def has_permission(user_role: UserRole, permission: str) -> bool:
    """Check if a role has a specific permission."""
    permissions = ROLE_PERMISSIONS.get(user_role, [])
    return "all" in permissions or permission in permissions


# ==================== Authentication Dependencies ====================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Dependency to get current authenticated user (any role)."""
    token = credentials.credentials
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Legacy: Dependency to get current authenticated admin."""
    return await get_current_user(credentials)


# ==================== Role-Based Dependencies ====================

def require_role(allowed_roles: List[UserRole]):
    """
    Dependency factory to require specific roles.
    Usage: Depends(require_role([UserRole.MANAGER, UserRole.SAFETY_OFFICER]))
    """
    async def role_checker(current_user: dict = Depends(get_current_user)) -> dict:
        user_role = current_user.get("role")
        if not user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User role not found in token"
            )

        try:
            user_role_enum = UserRole(user_role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid role: {user_role}"
            )

        # Super admin can access everything
        if user_role_enum == UserRole.SUPER_ADMIN:
            return current_user

        if user_role_enum not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[r.value for r in allowed_roles]}"
            )

        return current_user

    return role_checker


def require_min_role(min_role: UserRole):
    """
    Dependency factory to require minimum role level.
    Usage: Depends(require_min_role(UserRole.MANAGER))
    """
    async def role_checker(current_user: dict = Depends(get_current_user)) -> dict:
        user_role = current_user.get("role")
        if not user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User role not found in token"
            )

        try:
            user_role_enum = UserRole(user_role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid role: {user_role}"
            )

        if not has_higher_or_equal_role(user_role_enum, min_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Minimum required role: {min_role.value}"
            )

        return current_user

    return role_checker


def require_permission(permission: str):
    """
    Dependency factory to require specific permission.
    Usage: Depends(require_permission("manage_ppe_rules"))
    """
    async def permission_checker(current_user: dict = Depends(get_current_user)) -> dict:
        user_role = current_user.get("role")
        if not user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User role not found in token"
            )

        try:
            user_role_enum = UserRole(user_role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid role: {user_role}"
            )

        if not has_permission(user_role_enum, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required permission: {permission}"
            )

        return current_user

    return permission_checker


# ==================== Specific Role Dependencies ====================

async def get_super_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Require super admin role."""
    if current_user.get("role") != UserRole.SUPER_ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin access required"
        )
    return current_user


async def get_general_manager(current_user: dict = Depends(get_current_user)) -> dict:
    """Require general manager or higher."""
    user_role = current_user.get("role")
    allowed = [UserRole.GENERAL_MANAGER.value, UserRole.SUPER_ADMIN.value]
    if user_role not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="General manager access required"
        )
    return current_user


async def get_area_safety_officer_or_above(current_user: dict = Depends(get_current_user)) -> dict:
    """Require area safety officer or higher."""
    user_role = current_user.get("role")
    try:
        role_enum = UserRole(user_role)
        if not has_higher_or_equal_role(role_enum, UserRole.AREA_SAFETY_OFFICER):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Area safety officer or higher access required"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid role"
        )
    return current_user


async def get_manager_or_above(current_user: dict = Depends(get_current_user)) -> dict:
    """Require manager or higher."""
    user_role = current_user.get("role")
    try:
        role_enum = UserRole(user_role)
        if not has_higher_or_equal_role(role_enum, UserRole.MANAGER):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Manager or higher access required"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid role"
        )
    return current_user


async def get_safety_officer_or_above(current_user: dict = Depends(get_current_user)) -> dict:
    """Require safety officer or higher."""
    user_role = current_user.get("role")
    try:
        role_enum = UserRole(user_role)
        if not has_higher_or_equal_role(role_enum, UserRole.SAFETY_OFFICER):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Safety officer or higher access required"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid role"
        )
    return current_user


async def get_shift_incharge_or_above(current_user: dict = Depends(get_current_user)) -> dict:
    """Require shift incharge or higher."""
    user_role = current_user.get("role")
    try:
        role_enum = UserRole(user_role)
        if not has_higher_or_equal_role(role_enum, UserRole.SHIFT_INCHARGE):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Shift incharge or higher access required"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid role"
        )
    return current_user


async def get_worker(current_user: dict = Depends(get_current_user)) -> dict:
    """Get worker user (from worker login)."""
    if current_user.get("user_type") != "worker":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Worker access required"
        )
    return current_user


# ==================== Mine Access Check ====================

def check_mine_access(user: dict, mine_id: str) -> bool:
    """
    Check if user has access to a specific mine.
    - Super Admin, General Manager: All mines
    - Area Safety Officer: Assigned mines (mine_ids list)
    - Manager, Safety Officer, Shift Incharge: Their assigned mine (mine_id)
    """
    user_role = user.get("role")

    if user_role in [UserRole.SUPER_ADMIN.value, UserRole.GENERAL_MANAGER.value]:
        return True

    if user_role == UserRole.AREA_SAFETY_OFFICER.value:
        mine_ids = user.get("mine_ids", [])
        return mine_id in mine_ids

    user_mine_id = user.get("mine_id")
    return user_mine_id == mine_id


async def require_mine_access(mine_id: str, current_user: dict = Depends(get_current_user)) -> dict:
    """Dependency to check mine access."""
    if not check_mine_access(current_user, mine_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this mine"
        )
    return current_user
