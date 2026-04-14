"""
Pydantic schemas for request/response validation.
"""
from datetime import datetime, time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# ==================== Enums ====================

class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    GENERAL_MANAGER = "general_manager"
    AREA_SAFETY_OFFICER = "area_safety_officer"
    MANAGER = "manager"
    SAFETY_OFFICER = "safety_officer"
    SHIFT_INCHARGE = "shift_incharge"
    WORKER = "worker"


class ShiftType(str, Enum):
    DAY = "day"           # 6:00 AM - 2:00 PM
    AFTERNOON = "afternoon"  # 2:00 PM - 10:00 PM
    NIGHT = "night"       # 10:00 PM - 6:00 AM


class GateType(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    BOTH = "both"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class EntryStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    OVERRIDE = "override"


# ==================== Mine/Site Schemas ====================

class ZoneCreate(BaseModel):
    name: str
    description: Optional[str] = None
    risk_level: str = "normal"  # low, normal, high, critical
    coordinates: Optional[Dict[str, Any]] = None  # For visual map


class ZoneResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    risk_level: str
    coordinates: Optional[Dict[str, Any]] = None
    worker_count: int = 0


class GateCreate(BaseModel):
    name: str
    gate_type: GateType = GateType.BOTH
    zone_id: Optional[str] = None
    location: Optional[str] = None
    has_camera: bool = True


class GateResponse(BaseModel):
    id: str
    name: str
    gate_type: GateType
    zone_id: Optional[str] = None
    location: Optional[str] = None
    has_camera: bool
    is_active: bool = True


class MineCreate(BaseModel):
    name: str
    location: str
    description: Optional[str] = None


class MineResponse(BaseModel):
    id: str
    name: str
    location: str
    description: Optional[str] = None
    zones: List[ZoneResponse] = []
    gates: List[GateResponse] = []
    created_at: datetime
    is_active: bool = True


class MineList(BaseModel):
    mines: List[MineResponse]
    total: int


# ==================== Shift Schemas ====================

class ShiftInfo(BaseModel):
    shift_type: ShiftType
    start_time: str  # "06:00", "14:00", "22:00"
    end_time: str    # "14:00", "22:00", "06:00"
    name: str        # "Day Shift", "Afternoon Shift", "Night Shift"


SHIFT_DEFINITIONS = {
    ShiftType.DAY: ShiftInfo(
        shift_type=ShiftType.DAY,
        start_time="06:00",
        end_time="14:00",
        name="Day Shift"
    ),
    ShiftType.AFTERNOON: ShiftInfo(
        shift_type=ShiftType.AFTERNOON,
        start_time="14:00",
        end_time="22:00",
        name="Afternoon Shift"
    ),
    ShiftType.NIGHT: ShiftInfo(
        shift_type=ShiftType.NIGHT,
        start_time="22:00",
        end_time="06:00",
        name="Night Shift"
    ),
}


# ==================== User Schemas ====================

class UserBase(BaseModel):
    username: str
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    role: UserRole


class UserCreate(UserBase):
    password: str
    mine_id: Optional[str] = None  # For mine-specific roles
    mine_ids: Optional[List[str]] = None  # For Area Safety Officer (multiple mines)
    assigned_shift: Optional[ShiftType] = None  # For Shift Incharge
    assigned_gate_id: Optional[str] = None  # For Shift Incharge


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    mine_id: Optional[str] = None
    mine_ids: Optional[List[str]] = None
    assigned_shift: Optional[ShiftType] = None
    assigned_gate_id: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    id: str
    username: str
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    role: UserRole
    mine_id: Optional[str] = None
    mine_ids: Optional[List[str]] = None
    assigned_shift: Optional[ShiftType] = None
    assigned_gate_id: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None


class UserList(BaseModel):
    users: List[UserResponse]
    total: int


# ==================== Worker Schemas (Extended Employee) ====================

class WorkerCreate(BaseModel):
    employee_id: str
    name: str
    password: str
    department: Optional[str] = None
    mine_id: str
    zone_id: Optional[str] = None
    assigned_shift: ShiftType = ShiftType.DAY
    phone: Optional[str] = None
    emergency_contact: Optional[str] = None


class WorkerUpdate(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    mine_id: Optional[str] = None
    zone_id: Optional[str] = None
    assigned_shift: Optional[ShiftType] = None
    phone: Optional[str] = None
    emergency_contact: Optional[str] = None
    is_active: Optional[bool] = None


class WorkerResponse(BaseModel):
    id: str
    employee_id: str
    name: str
    department: Optional[str] = None
    mine_id: str
    mine_name: Optional[str] = None
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    assigned_shift: ShiftType
    phone: Optional[str] = None
    emergency_contact: Optional[str] = None
    face_registered: bool = False
    is_active: bool = True
    created_at: datetime
    compliance_score: float = 100.0
    total_violations: int = 0
    badges: List[str] = []


class WorkerList(BaseModel):
    workers: List[WorkerResponse]
    total: int


# ==================== Auth Schemas ====================

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class WorkerLoginRequest(BaseModel):
    employee_id: str
    password: str


class WorkerTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    worker: WorkerResponse


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


# ==================== Gate Entry Schemas ====================

class GateEntryCreate(BaseModel):
    gate_id: str
    worker_id: Optional[str] = None
    entry_type: str = "entry"  # entry or exit
    ppe_status: Dict[str, bool] = {}  # {"helmet": True, "vest": True, ...}
    violations: List[str] = []
    image: Optional[str] = None
    override_by: Optional[str] = None
    override_reason: Optional[str] = None


class GateEntryResponse(BaseModel):
    id: str
    gate_id: str
    gate_name: Optional[str] = None
    worker_id: Optional[str] = None
    worker_name: Optional[str] = None
    employee_id: Optional[str] = None
    entry_type: str
    status: EntryStatus
    ppe_status: Dict[str, bool]
    violations: List[str]
    timestamp: datetime
    shift: ShiftType
    override_by: Optional[str] = None
    override_reason: Optional[str] = None


class GateEntryList(BaseModel):
    entries: List[GateEntryResponse]
    total: int


# ==================== Alert Schemas ====================

class AlertCreate(BaseModel):
    alert_type: str  # ppe_violation, methane, emergency, etc.
    severity: AlertSeverity
    message: str
    mine_id: str
    zone_id: Optional[str] = None
    gate_id: Optional[str] = None
    worker_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AlertResponse(BaseModel):
    id: str
    alert_type: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    mine_id: str
    mine_name: Optional[str] = None
    zone_id: Optional[str] = None
    gate_id: Optional[str] = None
    worker_id: Optional[str] = None
    worker_name: Optional[str] = None
    created_at: datetime
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


class AlertList(BaseModel):
    alerts: List[AlertResponse]
    total: int


# ==================== PPE Configuration Schemas ====================

class PPEItemConfig(BaseModel):
    name: str
    required: bool = True
    description: Optional[str] = None


class PPEConfigCreate(BaseModel):
    mine_id: str
    zone_id: Optional[str] = None  # None means mine-wide
    required_items: List[PPEItemConfig]


class PPEConfigResponse(BaseModel):
    id: str
    mine_id: str
    zone_id: Optional[str] = None
    required_items: List[PPEItemConfig]
    created_at: datetime
    updated_at: datetime
    updated_by: Optional[str] = None


# ==================== Incident/Warning Schemas ====================

class WarningCreate(BaseModel):
    worker_id: str
    warning_type: str  # ppe_violation, late_arrival, behavior, etc.
    description: str
    severity: str = "minor"  # minor, moderate, severe


class WarningResponse(BaseModel):
    id: str
    worker_id: str
    worker_name: str
    employee_id: str
    warning_type: str
    description: str
    severity: str
    issued_by: str
    issued_by_name: str
    issued_at: datetime
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None


class IncidentCreate(BaseModel):
    incident_type: str
    description: str
    mine_id: str
    zone_id: Optional[str] = None
    severity: AlertSeverity
    workers_involved: List[str] = []


class IncidentResponse(BaseModel):
    id: str
    incident_type: str
    description: str
    mine_id: str
    mine_name: str
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    severity: AlertSeverity
    status: str  # open, investigating, resolved
    workers_involved: List[str]
    reported_by: str
    reported_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


# ==================== Dashboard/Stats Schemas ====================

class ShiftDashboardStats(BaseModel):
    """Stats for Shift Incharge dashboard"""
    current_shift: ShiftType
    shift_start: str
    shift_end: str
    total_workers_expected: int
    workers_entered: int
    workers_exited: int
    currently_inside: int
    ppe_compliant: int
    ppe_non_compliant: int
    violations_this_shift: int
    pending_alerts: int
    recent_entries: List[GateEntryResponse]


class SafetyOfficerStats(BaseModel):
    """Stats for Safety Officer dashboard"""
    compliance_rate_today: float
    compliance_rate_week: float
    compliance_rate_month: float
    total_violations_today: int
    total_violations_week: int
    violation_trends: Dict[str, int]  # {"helmet": 5, "vest": 3, ...}
    high_risk_workers: List[WorkerResponse]
    recent_alerts: List[AlertResponse]
    zone_risk_scores: Dict[str, float]


class ManagerStats(BaseModel):
    """Stats for Manager dashboard"""
    mine_id: str
    mine_name: str
    total_workers: int
    active_workers_today: int
    compliance_rate: float
    shift_performance: Dict[str, float]  # {"day": 95.5, "afternoon": 92.0, ...}
    pending_escalations: int
    pending_approvals: int
    entry_delays_avg_minutes: float
    top_compliant_workers: List[WorkerResponse]


class AreaSafetyOfficerStats(BaseModel):
    """Stats for Area Safety Officer dashboard"""
    mines_overview: List[Dict[str, Any]]  # [{mine_id, name, compliance_rate, violations}]
    overall_compliance_rate: float
    comparison_data: Dict[str, Any]
    risk_heatmap_data: List[Dict[str, Any]]
    critical_alerts: List[AlertResponse]


class GeneralManagerStats(BaseModel):
    """Stats for General Manager dashboard"""
    organization_compliance_rate: float
    total_mines: int
    total_workers: int
    kpi_summary: Dict[str, Any]
    productivity_vs_safety: Dict[str, Any]
    regulatory_compliance_status: Dict[str, Any]
    cost_savings_estimate: float
    strategic_alerts: List[Dict[str, Any]]


class WorkerDashboardStats(BaseModel):
    """Stats for Worker dashboard (mobile PWA)"""
    worker: WorkerResponse
    compliance_score: float
    current_streak_days: int
    total_entries: int
    total_violations: int
    recent_violations: List[Dict[str, Any]]
    badges_earned: List[str]
    upcoming_shift: ShiftInfo
    training_modules_pending: int
    notifications: List[Dict[str, Any]]


# ==================== Report Schemas ====================

class ComplianceReportRequest(BaseModel):
    start_date: str
    end_date: str
    mine_id: Optional[str] = None
    shift: Optional[ShiftType] = None
    group_by: str = "day"  # day, week, month


class ComplianceReport(BaseModel):
    start_date: str
    end_date: str
    mine_id: Optional[str] = None
    mine_name: Optional[str] = None
    overall_compliance_rate: float
    data_points: List[Dict[str, Any]]
    violation_breakdown: Dict[str, int]
    worker_compliance_summary: List[Dict[str, Any]]


# ==================== Mine Visualization Schemas ====================

class MineVisualizationData(BaseModel):
    """Data for rendering the mine visual representation"""
    mine_id: str
    mine_name: str
    zones: List[Dict[str, Any]]  # Zone data with coordinates
    gates: List[Dict[str, Any]]  # Gate positions
    workers_positions: List[Dict[str, Any]]  # Current worker locations
    risk_zones: List[Dict[str, Any]]  # Highlighted risk areas
    statistics: Dict[str, Any]


# ==================== Legacy Schemas (for backward compatibility) ====================

class EmployeeCreate(BaseModel):
    name: str
    employee_id: str
    department: Optional[str] = None


class EmployeeResponse(BaseModel):
    id: str
    name: str
    employee_id: str
    department: Optional[str] = None
    created_at: datetime
    face_registered: bool = False


class EmployeeList(BaseModel):
    employees: List[EmployeeResponse]
    total: int


class AttendanceRecord(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    type: str
    timestamp: datetime
    date: str
    image: Optional[str] = None


class AttendanceCreate(BaseModel):
    employee_id: str
    type: str


class AttendanceResponse(BaseModel):
    success: bool
    message: str
    record: Optional[AttendanceRecord] = None


class AttendanceList(BaseModel):
    records: List[AttendanceRecord]
    total: int


class DailyAttendance(BaseModel):
    employee_id: str
    employee_name: str
    department: Optional[str]
    date: str
    check_ins: List[datetime]
    check_outs: List[datetime]
    total_hours: Optional[float] = None


class PPEViolation(BaseModel):
    id: str
    employee_id: Optional[str] = None
    employee_name: Optional[str] = None
    violations: List[Dict[str, Any]]
    timestamp: datetime
    image: Optional[str] = None
    location: Optional[str] = None


class PPEViolationCreate(BaseModel):
    employee_id: Optional[str] = None
    violations: List[Dict[str, Any]]
    location: Optional[str] = None


class PPEViolationList(BaseModel):
    violations: List[PPEViolation]
    total: int


class DetectionResult(BaseModel):
    success: bool
    image: str
    detections: Dict[str, Any]
    violations_logged: bool = False
    attendance_marked: bool = False


class AdminCreate(BaseModel):
    username: str
    password: str


class AdminLogin(BaseModel):
    username: str
    password: str


class DashboardStats(BaseModel):
    total_employees: int
    present_today: int
    absent_today: int
    violations_today: int
    violations_this_week: int
    compliance_rate: float
    recent_attendance: List[AttendanceRecord]
    recent_violations: List[PPEViolation]


class AttendanceReport(BaseModel):
    start_date: str
    end_date: str
    records: List[DailyAttendance]
    summary: Dict[str, Any]


class ViolationReport(BaseModel):
    start_date: str
    end_date: str
    violations: List[PPEViolation]
    summary: Dict[str, Any]


# ==================== Prediction Schemas ====================

class RiskCategory(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttendancePattern(str, Enum):
    REGULAR = "regular"
    IRREGULAR = "irregular"
    DECLINING = "declining"


class RiskFactor(BaseModel):
    factor: str
    impact: float
    description: str


class WorkerPredictionResponse(BaseModel):
    worker_id: str
    employee_id: str
    worker_name: Optional[str] = None
    prediction_date: datetime

    # Risk Scores (0-100)
    overall_risk_score: float
    risk_category: RiskCategory
    violation_risk_score: float
    attendance_risk_score: float
    compliance_trend_score: float

    # Predictions
    predicted_violations_count: int
    predicted_absent_days: int
    high_risk_ppe_items: List[str]

    # Classification
    requires_intervention: bool
    attendance_pattern: AttendancePattern
    consecutive_absence_risk: float
    attendance_rate_30d: float

    # Explainability
    risk_factors: List[RiskFactor]
    confidence: float

    # Metadata
    model_version: str
    created_at: datetime
    expires_at: datetime


class AtRiskWorkerSummary(BaseModel):
    worker_id: str
    employee_id: str
    worker_name: str
    risk_score: float
    risk_category: RiskCategory
    main_issue: str
    requires_intervention: bool


class AtRiskWorkersSummary(BaseModel):
    total_at_risk: int
    by_category: Dict[str, int]
    workers: List[AtRiskWorkerSummary]


class PredictionTrends(BaseModel):
    trends: Dict[str, Dict[str, Dict[str, Any]]]  # date -> category -> {count, avg_score}


class BatchPredictionResult(BaseModel):
    worker_id: str
    status: str  # "success" or "error"
    risk_category: Optional[str] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    total_workers: int
    successful: int
    failed: int
    results: List[BatchPredictionResult]
