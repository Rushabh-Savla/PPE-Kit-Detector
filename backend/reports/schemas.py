"""
Pydantic schemas for report generation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, EmailStr, Field


class ReportType(str, Enum):
    """Available report types."""
    # Shift Incharge Reports
    SHIFT_SUMMARY = "shift_summary"
    SHIFT_HANDOVER = "shift_handover"
    ENTRY_EXIT_LOG = "entry_exit_log"
    ALERT_RESOLUTION = "alert_resolution"

    # Safety Officer Reports
    WEEKLY_COMPLIANCE = "weekly_compliance"
    HIGH_RISK_WORKERS = "high_risk_workers"
    ZONE_RISK_ANALYSIS = "zone_risk_analysis"
    VIOLATION_TRENDS = "violation_trends"

    # Manager Reports
    DAILY_OPERATIONS = "daily_operations"
    SHIFT_PERFORMANCE = "shift_performance"
    WORKER_RANKINGS = "worker_rankings"
    MONTHLY_SUMMARY = "monthly_summary"
    ESCALATION_REPORT = "escalation_report"

    # Area Safety Officer Reports
    MINE_COMPARISON = "mine_comparison"
    RISK_HEATMAP = "risk_heatmap"
    CRITICAL_INCIDENTS = "critical_incidents"
    COMPLIANCE_LEADERBOARD = "compliance_leaderboard"

    # General Manager Reports
    EXECUTIVE_SUMMARY = "executive_summary"
    KPI_DASHBOARD = "kpi_dashboard"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    FINANCIAL_IMPACT = "financial_impact"

    # Worker Reports
    COMPLIANCE_CARD = "compliance_card"
    WORKER_MONTHLY = "worker_monthly"


class ReportFormat(str, Enum):
    """Output formats for reports."""
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"


class ScheduleFrequency(str, Enum):
    """Schedule frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class DateRangeType(str, Enum):
    """Predefined date range options."""
    PREVIOUS_SHIFT = "previous_shift"
    PREVIOUS_DAY = "previous_day"
    PREVIOUS_WEEK = "previous_week"
    PREVIOUS_MONTH = "previous_month"
    CUSTOM = "custom"


class EmailRecipient(BaseModel):
    """Email recipient model."""
    email: EmailStr
    name: str
    type: str = "to"  # to, cc, bcc


class ReportScheduleConfig(BaseModel):
    """Configuration for scheduled reports."""
    format: List[ReportFormat] = [ReportFormat.PDF]
    date_range: DateRangeType = DateRangeType.PREVIOUS_DAY
    include_charts: bool = True
    filters: Dict[str, Any] = {}


class ScheduleConfig(BaseModel):
    """Schedule timing configuration."""
    frequency: ScheduleFrequency
    time: str = "06:00"  # HH:MM format UTC
    day_of_week: Optional[int] = None  # 1-7 for weekly
    day_of_month: Optional[int] = None  # 1-31 for monthly


class ReportScheduleCreate(BaseModel):
    """Request model for creating a report schedule."""
    name: str = Field(..., min_length=1, max_length=100)
    report_type: ReportType
    mine_id: Optional[str] = None
    schedule: ScheduleConfig
    recipients: List[EmailRecipient]
    config: ReportScheduleConfig = ReportScheduleConfig()


class ReportScheduleUpdate(BaseModel):
    """Request model for updating a report schedule."""
    name: Optional[str] = None
    schedule: Optional[ScheduleConfig] = None
    recipients: Optional[List[EmailRecipient]] = None
    config: Optional[ReportScheduleConfig] = None
    is_active: Optional[bool] = None


class ReportScheduleResponse(BaseModel):
    """Response model for report schedule."""
    id: str
    name: str
    report_type: ReportType
    role_target: str
    mine_id: Optional[str]
    schedule: ScheduleConfig
    recipients: List[EmailRecipient]
    config: ReportScheduleConfig
    is_active: bool
    next_run: Optional[datetime]
    last_run: Optional[datetime]
    run_count: int
    created_at: datetime
    created_by: str


class GenerateReportRequest(BaseModel):
    """Request model for on-demand report generation."""
    report_type: ReportType
    format: ReportFormat = ReportFormat.PDF
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    mine_id: Optional[str] = None
    filters: Dict[str, Any] = {}
    delivery: str = "download"  # download or email
    recipients: Optional[List[EmailRecipient]] = None


class GenerateReportResponse(BaseModel):
    """Response model for report generation."""
    success: bool
    report_id: str
    download_url: Optional[str] = None
    file_name: Optional[str] = None
    format: Optional[ReportFormat] = None
    generated_at: Optional[str] = None
    message: str


class ReportTypeInfo(BaseModel):
    """Information about a report type."""
    id: ReportType
    name: str
    description: str
    available_formats: List[ReportFormat]
    min_role: str
    parameters: List[str]


class ReportTypesResponse(BaseModel):
    """Response model for available report types."""
    report_types: List[ReportTypeInfo]


# Report Data Models

class ShiftSummaryData(BaseModel):
    """Data model for shift summary report."""
    shift_info: Dict[str, Any]
    workers_entered: int
    workers_exited: int
    currently_inside: int
    compliance_rate: float
    violations_count: int
    alerts_resolved: int
    alerts_pending: int
    entry_exit_logs: List[Dict[str, Any]]
    hourly_breakdown: List[Dict[str, Any]]


class ComplianceReportData(BaseModel):
    """Data model for compliance report."""
    overall_compliance: float
    compliance_trend: List[Dict[str, Any]]
    violations_by_type: Dict[str, int]
    violations_by_zone: List[Dict[str, Any]]
    high_risk_workers: List[Dict[str, Any]]
    zone_analysis: List[Dict[str, Any]]


class OperationsReportData(BaseModel):
    """Data model for operations report."""
    total_workers: int
    active_workers: int
    compliance_rate: float
    shift_performance: Dict[str, Any]
    worker_rankings: List[Dict[str, Any]]
    escalations: List[Dict[str, Any]]
    attendance_trends: List[Dict[str, Any]]


class MineComparisonData(BaseModel):
    """Data model for mine comparison report."""
    mines: List[Dict[str, Any]]
    compliance_comparison: List[Dict[str, Any]]
    risk_heatmap: List[Dict[str, Any]]
    critical_incidents: List[Dict[str, Any]]


class ExecutiveReportData(BaseModel):
    """Data model for executive report."""
    kpis: Dict[str, Any]
    regulatory_status: Dict[str, Any]
    financial_metrics: Dict[str, Any]
    mine_performance: List[Dict[str, Any]]
    strategic_alerts: List[Dict[str, Any]]


class WorkerComplianceCardData(BaseModel):
    """Data model for worker compliance card."""
    worker_info: Dict[str, Any]
    compliance_score: float
    total_violations: int
    streak_days: int
    badges: List[str]
    recent_violations: List[Dict[str, Any]]
    attendance_summary: Dict[str, Any]


# Role-based report type mappings
ROLE_REPORT_TYPES = {
    "shift_incharge": [
        ReportType.SHIFT_SUMMARY,
        ReportType.SHIFT_HANDOVER,
        ReportType.ENTRY_EXIT_LOG,
        ReportType.ALERT_RESOLUTION,
    ],
    "safety_officer": [
        ReportType.WEEKLY_COMPLIANCE,
        ReportType.HIGH_RISK_WORKERS,
        ReportType.ZONE_RISK_ANALYSIS,
        ReportType.VIOLATION_TRENDS,
        ReportType.SHIFT_SUMMARY,
        ReportType.ENTRY_EXIT_LOG,
    ],
    "manager": [
        ReportType.DAILY_OPERATIONS,
        ReportType.SHIFT_PERFORMANCE,
        ReportType.WORKER_RANKINGS,
        ReportType.MONTHLY_SUMMARY,
        ReportType.ESCALATION_REPORT,
        ReportType.WEEKLY_COMPLIANCE,
        ReportType.SHIFT_SUMMARY,
    ],
    "area_safety_officer": [
        ReportType.MINE_COMPARISON,
        ReportType.RISK_HEATMAP,
        ReportType.CRITICAL_INCIDENTS,
        ReportType.COMPLIANCE_LEADERBOARD,
        ReportType.WEEKLY_COMPLIANCE,
        ReportType.DAILY_OPERATIONS,
    ],
    "general_manager": [
        ReportType.EXECUTIVE_SUMMARY,
        ReportType.KPI_DASHBOARD,
        ReportType.REGULATORY_COMPLIANCE,
        ReportType.FINANCIAL_IMPACT,
        ReportType.MINE_COMPARISON,
        ReportType.MONTHLY_SUMMARY,
    ],
    "super_admin": list(ReportType),  # All report types
    "worker": [
        ReportType.COMPLIANCE_CARD,
        ReportType.WORKER_MONTHLY,
    ],
}


# Report type metadata
REPORT_TYPE_INFO = {
    ReportType.SHIFT_SUMMARY: {
        "name": "Shift Summary Report",
        "description": "Summary of shift operations including entries, exits, compliance, and alerts",
        "parameters": ["shift", "gate_id", "date"],
    },
    ReportType.SHIFT_HANDOVER: {
        "name": "Shift Handover Report",
        "description": "Handover notes including outstanding issues and workers currently inside",
        "parameters": ["shift", "date"],
    },
    ReportType.ENTRY_EXIT_LOG: {
        "name": "Entry/Exit Log",
        "description": "Detailed log of all gate entries and exits with PPE status",
        "parameters": ["date_range", "gate_id", "worker_id"],
    },
    ReportType.ALERT_RESOLUTION: {
        "name": "Alert Resolution Report",
        "description": "Summary of alerts triggered and resolved during the period",
        "parameters": ["date_range", "severity"],
    },
    ReportType.WEEKLY_COMPLIANCE: {
        "name": "Weekly Compliance Report",
        "description": "Compliance trends and violation breakdown over the week",
        "parameters": ["date_range", "zone_id"],
    },
    ReportType.HIGH_RISK_WORKERS: {
        "name": "High-Risk Workers Report",
        "description": "Workers with low compliance scores and intervention recommendations",
        "parameters": ["threshold", "include_predictions"],
    },
    ReportType.ZONE_RISK_ANALYSIS: {
        "name": "Zone Risk Analysis",
        "description": "Risk levels and violation patterns by zone",
        "parameters": ["date_range"],
    },
    ReportType.VIOLATION_TRENDS: {
        "name": "Violation Trends Report",
        "description": "Month-over-month violation comparison and repeat offenders",
        "parameters": ["date_range", "group_by"],
    },
    ReportType.DAILY_OPERATIONS: {
        "name": "Daily Operations Summary",
        "description": "Daily operational metrics including attendance and compliance",
        "parameters": ["date"],
    },
    ReportType.SHIFT_PERFORMANCE: {
        "name": "Shift Performance Report",
        "description": "Comparison of day, afternoon, and night shift performance",
        "parameters": ["date_range"],
    },
    ReportType.WORKER_RANKINGS: {
        "name": "Worker Rankings Report",
        "description": "Top performers and workers needing improvement",
        "parameters": ["limit", "include_predictions"],
    },
    ReportType.MONTHLY_SUMMARY: {
        "name": "Monthly Summary Report",
        "description": "Comprehensive monthly KPIs and trend analysis",
        "parameters": ["month", "year"],
    },
    ReportType.ESCALATION_REPORT: {
        "name": "Escalation Report",
        "description": "Pending and resolved escalations with resolution times",
        "parameters": ["date_range", "status"],
    },
    ReportType.MINE_COMPARISON: {
        "name": "Mine Comparison Report",
        "description": "Side-by-side comparison of multiple mines",
        "parameters": ["mine_ids", "date_range"],
    },
    ReportType.RISK_HEATMAP: {
        "name": "Risk Heatmap Report",
        "description": "Zone-level risk visualization across mines",
        "parameters": ["mine_ids", "date_range"],
    },
    ReportType.CRITICAL_INCIDENTS: {
        "name": "Critical Incidents Report",
        "description": "High and critical severity incidents across mines",
        "parameters": ["mine_ids", "date_range", "severity"],
    },
    ReportType.COMPLIANCE_LEADERBOARD: {
        "name": "Compliance Leaderboard",
        "description": "Ranking of mines and zones by compliance performance",
        "parameters": ["date_range"],
    },
    ReportType.EXECUTIVE_SUMMARY: {
        "name": "Executive Summary",
        "description": "Organization-wide KPIs and strategic insights",
        "parameters": ["date_range"],
    },
    ReportType.KPI_DASHBOARD: {
        "name": "KPI Dashboard Report",
        "description": "Key performance indicators with trend analysis",
        "parameters": ["date_range", "compare_previous"],
    },
    ReportType.REGULATORY_COMPLIANCE: {
        "name": "Regulatory Compliance Report",
        "description": "Compliance status against regulatory thresholds",
        "parameters": ["date_range"],
    },
    ReportType.FINANCIAL_IMPACT: {
        "name": "Financial Impact Report",
        "description": "Cost savings and incident prevention metrics",
        "parameters": ["date_range"],
    },
    ReportType.COMPLIANCE_CARD: {
        "name": "Personal Compliance Card",
        "description": "Individual worker compliance summary and badges",
        "parameters": ["worker_id"],
    },
    ReportType.WORKER_MONTHLY: {
        "name": "Worker Monthly Summary",
        "description": "Monthly personal performance and violation history",
        "parameters": ["worker_id", "month", "year"],
    },
}
