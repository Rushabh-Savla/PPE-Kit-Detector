"""
Report API Router.

Provides endpoints for:
- Listing available report types by role
- Generating on-demand reports (PDF, Excel, CSV)
- Managing report schedules
- Downloading generated reports
"""

from datetime import datetime, timedelta
from typing import Optional, List
from io import BytesIO
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from bson import ObjectId
import uuid

from database import get_database
from auth import (
    get_current_user,
    get_shift_incharge_or_above,
    get_safety_officer_or_above,
    get_manager_or_above,
    check_mine_access,
    UserRole
)

from .schemas import (
    ReportType, ReportFormat, ReportTypeInfo, ReportTypesResponse,
    GenerateReportRequest, GenerateReportResponse,
    ReportScheduleCreate, ReportScheduleUpdate, ReportScheduleResponse,
    ROLE_REPORT_TYPES, REPORT_TYPE_INFO,
    ScheduleFrequency, EmailRecipient
)
from .services.pdf_generator import PDFGenerator
from .services.excel_generator import ExcelGenerator
from .services.email_service import get_email_service
from .templates import (
    ShiftInchargeTemplate, SafetyOfficerTemplate, ManagerTemplate,
    AreaSafetyOfficerTemplate, GeneralManagerTemplate, WorkerTemplate,
    EmergencyIncidentTemplate
)
from .templates.shift_incharge import ShiftHandoverTemplate
from .templates.safety_officer import ViolationTrendsTemplate, HighRiskWorkersTemplate
from .templates.manager import MonthlySummaryTemplate, ShiftPerformanceTemplate
from .templates.area_safety_officer import RiskHeatmapTemplate, CriticalIncidentsTemplate
from .templates.general_manager import KPIDashboardTemplate, FinancialImpactTemplate
from .templates.worker import WorkerMonthlyTemplate


router = APIRouter(prefix="/reports", tags=["Reports"])

# In-memory cache for generated reports (in production, use Redis or file storage)
_report_cache = {}


# Template registry
TEMPLATE_REGISTRY = {
    ReportType.SHIFT_SUMMARY: ShiftInchargeTemplate,
    ReportType.SHIFT_HANDOVER: ShiftHandoverTemplate,
    ReportType.WEEKLY_COMPLIANCE: SafetyOfficerTemplate,
    ReportType.VIOLATION_TRENDS: ViolationTrendsTemplate,
    ReportType.HIGH_RISK_WORKERS: HighRiskWorkersTemplate,
    ReportType.DAILY_OPERATIONS: ManagerTemplate,
    ReportType.MONTHLY_SUMMARY: MonthlySummaryTemplate,
    ReportType.SHIFT_PERFORMANCE: ShiftPerformanceTemplate,
    ReportType.MINE_COMPARISON: AreaSafetyOfficerTemplate,
    ReportType.RISK_HEATMAP: RiskHeatmapTemplate,
    ReportType.CRITICAL_INCIDENTS: CriticalIncidentsTemplate,
    ReportType.EXECUTIVE_SUMMARY: GeneralManagerTemplate,
    ReportType.KPI_DASHBOARD: KPIDashboardTemplate,
    ReportType.FINANCIAL_IMPACT: FinancialImpactTemplate,
    ReportType.COMPLIANCE_CARD: WorkerTemplate,
    ReportType.WORKER_MONTHLY: WorkerMonthlyTemplate,
}


# ==================== Report Types ====================

@router.get("/types", response_model=ReportTypesResponse)
async def get_report_types(
    current_user: dict = Depends(get_current_user)
):
    """
    Get available report types for the current user's role.
    """
    user_role = current_user.get("role", "worker")

    # Get report types for this role
    available_types = ROLE_REPORT_TYPES.get(user_role, [])

    # Build response
    report_types = []
    for report_type in available_types:
        info = REPORT_TYPE_INFO.get(report_type, {})
        report_types.append(ReportTypeInfo(
            id=report_type,
            name=info.get("name", report_type.value),
            description=info.get("description", ""),
            available_formats=[ReportFormat.PDF, ReportFormat.EXCEL, ReportFormat.CSV],
            min_role=user_role,
            parameters=info.get("parameters", [])
        ))

    return ReportTypesResponse(report_types=report_types)


# ==================== Report Generation ====================

@router.post("/generate", response_model=GenerateReportResponse)
async def generate_report(
    request: GenerateReportRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate a report on-demand.

    Returns a report ID for download or sends via email.
    """
    db = get_database()

    # Verify user has access to this report type
    user_role = current_user.get("role", "worker")
    available_types = ROLE_REPORT_TYPES.get(user_role, [])

    if request.report_type not in available_types:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this report type"
        )

    # Verify mine access if mine_id provided
    if request.mine_id and not check_mine_access(current_user, request.mine_id):
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this mine"
        )

    # Get template
    template_class = TEMPLATE_REGISTRY.get(request.report_type)
    if not template_class:
        raise HTTPException(
            status_code=400,
            detail=f"No template found for report type: {request.report_type}"
        )

    template = template_class(db)

    # Parse dates
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d") + timedelta(days=1)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD"
        )

    # Aggregate data - ensure filters is always a dict
    filters = request.filters if isinstance(request.filters, dict) else {}
    data = await template.aggregate_data(
        start_date=start_date,
        end_date=end_date,
        mine_id=request.mine_id,
        filters=filters
    )

    # Generate report
    report_id = str(uuid.uuid4())

    if request.format == ReportFormat.PDF:
        generator = PDFGenerator()
        buffer = await generator.generate(template, data)
        content_type = "application/pdf"
        extension = "pdf"
    elif request.format == ReportFormat.EXCEL:
        generator = ExcelGenerator()
        buffer = await generator.generate(template, data)
        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        extension = "xlsx"
    else:  # CSV
        # Generate CSV from data
        buffer = _generate_csv(template, data)
        content_type = "text/csv"
        extension = "csv"

    # Store in cache
    filename = f"{request.report_type.value}_{request.start_date}_to_{request.end_date}.{extension}"
    _report_cache[report_id] = {
        "buffer": buffer,
        "filename": filename,
        "content_type": content_type,
        "created_at": datetime.utcnow(),
        "format": extension
    }

    # Handle delivery
    if request.delivery == "email" and request.recipients:
        background_tasks.add_task(
            _send_report_email,
            recipients=request.recipients,
            report_name=template.report_name,
            date_range=data.get("date_range", ""),
            buffer=buffer,
            filename=filename,
            content_type=content_type,
            summary_metrics=template.get_summary_metrics(data)
        )

        return GenerateReportResponse(
            success=True,
            report_id=report_id,
            file_name=filename,
            format=request.format,
            generated_at=datetime.utcnow().isoformat(),
            message="Report generated and email scheduled"
        )

    return GenerateReportResponse(
        success=True,
        report_id=report_id,
        download_url=f"/reports/download/{report_id}",
        file_name=filename,
        format=request.format,
        generated_at=datetime.utcnow().isoformat(),
        message="Report generated successfully"
    )


@router.get("/download/{report_id}")
async def download_report(
    report_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Download a generated report by ID.
    """
    if report_id not in _report_cache:
        raise HTTPException(
            status_code=404,
            detail="Report not found or expired"
        )

    report = _report_cache[report_id]
    buffer = report["buffer"]

    # Reset buffer position
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type=report["content_type"],
        headers={
            "Content-Disposition": f"attachment; filename={report['filename']}"
        }
    )


# ==================== Report Schedules ====================

@router.get("/schedules")
async def list_schedules(
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """
    List all report schedules accessible to the current user.
    """
    db = get_database()
    user_id = current_user.get("user_id")
    user_role = UserRole(current_user.get("role"))

    # Build query based on role
    if user_role == UserRole.SUPER_ADMIN:
        query = {}  # Can see all
    elif user_role == UserRole.GENERAL_MANAGER:
        query = {}  # Can see all
    else:
        # Can only see their own schedules or schedules for their mine
        mine_id = current_user.get("mine_id")
        mine_ids = current_user.get("mine_ids", [])

        query = {
            "$or": [
                {"created_by": ObjectId(user_id)},
                {"mine_id": {"$in": [ObjectId(m) for m in mine_ids] if mine_ids else [ObjectId(mine_id)] if mine_id else []}}
            ]
        }

    schedules = await db.report_schedules.find(query).sort("created_at", -1).to_list(length=100)

    return {
        "schedules": [
            _format_schedule(s) for s in schedules
        ]
    }


@router.post("/schedules")
async def create_schedule(
    schedule: ReportScheduleCreate,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """
    Create a new report schedule.
    """
    db = get_database()
    user_id = current_user.get("user_id")
    user_role = current_user.get("role")

    # Verify user has access to this report type
    available_types = ROLE_REPORT_TYPES.get(user_role, [])
    if schedule.report_type not in available_types:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this report type"
        )

    # Verify mine access
    if schedule.mine_id and not check_mine_access(current_user, schedule.mine_id):
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this mine"
        )

    # Calculate next run time
    next_run = _calculate_next_run(schedule.schedule)

    # Create schedule document
    schedule_doc = {
        "name": schedule.name,
        "report_type": schedule.report_type.value,
        "role_target": user_role,
        "created_by": ObjectId(user_id),
        "mine_id": ObjectId(schedule.mine_id) if schedule.mine_id else None,
        "schedule": {
            "frequency": schedule.schedule.frequency.value,
            "time": schedule.schedule.time,
            "day_of_week": schedule.schedule.day_of_week,
            "day_of_month": schedule.schedule.day_of_month
        },
        "recipients": [r.dict() for r in schedule.recipients],
        "config": schedule.config.dict(),
        "is_active": True,
        "next_run": next_run,
        "last_run": None,
        "run_count": 0,
        "last_error": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    result = await db.report_schedules.insert_one(schedule_doc)

    return {
        "message": "Schedule created successfully",
        "schedule_id": str(result.inserted_id),
        "next_run": next_run.isoformat()
    }


@router.put("/schedules/{schedule_id}")
async def update_schedule(
    schedule_id: str,
    updates: ReportScheduleUpdate,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """
    Update an existing report schedule.
    """
    db = get_database()
    user_id = current_user.get("user_id")
    user_role = UserRole(current_user.get("role"))

    # Get existing schedule
    schedule = await db.report_schedules.find_one({"_id": ObjectId(schedule_id)})
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    # Check permissions
    if user_role not in [UserRole.SUPER_ADMIN, UserRole.GENERAL_MANAGER]:
        if str(schedule["created_by"]) != user_id:
            raise HTTPException(
                status_code=403,
                detail="You can only update your own schedules"
            )

    # Build update document
    update_doc = {"updated_at": datetime.utcnow()}

    if updates.name is not None:
        update_doc["name"] = updates.name
    if updates.recipients is not None:
        update_doc["recipients"] = [r.dict() for r in updates.recipients]
    if updates.config is not None:
        update_doc["config"] = updates.config.dict()
    if updates.is_active is not None:
        update_doc["is_active"] = updates.is_active
    if updates.schedule is not None:
        update_doc["schedule"] = {
            "frequency": updates.schedule.frequency.value,
            "time": updates.schedule.time,
            "day_of_week": updates.schedule.day_of_week,
            "day_of_month": updates.schedule.day_of_month
        }
        update_doc["next_run"] = _calculate_next_run(updates.schedule)

    await db.report_schedules.update_one(
        {"_id": ObjectId(schedule_id)},
        {"$set": update_doc}
    )

    return {"message": "Schedule updated successfully"}


@router.delete("/schedules/{schedule_id}")
async def delete_schedule(
    schedule_id: str,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """
    Delete a report schedule.
    """
    db = get_database()
    user_id = current_user.get("user_id")
    user_role = UserRole(current_user.get("role"))

    # Get existing schedule
    schedule = await db.report_schedules.find_one({"_id": ObjectId(schedule_id)})
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    # Check permissions
    if user_role not in [UserRole.SUPER_ADMIN, UserRole.GENERAL_MANAGER]:
        if str(schedule["created_by"]) != user_id:
            raise HTTPException(
                status_code=403,
                detail="You can only delete your own schedules"
            )

    await db.report_schedules.delete_one({"_id": ObjectId(schedule_id)})

    return {"message": "Schedule deleted successfully"}


@router.post("/schedules/{schedule_id}/test")
async def test_schedule(
    schedule_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """
    Run a schedule immediately for testing.
    """
    db = get_database()

    schedule = await db.report_schedules.find_one({"_id": ObjectId(schedule_id)})
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    # Queue the report generation
    background_tasks.add_task(
        _run_scheduled_report,
        db,
        schedule
    )

    return {"message": "Test run scheduled"}


# ==================== Role-Specific Data Endpoints ====================

@router.get("/shift-incharge/summary")
async def get_shift_incharge_summary(
    shift: str = Query("day", description="Shift name (day, afternoon, night)"),
    date: str = Query(None, description="Date in YYYY-MM-DD format"),
    gate_id: Optional[str] = Query(None),
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """Get shift summary data for shift incharge."""
    db = get_database()

    mine_id = current_user.get("mine_id")
    if not mine_id:
        raise HTTPException(status_code=400, detail="Mine ID required")

    template = ShiftInchargeTemplate(db)

    date_obj = datetime.strptime(date, "%Y-%m-%d") if date else datetime.utcnow()

    data = await template.aggregate_data(
        start_date=date_obj,
        end_date=date_obj + timedelta(days=1),
        mine_id=mine_id,
        filters={"shift": shift, "gate_id": gate_id}
    )

    return data


@router.get("/safety-officer/compliance")
async def get_safety_officer_compliance(
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    group_by: str = Query("day", description="Group by: day, week, month"),
    current_user: dict = Depends(get_safety_officer_or_above)
):
    """Get compliance data for safety officer."""
    db = get_database()

    mine_id = current_user.get("mine_id")
    template = SafetyOfficerTemplate(db)

    data = await template.aggregate_data(
        start_date=datetime.strptime(start_date, "%Y-%m-%d"),
        end_date=datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1),
        mine_id=mine_id,
        filters={"group_by": group_by}
    )

    return data


@router.get("/manager/operations")
async def get_manager_operations(
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    current_user: dict = Depends(get_manager_or_above)
):
    """Get operations data for manager."""
    db = get_database()

    mine_id = current_user.get("mine_id")
    template = ManagerTemplate(db)

    data = await template.aggregate_data(
        start_date=datetime.strptime(start_date, "%Y-%m-%d"),
        end_date=datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1),
        mine_id=mine_id
    )

    return data


@router.get("/worker/{worker_id}/compliance-card")
async def get_worker_compliance_card(
    worker_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get compliance card for a worker."""
    db = get_database()

    # Workers can only see their own card
    user_role = current_user.get("role")
    if user_role == "worker":
        if current_user.get("user_id") != worker_id:
            raise HTTPException(
                status_code=403,
                detail="You can only view your own compliance card"
            )

    template = WorkerTemplate(db)

    data = await template.aggregate_data(
        start_date=datetime.utcnow() - timedelta(days=30),
        end_date=datetime.utcnow(),
        filters={"worker_id": worker_id}
    )

    return data


# ==================== Emergency Incident Report ====================

@router.get("/emergency-incident")
async def generate_emergency_incident_report(
    current_user: dict = Depends(get_current_user)
):
    """
    Generate an Emergency Incident PDF Report.

    For the demo, this returns a pre-generated report for the
    Dec 12, 2024 Gas Emergency at Jharia Coal Mine.
    """
    from .templates.emergency_incident import EmergencyIncidentTemplate, get_demo_emergency_data

    # Get demo data
    data = get_demo_emergency_data()

    # Create template (db not needed for demo data)
    template = EmergencyIncidentTemplate(db=None)

    # Generate PDF
    generator = PDFGenerator()
    buffer = await generator.generate(template, data)

    # Return as downloadable PDF
    filename = "Emergency_Incident_Report_Dec_12_2024.pdf"

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@router.post("/emergency-incident")
async def generate_emergency_incident_report_post(
    incident_date: str = "2024-12-12",
    zone_name: str = "Zone A - Extraction",
    gas_type: str = "Methane",
    gas_level: float = 15200,
    workers_affected: int = 8,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate a customized Emergency Incident PDF Report.

    For the demo, parameters are optional and default to the Dec 12 incident.
    """
    from .templates.emergency_incident import EmergencyIncidentTemplate, get_demo_emergency_data

    # Get demo data (in production, this would query actual incident data)
    data = get_demo_emergency_data()

    # Create template
    template = EmergencyIncidentTemplate(db=None)

    # Generate PDF
    generator = PDFGenerator()
    buffer = await generator.generate(template, data)

    # Generate report ID for caching
    report_id = str(uuid.uuid4())
    filename = f"Emergency_Incident_Report_{incident_date.replace('-', '_')}.pdf"

    # Store in cache
    _report_cache[report_id] = {
        "buffer": buffer,
        "filename": filename,
        "content_type": "application/pdf",
        "created_at": datetime.utcnow(),
        "format": "pdf"
    }

    return GenerateReportResponse(
        success=True,
        report_id=report_id,
        download_url=f"/reports/download/{report_id}",
        file_name=filename,
        format=ReportFormat.PDF,
        generated_at=datetime.utcnow().isoformat(),
        message="Emergency Incident Report generated successfully"
    )


# ==================== Helper Functions ====================

def _format_schedule(schedule: dict) -> dict:
    """Format a schedule document for response."""
    return {
        "id": str(schedule["_id"]),
        "name": schedule.get("name"),
        "report_type": schedule.get("report_type"),
        "role_target": schedule.get("role_target"),
        "mine_id": str(schedule["mine_id"]) if schedule.get("mine_id") else None,
        "schedule": schedule.get("schedule"),
        "recipients": schedule.get("recipients", []),
        "config": schedule.get("config", {}),
        "is_active": schedule.get("is_active", True),
        "next_run": schedule.get("next_run"),
        "last_run": schedule.get("last_run"),
        "run_count": schedule.get("run_count", 0),
        "created_at": schedule.get("created_at")
    }


def _calculate_next_run(schedule_config) -> datetime:
    """Calculate the next run time for a schedule."""
    now = datetime.utcnow()
    time_parts = schedule_config.time.split(":")
    hour = int(time_parts[0])
    minute = int(time_parts[1]) if len(time_parts) > 1 else 0

    if schedule_config.frequency == ScheduleFrequency.DAILY:
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)

    elif schedule_config.frequency == ScheduleFrequency.WEEKLY:
        day_of_week = schedule_config.day_of_week or 1  # Default to Monday
        days_ahead = day_of_week - now.isoweekday()
        if days_ahead < 0:
            days_ahead += 7
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=days_ahead)
        if next_run <= now:
            next_run += timedelta(weeks=1)

    elif schedule_config.frequency == ScheduleFrequency.MONTHLY:
        day_of_month = schedule_config.day_of_month or 1
        next_run = now.replace(day=day_of_month, hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            # Move to next month
            if now.month == 12:
                next_run = next_run.replace(year=now.year + 1, month=1)
            else:
                next_run = next_run.replace(month=now.month + 1)

    else:
        next_run = now + timedelta(days=1)

    return next_run


def _generate_csv(template, data: dict) -> BytesIO:
    """Generate a simple CSV from report data."""
    from io import StringIO
    import csv

    output = StringIO()
    writer = csv.writer(output)

    # Get structure
    structure = template.get_excel_structure()

    for sheet in structure:
        if sheet.get("type") == "data_table":
            data_key = sheet.get("data_key")
            columns = sheet.get("columns", [])
            table_data = _get_nested_value(data, data_key)

            if table_data and isinstance(table_data, list):
                # Write headers
                writer.writerow([c.get("label", c.get("key")) for c in columns])

                # Write data
                for row in table_data:
                    row_values = []
                    for col in columns:
                        value = _get_nested_value(row, col.get("key"))
                        if isinstance(value, list):
                            value = ", ".join(str(v) for v in value)
                        row_values.append(value if value is not None else "")
                    writer.writerow(row_values)

                writer.writerow([])  # Empty row between tables

    # Convert to bytes
    buffer = BytesIO()
    buffer.write(output.getvalue().encode('utf-8'))
    buffer.seek(0)
    return buffer


def _get_nested_value(data: dict, key: str):
    """Get nested value from dict using dot notation."""
    if not key or not data:
        return None

    keys = key.split(".")
    value = data

    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
        else:
            return None

    return value


async def _send_report_email(
    recipients: List[EmailRecipient],
    report_name: str,
    date_range: str,
    buffer: BytesIO,
    filename: str,
    content_type: str,
    summary_metrics: list
):
    """Send report email in background."""
    email_service = get_email_service()

    await email_service.send_report(
        recipients=recipients,
        report_name=report_name,
        date_range=date_range,
        attachments=[{
            "filename": filename,
            "data": buffer,
            "content_type": content_type,
            "format": filename.split(".")[-1]
        }],
        summary_metrics=summary_metrics
    )


async def _run_scheduled_report(db, schedule: dict):
    """Run a scheduled report."""
    from .report_scheduler import run_single_schedule
    await run_single_schedule(db, schedule)
