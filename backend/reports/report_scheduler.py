"""
Report Scheduler Integration.

Integrates with the existing APScheduler to run scheduled reports.
"""

from datetime import datetime, timedelta
from typing import Optional
from bson import ObjectId

from .schemas import ReportType, ReportFormat, DateRangeType, EmailRecipient
from .services.pdf_generator import PDFGenerator
from .services.excel_generator import ExcelGenerator
from .services.email_service import get_email_service


# Template mapping
def get_template_for_type(report_type: str, db):
    """Get the appropriate template class for a report type."""
    from .templates import (
        ShiftInchargeTemplate, SafetyOfficerTemplate, ManagerTemplate,
        AreaSafetyOfficerTemplate, GeneralManagerTemplate, WorkerTemplate
    )
    from .templates.shift_incharge import ShiftHandoverTemplate
    from .templates.safety_officer import ViolationTrendsTemplate, HighRiskWorkersTemplate
    from .templates.manager import MonthlySummaryTemplate, ShiftPerformanceTemplate
    from .templates.area_safety_officer import RiskHeatmapTemplate, CriticalIncidentsTemplate
    from .templates.general_manager import KPIDashboardTemplate, FinancialImpactTemplate
    from .templates.worker import WorkerMonthlyTemplate

    template_map = {
        "shift_summary": ShiftInchargeTemplate,
        "shift_handover": ShiftHandoverTemplate,
        "weekly_compliance": SafetyOfficerTemplate,
        "violation_trends": ViolationTrendsTemplate,
        "high_risk_workers": HighRiskWorkersTemplate,
        "daily_operations": ManagerTemplate,
        "monthly_summary": MonthlySummaryTemplate,
        "shift_performance": ShiftPerformanceTemplate,
        "mine_comparison": AreaSafetyOfficerTemplate,
        "risk_heatmap": RiskHeatmapTemplate,
        "critical_incidents": CriticalIncidentsTemplate,
        "executive_summary": GeneralManagerTemplate,
        "kpi_dashboard": KPIDashboardTemplate,
        "financial_impact": FinancialImpactTemplate,
        "compliance_card": WorkerTemplate,
        "worker_monthly": WorkerMonthlyTemplate,
    }

    template_class = template_map.get(report_type)
    if template_class:
        return template_class(db)
    return None


def calculate_date_range(range_type: str) -> tuple:
    """Calculate start and end dates based on range type."""
    now = datetime.utcnow()

    if range_type == "previous_shift":
        # Assuming 8-hour shifts
        end_date = now.replace(minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(hours=8)

    elif range_type == "previous_day":
        end_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=1)

    elif range_type == "previous_week":
        end_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(weeks=1)

    elif range_type == "previous_month":
        end_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Go back to first day of previous month
        if now.month == 1:
            start_date = datetime(now.year - 1, 12, 1)
        else:
            start_date = datetime(now.year, now.month - 1, 1)

    else:
        # Default to previous day
        end_date = now
        start_date = now - timedelta(days=1)

    return start_date, end_date


def calculate_next_run(schedule: dict) -> datetime:
    """Calculate the next run time for a schedule."""
    now = datetime.utcnow()
    freq = schedule.get("frequency", "daily")
    time_str = schedule.get("time", "06:00")

    time_parts = time_str.split(":")
    hour = int(time_parts[0])
    minute = int(time_parts[1]) if len(time_parts) > 1 else 0

    if freq == "daily":
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)

    elif freq == "weekly":
        day_of_week = schedule.get("day_of_week", 1)  # 1 = Monday
        days_ahead = day_of_week - now.isoweekday()
        if days_ahead < 0:
            days_ahead += 7
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        next_run += timedelta(days=days_ahead)
        if next_run <= now:
            next_run += timedelta(weeks=1)

    elif freq == "monthly":
        day_of_month = schedule.get("day_of_month", 1)
        try:
            next_run = now.replace(day=day_of_month, hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            # Handle months with fewer days
            next_run = now.replace(day=28, hour=hour, minute=minute, second=0, microsecond=0)

        if next_run <= now:
            if now.month == 12:
                next_run = next_run.replace(year=now.year + 1, month=1)
            else:
                next_run = next_run.replace(month=now.month + 1)

    else:
        next_run = now + timedelta(days=1)

    return next_run


async def run_single_schedule(db, schedule: dict):
    """
    Run a single scheduled report.

    Args:
        db: MongoDB database instance
        schedule: Schedule document
    """
    schedule_id = schedule["_id"]
    report_type = schedule.get("report_type")
    config = schedule.get("config", {})
    recipients_data = schedule.get("recipients", [])

    print(f"[{datetime.utcnow()}] Running scheduled report: {schedule.get('name')}")

    try:
        # Get template
        template = get_template_for_type(report_type, db)
        if not template:
            raise ValueError(f"No template for report type: {report_type}")

        # Calculate date range
        date_range_type = config.get("date_range", "previous_day")
        start_date, end_date = calculate_date_range(date_range_type)

        # Get mine_id
        mine_id = str(schedule.get("mine_id")) if schedule.get("mine_id") else None

        # Aggregate data
        data = await template.aggregate_data(
            start_date=start_date,
            end_date=end_date,
            mine_id=mine_id,
            filters=config.get("filters", {})
        )

        # Generate reports in requested formats
        formats = config.get("format", ["pdf"])
        attachments = []

        for fmt in formats:
            if fmt == "pdf":
                generator = PDFGenerator()
                buffer = await generator.generate(template, data, config.get("include_charts", True))
                attachments.append({
                    "filename": f"{report_type}_{start_date.strftime('%Y%m%d')}.pdf",
                    "data": buffer,
                    "content_type": "application/pdf",
                    "format": "pdf"
                })

            elif fmt == "excel":
                generator = ExcelGenerator()
                buffer = await generator.generate(template, data, config.get("include_charts", True))
                attachments.append({
                    "filename": f"{report_type}_{start_date.strftime('%Y%m%d')}.xlsx",
                    "data": buffer,
                    "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "format": "xlsx"
                })

        # Send email
        if recipients_data and attachments:
            recipients = [
                EmailRecipient(
                    email=r.get("email"),
                    name=r.get("name", ""),
                    type=r.get("type", "to")
                )
                for r in recipients_data
            ]

            email_service = get_email_service()
            await email_service.send_report(
                recipients=recipients,
                report_name=template.report_name,
                date_range=data.get("date_range", f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"),
                attachments=attachments,
                summary_metrics=template.get_summary_metrics(data)
            )

        # Update schedule
        next_run = calculate_next_run(schedule.get("schedule", {}))

        await db.report_schedules.update_one(
            {"_id": schedule_id},
            {
                "$set": {
                    "last_run": datetime.utcnow(),
                    "next_run": next_run,
                    "last_error": None
                },
                "$inc": {"run_count": 1}
            }
        )

        # Log to history
        await db.report_history.insert_one({
            "schedule_id": schedule_id,
            "report_type": report_type,
            "generated_by": "scheduler",
            "generated_at": datetime.utcnow(),
            "mine_id": schedule.get("mine_id"),
            "parameters": {
                "start_date": start_date,
                "end_date": end_date,
                "filters": config.get("filters", {})
            },
            "delivery": {
                "method": "email",
                "recipients": [r.get("email") for r in recipients_data],
                "status": "sent"
            },
            "files": [
                {
                    "format": a.get("format"),
                    "filename": a.get("filename")
                }
                for a in attachments
            ]
        })

        print(f"[{datetime.utcnow()}] Scheduled report completed: {schedule.get('name')}")

    except Exception as e:
        print(f"[{datetime.utcnow()}] Error running scheduled report: {e}")

        # Update schedule with error
        await db.report_schedules.update_one(
            {"_id": schedule_id},
            {
                "$set": {
                    "last_error": str(e),
                    "next_run": calculate_next_run(schedule.get("schedule", {}))
                }
            }
        )


def register_report_scheduler(scheduler, db):
    """
    Register the report scheduler job with APScheduler.

    Args:
        scheduler: APScheduler instance
        db: MongoDB database instance
    """
    async def check_and_run_schedules():
        """Check for due report schedules and execute them."""
        now = datetime.utcnow()

        print(f"[{now}] Checking for due report schedules...")

        try:
            # Find schedules that are due
            due_schedules = await db.report_schedules.find({
                "is_active": True,
                "next_run": {"$lte": now}
            }).to_list(length=50)

            if not due_schedules:
                print(f"[{now}] No due schedules found")
                return

            print(f"[{now}] Found {len(due_schedules)} due schedules")

            for schedule in due_schedules:
                try:
                    await run_single_schedule(db, schedule)
                except Exception as e:
                    print(f"[{now}] Error running schedule {schedule.get('name')}: {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"[{now}] Error checking report schedules: {e}")
            import traceback
            traceback.print_exc()

    # Remove existing job if it exists (for restarts)
    try:
        scheduler.remove_job('check_report_schedules')
    except:
        pass

    # Add job using add_job instead of decorator (works better after scheduler started)
    scheduler.add_job(
        check_and_run_schedules,
        'interval',
        minutes=5,
        id='check_report_schedules',
        replace_existing=True
    )

    print("Report scheduler registered successfully")
    print("  - Checking for due schedules every 5 minutes")
    print("  - Next check in 5 minutes")
