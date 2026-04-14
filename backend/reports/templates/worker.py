"""
Report templates for Worker role.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from .base import BaseReportTemplate
from ..schemas import ReportType


class WorkerTemplate(BaseReportTemplate):
    """Template for Worker personal compliance card."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.COMPLIANCE_CARD

    @property
    def report_name(self) -> str:
        return "Personal Compliance Card"

    @property
    def required_role(self) -> str:
        return "worker"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        from ..services.data_aggregator import DataAggregator

        filters = filters or {}
        worker_id = filters.get("worker_id")

        if not worker_id:
            return {"error": "Worker ID is required"}

        aggregator = DataAggregator(self.db)

        card_data = await aggregator.get_worker_compliance_card(worker_id)

        return {
            "mine_name": "Personal Report",
            "date_range": f"As of {self.format_date(datetime.utcnow())}",
            "generated_at": datetime.utcnow(),
            **card_data
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "Personal Compliance Card",
                "include_logo": True
            },
            {
                "type": "worker_profile",
                "data_key": "worker_info",
                "fields": ["name", "employee_id", "department", "assigned_shift"]
            },
            {
                "type": "score_gauge",
                "title": "Compliance Score",
                "value_key": "compliance_score",
                "max_value": 100
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "compliance_score", "label": "Compliance Score", "color": "auto", "format": "score"},
                    {"key": "streak_days", "label": "Streak Days", "color": "green"},
                    {"key": "total_violations", "label": "Total Violations", "color": "red"}
                ]
            },
            {
                "type": "badges_row",
                "title": "Badges Earned",
                "data_key": "badges"
            },
            {
                "type": "attendance_summary",
                "title": "Attendance (Last 30 Days)",
                "data_key": "attendance_summary",
                "fields": ["days_worked_last_30", "total_entries_last_30", "attendance_rate"]
            },
            {
                "type": "table",
                "title": "Recent Violations",
                "data_key": "recent_violations",
                "columns": [
                    {"key": "timestamp", "label": "Date/Time", "format": "datetime"},
                    {"key": "violations", "label": "Violations", "format": "list"}
                ],
                "max_rows": 10,
                "empty_message": "Great job! No recent violations."
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Compliance Card",
                "type": "summary",
                "sections": [
                    {
                        "title": "Worker Information",
                        "data_key": "worker_info",
                        "fields": [
                            ("Name", "name"),
                            ("Employee ID", "employee_id"),
                            ("Department", "department"),
                            ("Assigned Shift", "assigned_shift")
                        ]
                    },
                    {
                        "title": "Compliance Metrics",
                        "fields": [
                            ("Compliance Score", "compliance_score"),
                            ("Streak Days", "streak_days"),
                            ("Total Violations", "total_violations")
                        ]
                    },
                    {
                        "title": "Attendance (Last 30 Days)",
                        "data_key": "attendance_summary",
                        "fields": [
                            ("Days Worked", "days_worked_last_30"),
                            ("Total Entries", "total_entries_last_30"),
                            ("Attendance Rate", "attendance_rate", "percentage")
                        ]
                    }
                ]
            },
            {
                "name": "Recent Violations",
                "type": "data_table",
                "data_key": "recent_violations",
                "columns": [
                    {"key": "timestamp", "label": "Date/Time", "width": 20},
                    {"key": "violations", "label": "Violations", "width": 40}
                ]
            },
            {
                "name": "Badges",
                "type": "list",
                "data_key": "badges",
                "title": "Earned Badges"
            }
        ]

    def get_summary_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        score = data.get("compliance_score", 0)
        return [
            {
                "label": "Compliance Score",
                "value": f"{score}",
                "color": self.get_compliance_color(score)
            },
            {
                "label": "Streak Days",
                "value": data.get("streak_days", 0),
                "color": "green"
            },
            {
                "label": "Badges Earned",
                "value": len(data.get("badges", [])),
                "color": "blue"
            }
        ]


class WorkerMonthlyTemplate(BaseReportTemplate):
    """Template for worker monthly summary."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.WORKER_MONTHLY

    @property
    def report_name(self) -> str:
        return "Worker Monthly Summary"

    @property
    def required_role(self) -> str:
        return "worker"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        from ..services.data_aggregator import DataAggregator
        from bson import ObjectId

        filters = filters or {}
        worker_id = filters.get("worker_id")

        if not worker_id:
            return {"error": "Worker ID is required"}

        aggregator = DataAggregator(self.db)

        # Get basic compliance card data
        card_data = await aggregator.get_worker_compliance_card(worker_id)

        # Get monthly entries
        entries = await self.db.gate_entries.find({
            "worker_id": ObjectId(worker_id),
            "timestamp": {"$gte": start_date, "$lt": end_date}
        }).to_list(length=None)

        total_entries = len(entries)
        compliant_entries = len([e for e in entries if e.get("ppe_compliant", False)])
        monthly_compliance = (compliant_entries / total_entries * 100) if total_entries > 0 else 100

        # Count violations in period
        violations_in_period = sum(len(e.get("violations", [])) for e in entries)

        return {
            "mine_name": "Monthly Summary",
            "date_range": f"{start_date.strftime('%B %Y')}",
            "generated_at": datetime.utcnow(),
            "worker_info": card_data.get("worker_info", {}),
            "monthly_metrics": {
                "total_entries": total_entries,
                "compliant_entries": compliant_entries,
                "monthly_compliance": round(monthly_compliance, 1),
                "violations_count": violations_in_period
            },
            "current_score": card_data.get("compliance_score", 0),
            "badges": card_data.get("badges", []),
            "entries": [
                {
                    "timestamp": e.get("timestamp"),
                    "entry_type": e.get("entry_type"),
                    "ppe_compliant": e.get("ppe_compliant", False),
                    "violations": e.get("violations", [])
                }
                for e in entries
            ]
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "Monthly Performance Summary",
                "include_logo": True
            },
            {
                "type": "worker_profile",
                "data_key": "worker_info",
                "compact": True
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "monthly_metrics.total_entries", "label": "Total Entries", "color": "blue"},
                    {"key": "monthly_metrics.monthly_compliance", "label": "Monthly Compliance", "color": "auto", "format": "percentage"},
                    {"key": "monthly_metrics.violations_count", "label": "Violations", "color": "red"},
                    {"key": "current_score", "label": "Current Score", "color": "auto"}
                ]
            },
            {
                "type": "badges_row",
                "title": "Your Badges",
                "data_key": "badges"
            },
            {
                "type": "table",
                "title": "Entry Log",
                "data_key": "entries",
                "columns": [
                    {"key": "timestamp", "label": "Date/Time", "format": "datetime"},
                    {"key": "entry_type", "label": "Type"},
                    {"key": "ppe_compliant", "label": "PPE OK", "format": "boolean"},
                    {"key": "violations", "label": "Issues", "format": "list"}
                ],
                "max_rows": 30
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Summary",
                "type": "summary",
                "sections": [
                    {
                        "title": "Monthly Performance",
                        "fields": [
                            ("Total Entries", "monthly_metrics.total_entries"),
                            ("Compliant Entries", "monthly_metrics.compliant_entries"),
                            ("Monthly Compliance", "monthly_metrics.monthly_compliance", "percentage"),
                            ("Violations", "monthly_metrics.violations_count"),
                            ("Current Score", "current_score")
                        ]
                    }
                ]
            },
            {
                "name": "Entry Log",
                "type": "data_table",
                "data_key": "entries",
                "columns": [
                    {"key": "timestamp", "label": "Date/Time", "width": 20},
                    {"key": "entry_type", "label": "Type", "width": 10},
                    {"key": "ppe_compliant", "label": "PPE Compliant", "width": 15},
                    {"key": "violations", "label": "Violations", "width": 30}
                ]
            }
        ]
