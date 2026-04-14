"""
Report templates for Shift Incharge role.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from .base import BaseReportTemplate
from ..schemas import ReportType


class ShiftInchargeTemplate(BaseReportTemplate):
    """Template for Shift Incharge reports."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.SHIFT_SUMMARY

    @property
    def report_name(self) -> str:
        return "Shift Summary Report"

    @property
    def required_role(self) -> str:
        return "shift_incharge"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Aggregate shift data."""
        from ..services.data_aggregator import DataAggregator

        filters = filters or {}
        aggregator = DataAggregator(self.db)

        shift = filters.get("shift", "day")
        gate_id = filters.get("gate_id")

        # Get mine info
        mine_info = await self.get_mine_info(mine_id)

        # Get shift summary
        summary = await aggregator.get_shift_summary(
            mine_id=mine_id,
            shift=shift,
            date=start_date,
            gate_id=gate_id
        )

        return {
            "mine_name": mine_info.get("name", "All Mines"),
            "date_range": f"{self.format_date(start_date)} - {self.format_date(end_date)}",
            "generated_at": datetime.utcnow(),
            **summary
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        """Define PDF layout for shift summary."""
        return [
            {
                "type": "header",
                "title": "Shift Summary Report",
                "include_logo": True
            },
            {
                "type": "info_box",
                "fields": ["shift_info.shift", "shift_info.date", "shift_info.start_time", "shift_info.end_time"]
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "workers_entered", "label": "Workers Entered", "color": "green"},
                    {"key": "workers_exited", "label": "Workers Exited", "color": "blue"},
                    {"key": "currently_inside", "label": "Currently Inside", "color": "orange"},
                    {"key": "compliance_rate", "label": "Compliance Rate", "color": "auto", "format": "percentage"}
                ]
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "violations_count", "label": "Violations", "color": "red"},
                    {"key": "alerts_resolved", "label": "Alerts Resolved", "color": "green"},
                    {"key": "alerts_pending", "label": "Alerts Pending", "color": "orange"}
                ]
            },
            {
                "type": "chart",
                "chart_type": "bar",
                "data_key": "hourly_breakdown",
                "title": "Hourly Entry Distribution",
                "x_key": "hour",
                "y_key": "entries"
            },
            {
                "type": "table",
                "title": "Recent Entry/Exit Log",
                "data_key": "entry_exit_logs",
                "columns": [
                    {"key": "timestamp", "label": "Time", "format": "datetime"},
                    {"key": "worker_name", "label": "Worker"},
                    {"key": "employee_id", "label": "ID"},
                    {"key": "entry_type", "label": "Type"},
                    {"key": "ppe_compliant", "label": "PPE Status", "format": "boolean"},
                    {"key": "violations", "label": "Violations", "format": "list"}
                ],
                "max_rows": 50
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        """Define Excel sheets for shift summary."""
        return [
            {
                "name": "Summary",
                "type": "summary",
                "sections": [
                    {
                        "title": "Shift Information",
                        "data_key": "shift_info",
                        "layout": "key_value"
                    },
                    {
                        "title": "Metrics",
                        "fields": [
                            ("Workers Entered", "workers_entered"),
                            ("Workers Exited", "workers_exited"),
                            ("Currently Inside", "currently_inside"),
                            ("Compliance Rate", "compliance_rate", "percentage"),
                            ("Violations", "violations_count"),
                            ("Alerts Resolved", "alerts_resolved"),
                            ("Alerts Pending", "alerts_pending")
                        ]
                    }
                ]
            },
            {
                "name": "Entry Exit Log",
                "type": "data_table",
                "data_key": "entry_exit_logs",
                "columns": [
                    {"key": "timestamp", "label": "Timestamp", "width": 20},
                    {"key": "worker_name", "label": "Worker Name", "width": 25},
                    {"key": "employee_id", "label": "Employee ID", "width": 15},
                    {"key": "entry_type", "label": "Entry Type", "width": 12},
                    {"key": "ppe_compliant", "label": "PPE Compliant", "width": 15},
                    {"key": "violations", "label": "Violations", "width": 30}
                ]
            },
            {
                "name": "Hourly Breakdown",
                "type": "data_table",
                "data_key": "hourly_breakdown",
                "columns": [
                    {"key": "hour", "label": "Hour", "width": 10},
                    {"key": "entries", "label": "Entries", "width": 12},
                    {"key": "violations", "label": "Violations", "width": 12},
                    {"key": "compliance_rate", "label": "Compliance %", "width": 15}
                ]
            }
        ]

    def get_summary_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract summary metrics for report header."""
        compliance = data.get("compliance_rate", 0)
        return [
            {
                "label": "Workers Entered",
                "value": data.get("workers_entered", 0),
                "color": "green"
            },
            {
                "label": "Compliance Rate",
                "value": f"{compliance}%",
                "color": self.get_compliance_color(compliance)
            },
            {
                "label": "Violations",
                "value": data.get("violations_count", 0),
                "color": "red" if data.get("violations_count", 0) > 0 else "green"
            },
            {
                "label": "Pending Alerts",
                "value": data.get("alerts_pending", 0),
                "color": "orange" if data.get("alerts_pending", 0) > 0 else "green"
            }
        ]


class ShiftHandoverTemplate(BaseReportTemplate):
    """Template for Shift Handover reports."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.SHIFT_HANDOVER

    @property
    def report_name(self) -> str:
        return "Shift Handover Report"

    @property
    def required_role(self) -> str:
        return "shift_incharge"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Aggregate handover data."""
        from ..services.data_aggregator import DataAggregator

        filters = filters or {}
        aggregator = DataAggregator(self.db)

        shift = filters.get("shift", "day")
        mine_info = await self.get_mine_info(mine_id)

        # Get shift summary
        summary = await aggregator.get_shift_summary(
            mine_id=mine_id,
            shift=shift,
            date=start_date
        )

        # Get unresolved alerts
        from bson import ObjectId
        unresolved_alerts = await self.db.alerts.find({
            "mine_id": ObjectId(mine_id) if mine_id else {"$exists": True},
            "status": "active"
        }).sort("created_at", -1).to_list(length=20)

        return {
            "mine_name": mine_info.get("name", "All Mines"),
            "date_range": f"Shift: {shift.title()} - {self.format_date(start_date)}",
            "generated_at": datetime.utcnow(),
            "shift_info": summary.get("shift_info"),
            "workers_inside": summary.get("currently_inside", 0),
            "pending_alerts": [
                {
                    "id": str(a["_id"]),
                    "type": a.get("alert_type"),
                    "severity": a.get("severity"),
                    "message": a.get("message"),
                    "created_at": a.get("created_at")
                }
                for a in unresolved_alerts
            ],
            "summary_metrics": {
                "total_entries": summary.get("workers_entered", 0),
                "total_exits": summary.get("workers_exited", 0),
                "compliance_rate": summary.get("compliance_rate", 0),
                "violations": summary.get("violations_count", 0)
            }
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "Shift Handover Report",
                "include_logo": True
            },
            {
                "type": "info_box",
                "fields": ["shift_info.shift", "shift_info.date"]
            },
            {
                "type": "alert_box",
                "title": "Workers Currently Inside",
                "data_key": "workers_inside",
                "severity": "info"
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "summary_metrics.total_entries", "label": "Total Entries", "color": "green"},
                    {"key": "summary_metrics.total_exits", "label": "Total Exits", "color": "blue"},
                    {"key": "summary_metrics.compliance_rate", "label": "Compliance", "color": "auto", "format": "percentage"},
                    {"key": "summary_metrics.violations", "label": "Violations", "color": "red"}
                ]
            },
            {
                "type": "table",
                "title": "Pending Alerts - Requires Attention",
                "data_key": "pending_alerts",
                "columns": [
                    {"key": "severity", "label": "Severity"},
                    {"key": "type", "label": "Type"},
                    {"key": "message", "label": "Message"},
                    {"key": "created_at", "label": "Time", "format": "datetime"}
                ],
                "highlight_severity": True
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Handover Summary",
                "type": "summary",
                "sections": [
                    {
                        "title": "Shift Information",
                        "data_key": "shift_info",
                        "layout": "key_value"
                    },
                    {
                        "title": "Status",
                        "fields": [
                            ("Workers Inside", "workers_inside"),
                            ("Total Entries", "summary_metrics.total_entries"),
                            ("Total Exits", "summary_metrics.total_exits"),
                            ("Compliance Rate", "summary_metrics.compliance_rate", "percentage"),
                            ("Violations", "summary_metrics.violations")
                        ]
                    }
                ]
            },
            {
                "name": "Pending Alerts",
                "type": "data_table",
                "data_key": "pending_alerts",
                "columns": [
                    {"key": "severity", "label": "Severity", "width": 12},
                    {"key": "type", "label": "Alert Type", "width": 20},
                    {"key": "message", "label": "Message", "width": 50},
                    {"key": "created_at", "label": "Created At", "width": 20}
                ]
            }
        ]
