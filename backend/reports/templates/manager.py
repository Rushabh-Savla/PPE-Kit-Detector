"""
Report templates for Manager role.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from .base import BaseReportTemplate
from ..schemas import ReportType


class ManagerTemplate(BaseReportTemplate):
    """Template for Manager operations reports."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.DAILY_OPERATIONS

    @property
    def report_name(self) -> str:
        return "Daily Operations Summary"

    @property
    def required_role(self) -> str:
        return "manager"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        from ..services.data_aggregator import DataAggregator

        aggregator = DataAggregator(self.db)
        mine_info = await self.get_mine_info(mine_id)

        operations = await aggregator.get_operations_data(
            mine_id=mine_id,
            start_date=start_date,
            end_date=end_date
        )

        return {
            "mine_name": mine_info.get("name", "All Mines"),
            "date_range": f"{self.format_date(start_date)} to {self.format_date(end_date)}",
            "generated_at": datetime.utcnow(),
            **operations
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "Daily Operations Summary",
                "include_logo": True
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "total_workers", "label": "Total Workers", "color": "blue"},
                    {"key": "active_workers", "label": "Active Today", "color": "green"},
                    {"key": "compliance_rate", "label": "Compliance", "color": "auto", "format": "percentage"}
                ]
            },
            {
                "type": "chart",
                "chart_type": "bar",
                "data_key": "shift_performance",
                "title": "Shift Performance Comparison",
                "grouped": True
            },
            {
                "type": "two_column",
                "left": {
                    "type": "table",
                    "title": "Top Performers",
                    "data_key": "worker_rankings.top_performers",
                    "columns": [
                        {"key": "name", "label": "Name"},
                        {"key": "compliance_score", "label": "Score"}
                    ],
                    "max_rows": 5
                },
                "right": {
                    "type": "table",
                    "title": "Needs Improvement",
                    "data_key": "worker_rankings.needs_improvement",
                    "columns": [
                        {"key": "name", "label": "Name"},
                        {"key": "compliance_score", "label": "Score"}
                    ],
                    "max_rows": 5
                }
            },
            {
                "type": "chart",
                "chart_type": "pie",
                "data_key": "violations_by_type",
                "title": "Violations Distribution"
            },
            {
                "type": "table",
                "title": "Recent Escalations",
                "data_key": "escalations",
                "columns": [
                    {"key": "created_at", "label": "Date", "format": "datetime"},
                    {"key": "type", "label": "Type"},
                    {"key": "severity", "label": "Severity", "format": "severity_badge"},
                    {"key": "message", "label": "Description"},
                    {"key": "status", "label": "Status"}
                ],
                "max_rows": 10
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Summary",
                "type": "summary",
                "sections": [
                    {
                        "title": "Operations Overview",
                        "fields": [
                            ("Total Workers", "total_workers"),
                            ("Active Workers Today", "active_workers"),
                            ("Compliance Rate", "compliance_rate", "percentage")
                        ]
                    }
                ]
            },
            {
                "name": "Shift Performance",
                "type": "shift_comparison",
                "data_key": "shift_performance",
                "columns": [
                    {"key": "shift", "label": "Shift", "width": 15},
                    {"key": "total_entries", "label": "Total Entries", "width": 15},
                    {"key": "compliant_entries", "label": "Compliant", "width": 15},
                    {"key": "compliance_rate", "label": "Compliance %", "width": 15}
                ]
            },
            {
                "name": "Top Performers",
                "type": "data_table",
                "data_key": "worker_rankings.top_performers",
                "columns": [
                    {"key": "employee_id", "label": "Employee ID", "width": 15},
                    {"key": "name", "label": "Name", "width": 25},
                    {"key": "compliance_score", "label": "Score", "width": 12}
                ]
            },
            {
                "name": "Needs Improvement",
                "type": "data_table",
                "data_key": "worker_rankings.needs_improvement",
                "columns": [
                    {"key": "employee_id", "label": "Employee ID", "width": 15},
                    {"key": "name", "label": "Name", "width": 25},
                    {"key": "compliance_score", "label": "Score", "width": 12}
                ]
            },
            {
                "name": "Escalations",
                "type": "data_table",
                "data_key": "escalations",
                "columns": [
                    {"key": "created_at", "label": "Date", "width": 20},
                    {"key": "type", "label": "Type", "width": 20},
                    {"key": "severity", "label": "Severity", "width": 12},
                    {"key": "message", "label": "Description", "width": 40},
                    {"key": "status", "label": "Status", "width": 12}
                ]
            }
        ]

    def get_summary_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        compliance = data.get("compliance_rate", 0)
        return [
            {
                "label": "Active Workers",
                "value": f"{data.get('active_workers', 0)}/{data.get('total_workers', 0)}",
                "color": "blue"
            },
            {
                "label": "Compliance Rate",
                "value": f"{compliance}%",
                "color": self.get_compliance_color(compliance)
            },
            {
                "label": "Escalations",
                "value": len(data.get("escalations", [])),
                "color": "orange" if len(data.get("escalations", [])) > 0 else "green"
            }
        ]


class MonthlySummaryTemplate(BaseReportTemplate):
    """Template for monthly summary report."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.MONTHLY_SUMMARY

    @property
    def report_name(self) -> str:
        return "Monthly Summary Report"

    @property
    def required_role(self) -> str:
        return "manager"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        from ..services.data_aggregator import DataAggregator

        filters = filters or {}
        month = filters.get("month", start_date.month)
        year = filters.get("year", start_date.year)

        aggregator = DataAggregator(self.db)
        mine_info = await self.get_mine_info(mine_id)

        monthly = await aggregator.get_monthly_summary(mine_id, month, year)

        return {
            "mine_name": mine_info.get("name", "All Mines"),
            "date_range": f"{datetime(year, month, 1).strftime('%B %Y')}",
            "generated_at": datetime.utcnow(),
            **monthly
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "Monthly Summary Report",
                "include_logo": True
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "total_workers", "label": "Total Workers", "color": "blue"},
                    {"key": "active_workers", "label": "Active Workers", "color": "green"},
                    {"key": "compliance_rate", "label": "Compliance", "color": "auto", "format": "percentage"}
                ]
            },
            {
                "type": "chart",
                "chart_type": "line",
                "data_key": "compliance_trend",
                "title": "Weekly Compliance Trend",
                "x_key": "date",
                "y_key": "compliance_rate"
            },
            {
                "type": "comparison_box",
                "current_key": "violation_trends.current_period.total",
                "previous_key": "violation_trends.previous_period.total",
                "change_key": "violation_trends.change_percentage",
                "title": "Violations vs Previous Month"
            },
            {
                "type": "chart",
                "chart_type": "bar",
                "data_key": "shift_performance",
                "title": "Shift Performance"
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Summary",
                "type": "summary",
                "sections": [
                    {
                        "title": "Monthly Overview",
                        "fields": [
                            ("Month", "date_range"),
                            ("Total Workers", "total_workers"),
                            ("Active Workers", "active_workers"),
                            ("Compliance Rate", "compliance_rate", "percentage")
                        ]
                    },
                    {
                        "title": "Violation Comparison",
                        "fields": [
                            ("Current Month", "violation_trends.current_period.total"),
                            ("Previous Month", "violation_trends.previous_period.total"),
                            ("Change", "violation_trends.change_percentage", "percentage")
                        ]
                    }
                ]
            },
            {
                "name": "Weekly Trend",
                "type": "data_table",
                "data_key": "compliance_trend",
                "columns": [
                    {"key": "date", "label": "Week", "width": 15},
                    {"key": "total_entries", "label": "Total Entries", "width": 15},
                    {"key": "compliant_entries", "label": "Compliant", "width": 15},
                    {"key": "compliance_rate", "label": "Compliance %", "width": 15}
                ],
                "include_chart": True
            }
        ]


class ShiftPerformanceTemplate(BaseReportTemplate):
    """Template for shift performance comparison."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.SHIFT_PERFORMANCE

    @property
    def report_name(self) -> str:
        return "Shift Performance Report"

    @property
    def required_role(self) -> str:
        return "manager"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        from ..services.data_aggregator import DataAggregator

        aggregator = DataAggregator(self.db)
        mine_info = await self.get_mine_info(mine_id)

        operations = await aggregator.get_operations_data(mine_id, start_date, end_date)

        # Transform shift performance for comparison
        shift_data = operations.get("shift_performance", {})
        shifts_list = [
            {"shift": shift, **data}
            for shift, data in shift_data.items()
        ]

        return {
            "mine_name": mine_info.get("name", "All Mines"),
            "date_range": f"{self.format_date(start_date)} to {self.format_date(end_date)}",
            "generated_at": datetime.utcnow(),
            "shift_performance": shifts_list,
            "best_shift": max(shifts_list, key=lambda x: x.get("compliance_rate", 0)) if shifts_list else None,
            "worst_shift": min(shifts_list, key=lambda x: x.get("compliance_rate", 0)) if shifts_list else None
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "Shift Performance Report",
                "include_logo": True
            },
            {
                "type": "chart",
                "chart_type": "bar",
                "data_key": "shift_performance",
                "title": "Compliance by Shift",
                "x_key": "shift",
                "y_key": "compliance_rate"
            },
            {
                "type": "table",
                "title": "Shift Details",
                "data_key": "shift_performance",
                "columns": [
                    {"key": "shift", "label": "Shift"},
                    {"key": "total_entries", "label": "Total Entries"},
                    {"key": "compliant_entries", "label": "Compliant"},
                    {"key": "compliance_rate", "label": "Compliance %", "format": "percentage"}
                ]
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Shift Performance",
                "type": "data_table",
                "data_key": "shift_performance",
                "columns": [
                    {"key": "shift", "label": "Shift", "width": 15},
                    {"key": "total_entries", "label": "Total Entries", "width": 15},
                    {"key": "compliant_entries", "label": "Compliant Entries", "width": 18},
                    {"key": "compliance_rate", "label": "Compliance %", "width": 15}
                ],
                "include_chart": True
            }
        ]
