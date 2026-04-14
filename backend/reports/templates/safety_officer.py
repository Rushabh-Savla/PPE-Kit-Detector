"""
Report templates for Safety Officer role.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from .base import BaseReportTemplate
from ..schemas import ReportType


class SafetyOfficerTemplate(BaseReportTemplate):
    """Template for Safety Officer compliance reports."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.WEEKLY_COMPLIANCE

    @property
    def report_name(self) -> str:
        return "Weekly Compliance Report"

    @property
    def required_role(self) -> str:
        return "safety_officer"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Aggregate compliance data."""
        from ..services.data_aggregator import DataAggregator

        filters = filters or {}
        aggregator = DataAggregator(self.db)

        group_by = filters.get("group_by", "day")
        mine_info = await self.get_mine_info(mine_id)

        compliance_data = await aggregator.get_compliance_data(
            mine_id=mine_id,
            start_date=start_date,
            end_date=end_date,
            group_by=group_by
        )

        return {
            "mine_name": mine_info.get("name", "All Mines"),
            "date_range": f"{self.format_date(start_date)} to {self.format_date(end_date)}",
            "generated_at": datetime.utcnow(),
            **compliance_data
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "Weekly Compliance Report",
                "include_logo": True
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "overall_compliance", "label": "Overall Compliance", "color": "auto", "format": "percentage"}
                ]
            },
            {
                "type": "chart",
                "chart_type": "line",
                "data_key": "compliance_trend",
                "title": "Compliance Trend",
                "x_key": "date",
                "y_key": "compliance_rate"
            },
            {
                "type": "chart",
                "chart_type": "pie",
                "data_key": "violations_by_type",
                "title": "Violations by Type"
            },
            {
                "type": "table",
                "title": "Zone Risk Analysis",
                "data_key": "zone_analysis",
                "columns": [
                    {"key": "zone_name", "label": "Zone"},
                    {"key": "total_entries", "label": "Total Entries"},
                    {"key": "violations", "label": "Violations"},
                    {"key": "violation_rate", "label": "Violation Rate", "format": "percentage"},
                    {"key": "risk_level", "label": "Risk Level", "format": "risk_badge"}
                ]
            },
            {
                "type": "table",
                "title": "High Risk Workers",
                "data_key": "high_risk_workers",
                "columns": [
                    {"key": "employee_id", "label": "Employee ID"},
                    {"key": "name", "label": "Name"},
                    {"key": "compliance_score", "label": "Score", "format": "score"},
                    {"key": "violations_last_30_days", "label": "Recent Violations"}
                ],
                "max_rows": 15
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Summary",
                "type": "summary",
                "sections": [
                    {
                        "title": "Overall Metrics",
                        "fields": [
                            ("Overall Compliance", "overall_compliance", "percentage")
                        ]
                    }
                ]
            },
            {
                "name": "Compliance Trend",
                "type": "data_table",
                "data_key": "compliance_trend",
                "columns": [
                    {"key": "date", "label": "Date", "width": 15},
                    {"key": "total_entries", "label": "Total Entries", "width": 15},
                    {"key": "compliant_entries", "label": "Compliant", "width": 15},
                    {"key": "compliance_rate", "label": "Compliance %", "width": 15}
                ],
                "include_chart": True
            },
            {
                "name": "Violations by Type",
                "type": "dict_table",
                "data_key": "violations_by_type",
                "columns": [
                    {"key": "key", "label": "Violation Type", "width": 25},
                    {"key": "value", "label": "Count", "width": 15}
                ]
            },
            {
                "name": "Zone Analysis",
                "type": "data_table",
                "data_key": "zone_analysis",
                "columns": [
                    {"key": "zone_name", "label": "Zone", "width": 20},
                    {"key": "total_entries", "label": "Total Entries", "width": 15},
                    {"key": "violations", "label": "Violations", "width": 12},
                    {"key": "violation_rate", "label": "Violation %", "width": 15},
                    {"key": "risk_level", "label": "Risk Level", "width": 12}
                ]
            },
            {
                "name": "High Risk Workers",
                "type": "data_table",
                "data_key": "high_risk_workers",
                "columns": [
                    {"key": "employee_id", "label": "Employee ID", "width": 15},
                    {"key": "name", "label": "Name", "width": 25},
                    {"key": "compliance_score", "label": "Score", "width": 12},
                    {"key": "total_violations", "label": "Total Violations", "width": 15},
                    {"key": "violations_last_30_days", "label": "Last 30 Days", "width": 15}
                ]
            }
        ]

    def get_summary_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        compliance = data.get("overall_compliance", 0)
        return [
            {
                "label": "Overall Compliance",
                "value": f"{compliance}%",
                "color": self.get_compliance_color(compliance)
            },
            {
                "label": "High Risk Workers",
                "value": len(data.get("high_risk_workers", [])),
                "color": "red" if len(data.get("high_risk_workers", [])) > 0 else "green"
            },
            {
                "label": "Violation Types",
                "value": len(data.get("violations_by_type", {})),
                "color": "orange"
            }
        ]


class ViolationTrendsTemplate(BaseReportTemplate):
    """Template for violation trends report."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.VIOLATION_TRENDS

    @property
    def report_name(self) -> str:
        return "Violation Trends Report"

    @property
    def required_role(self) -> str:
        return "safety_officer"

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

        trends = await aggregator.get_violation_trends(
            mine_id=mine_id,
            start_date=start_date,
            end_date=end_date
        )

        return {
            "mine_name": mine_info.get("name", "All Mines"),
            "date_range": f"{self.format_date(start_date)} to {self.format_date(end_date)}",
            "generated_at": datetime.utcnow(),
            **trends
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "Violation Trends Report",
                "include_logo": True
            },
            {
                "type": "comparison_box",
                "current_key": "current_period.total",
                "previous_key": "previous_period.total",
                "change_key": "change_percentage",
                "title": "Period Comparison"
            },
            {
                "type": "chart",
                "chart_type": "bar",
                "data_key": "current_period.by_type",
                "title": "Current Period - Violations by Type"
            },
            {
                "type": "table",
                "title": "Repeat Offenders",
                "data_key": "repeat_offenders",
                "columns": [
                    {"key": "employee_id", "label": "Employee ID"},
                    {"key": "name", "label": "Name"},
                    {"key": "violation_count", "label": "Violations"}
                ]
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Summary",
                "type": "summary",
                "sections": [
                    {
                        "title": "Period Comparison",
                        "fields": [
                            ("Current Period Violations", "current_period.total"),
                            ("Previous Period Violations", "previous_period.total"),
                            ("Change %", "change_percentage", "percentage")
                        ]
                    }
                ]
            },
            {
                "name": "Current Violations",
                "type": "dict_table",
                "data_key": "current_period.by_type",
                "columns": [
                    {"key": "key", "label": "Violation Type", "width": 25},
                    {"key": "value", "label": "Count", "width": 15}
                ]
            },
            {
                "name": "Repeat Offenders",
                "type": "data_table",
                "data_key": "repeat_offenders",
                "columns": [
                    {"key": "employee_id", "label": "Employee ID", "width": 15},
                    {"key": "name", "label": "Name", "width": 25},
                    {"key": "violation_count", "label": "Violations", "width": 15}
                ]
            }
        ]


class HighRiskWorkersTemplate(BaseReportTemplate):
    """Template for high risk workers report."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.HIGH_RISK_WORKERS

    @property
    def report_name(self) -> str:
        return "High Risk Workers Report"

    @property
    def required_role(self) -> str:
        return "safety_officer"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        from ..services.data_aggregator import DataAggregator

        filters = filters or {}
        threshold = filters.get("threshold", 70)

        aggregator = DataAggregator(self.db)
        mine_info = await self.get_mine_info(mine_id)

        workers = await aggregator.get_high_risk_workers(mine_id, threshold)

        # Group by risk level
        critical = [w for w in workers if w["compliance_score"] < 50]
        high = [w for w in workers if 50 <= w["compliance_score"] < 60]
        medium = [w for w in workers if 60 <= w["compliance_score"] < 70]

        return {
            "mine_name": mine_info.get("name", "All Mines"),
            "date_range": f"As of {self.format_date(datetime.utcnow())}",
            "generated_at": datetime.utcnow(),
            "threshold": threshold,
            "total_at_risk": len(workers),
            "critical_count": len(critical),
            "high_count": len(high),
            "medium_count": len(medium),
            "critical_workers": critical,
            "high_risk_workers": high,
            "medium_risk_workers": medium,
            "all_workers": workers
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "High Risk Workers Report",
                "include_logo": True
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "total_at_risk", "label": "Total At Risk", "color": "red"},
                    {"key": "critical_count", "label": "Critical", "color": "darkred"},
                    {"key": "high_count", "label": "High Risk", "color": "red"},
                    {"key": "medium_count", "label": "Medium Risk", "color": "orange"}
                ]
            },
            {
                "type": "table",
                "title": "Critical Risk Workers (Score < 50)",
                "data_key": "critical_workers",
                "columns": [
                    {"key": "employee_id", "label": "ID"},
                    {"key": "name", "label": "Name"},
                    {"key": "compliance_score", "label": "Score", "format": "score"},
                    {"key": "total_violations", "label": "Total Violations"},
                    {"key": "violations_last_30_days", "label": "Last 30 Days"}
                ],
                "row_color": "red"
            },
            {
                "type": "table",
                "title": "High Risk Workers (Score 50-60)",
                "data_key": "high_risk_workers",
                "columns": [
                    {"key": "employee_id", "label": "ID"},
                    {"key": "name", "label": "Name"},
                    {"key": "compliance_score", "label": "Score", "format": "score"},
                    {"key": "total_violations", "label": "Total Violations"}
                ],
                "row_color": "orange"
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Summary",
                "type": "summary",
                "sections": [
                    {
                        "title": "Risk Distribution",
                        "fields": [
                            ("Total At Risk", "total_at_risk"),
                            ("Critical (< 50)", "critical_count"),
                            ("High Risk (50-60)", "high_count"),
                            ("Medium Risk (60-70)", "medium_count")
                        ]
                    }
                ]
            },
            {
                "name": "All At-Risk Workers",
                "type": "data_table",
                "data_key": "all_workers",
                "columns": [
                    {"key": "employee_id", "label": "Employee ID", "width": 15},
                    {"key": "name", "label": "Name", "width": 25},
                    {"key": "compliance_score", "label": "Score", "width": 10},
                    {"key": "total_violations", "label": "Total Violations", "width": 18},
                    {"key": "violations_last_30_days", "label": "Last 30 Days", "width": 15}
                ],
                "conditional_format": {
                    "column": "compliance_score",
                    "rules": [
                        {"condition": "< 50", "color": "red"},
                        {"condition": "< 60", "color": "orange"},
                        {"condition": "< 70", "color": "yellow"}
                    ]
                }
            }
        ]
