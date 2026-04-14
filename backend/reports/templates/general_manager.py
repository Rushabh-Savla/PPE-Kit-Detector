"""
Report templates for General Manager role.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from .base import BaseReportTemplate
from ..schemas import ReportType


class GeneralManagerTemplate(BaseReportTemplate):
    """Template for General Manager executive reports."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.EXECUTIVE_SUMMARY

    @property
    def report_name(self) -> str:
        return "Executive Summary Report"

    @property
    def required_role(self) -> str:
        return "general_manager"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        from ..services.data_aggregator import DataAggregator

        aggregator = DataAggregator(self.db)

        executive = await aggregator.get_executive_summary(start_date, end_date)

        return {
            "mine_name": "Organization Overview",
            "date_range": f"{self.format_date(start_date)} to {self.format_date(end_date)}",
            "generated_at": datetime.utcnow(),
            **executive
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "Executive Summary Report",
                "include_logo": True,
                "subtitle": "Organization-Wide Performance Overview"
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "kpis.total_mines", "label": "Total Mines", "color": "blue"},
                    {"key": "kpis.total_workers", "label": "Total Workers", "color": "blue"},
                    {"key": "kpis.overall_compliance", "label": "Org Compliance", "color": "auto", "format": "percentage"},
                    {"key": "kpis.total_violations", "label": "Total Violations", "color": "red"}
                ]
            },
            {
                "type": "regulatory_box",
                "title": "Regulatory Compliance Status",
                "data_key": "regulatory_status",
                "threshold_key": "threshold",
                "current_key": "current_compliance",
                "status_key": "status"
            },
            {
                "type": "chart",
                "chart_type": "bar",
                "data_key": "mine_performance",
                "title": "Mine Performance Comparison",
                "x_key": "mine_name",
                "y_key": "compliance_rate"
            },
            {
                "type": "two_column",
                "left": {
                    "type": "highlight_box",
                    "title": "Best Performing Mine",
                    "data_key": "best_performing_mine",
                    "fields": ["mine_name", "compliance_rate", "worker_count"]
                },
                "right": {
                    "type": "table",
                    "title": "Mines Needing Attention",
                    "data_key": "mines_needing_attention",
                    "columns": [
                        {"key": "mine_name", "label": "Mine"},
                        {"key": "compliance_rate", "label": "Compliance"}
                    ],
                    "max_rows": 5
                }
            },
            {
                "type": "table",
                "title": "Critical Incidents",
                "data_key": "critical_incidents",
                "columns": [
                    {"key": "created_at", "label": "Date", "format": "date"},
                    {"key": "severity", "label": "Severity", "format": "severity_badge"},
                    {"key": "message", "label": "Description"}
                ],
                "max_rows": 5
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Executive Summary",
                "type": "summary",
                "sections": [
                    {
                        "title": "Key Performance Indicators",
                        "fields": [
                            ("Total Mines", "kpis.total_mines"),
                            ("Total Workers", "kpis.total_workers"),
                            ("Total Entries", "kpis.total_entries"),
                            ("Overall Compliance", "kpis.overall_compliance", "percentage"),
                            ("Total Violations", "kpis.total_violations")
                        ]
                    },
                    {
                        "title": "Regulatory Status",
                        "fields": [
                            ("Compliance Threshold", "regulatory_status.threshold", "percentage"),
                            ("Current Compliance", "regulatory_status.current_compliance", "percentage"),
                            ("Status", "regulatory_status.status")
                        ]
                    }
                ]
            },
            {
                "name": "Mine Performance",
                "type": "data_table",
                "data_key": "mine_performance",
                "columns": [
                    {"key": "mine_name", "label": "Mine Name", "width": 25},
                    {"key": "location", "label": "Location", "width": 20},
                    {"key": "worker_count", "label": "Workers", "width": 12},
                    {"key": "compliance_rate", "label": "Compliance %", "width": 15},
                    {"key": "high_risk_workers", "label": "High Risk", "width": 12}
                ],
                "include_chart": True
            },
            {
                "name": "Critical Incidents",
                "type": "data_table",
                "data_key": "critical_incidents",
                "columns": [
                    {"key": "created_at", "label": "Date/Time", "width": 20},
                    {"key": "severity", "label": "Severity", "width": 12},
                    {"key": "message", "label": "Description", "width": 50}
                ]
            }
        ]

    def get_summary_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        kpis = data.get("kpis", {})
        compliance = kpis.get("overall_compliance", 0)
        regulatory = data.get("regulatory_status", {})

        return [
            {
                "label": "Total Mines",
                "value": kpis.get("total_mines", 0),
                "color": "blue"
            },
            {
                "label": "Organization Compliance",
                "value": f"{compliance}%",
                "color": self.get_compliance_color(compliance)
            },
            {
                "label": "Regulatory Status",
                "value": regulatory.get("status", "Unknown").title(),
                "color": "green" if regulatory.get("status") == "compliant" else "red"
            },
            {
                "label": "Critical Incidents",
                "value": len(data.get("critical_incidents", [])),
                "color": "red" if len(data.get("critical_incidents", [])) > 0 else "green"
            }
        ]


class KPIDashboardTemplate(BaseReportTemplate):
    """Template for KPI dashboard report."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.KPI_DASHBOARD

    @property
    def report_name(self) -> str:
        return "KPI Dashboard Report"

    @property
    def required_role(self) -> str:
        return "general_manager"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        from ..services.data_aggregator import DataAggregator

        aggregator = DataAggregator(self.db)

        executive = await aggregator.get_executive_summary(start_date, end_date)

        # Calculate trends by getting previous period
        from datetime import timedelta
        duration = end_date - start_date
        prev_start = start_date - duration
        prev_end = start_date

        prev_data = await aggregator.get_executive_summary(prev_start, prev_end)

        # Calculate changes
        current_compliance = executive["kpis"]["overall_compliance"]
        prev_compliance = prev_data["kpis"]["overall_compliance"]
        compliance_change = current_compliance - prev_compliance

        current_violations = executive["kpis"]["total_violations"]
        prev_violations = prev_data["kpis"]["total_violations"]
        violation_change = ((current_violations - prev_violations) / prev_violations * 100) if prev_violations > 0 else 0

        return {
            "mine_name": "KPI Dashboard",
            "date_range": f"{self.format_date(start_date)} to {self.format_date(end_date)}",
            "generated_at": datetime.utcnow(),
            "kpis": executive["kpis"],
            "trends": {
                "compliance_change": round(compliance_change, 1),
                "violation_change": round(violation_change, 1),
                "compliance_direction": "up" if compliance_change > 0 else "down",
                "violation_direction": "up" if violation_change > 0 else "down"
            },
            "mine_performance": executive["mine_performance"],
            "regulatory_status": executive["regulatory_status"]
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "KPI Dashboard",
                "include_logo": True
            },
            {
                "type": "kpi_cards",
                "cards": [
                    {
                        "title": "Overall Compliance",
                        "value_key": "kpis.overall_compliance",
                        "format": "percentage",
                        "trend_key": "trends.compliance_change",
                        "trend_direction_key": "trends.compliance_direction"
                    },
                    {
                        "title": "Total Entries",
                        "value_key": "kpis.total_entries"
                    },
                    {
                        "title": "Total Violations",
                        "value_key": "kpis.total_violations",
                        "trend_key": "trends.violation_change",
                        "trend_direction_key": "trends.violation_direction",
                        "invert_trend": True
                    }
                ]
            },
            {
                "type": "chart",
                "chart_type": "bar",
                "data_key": "mine_performance",
                "title": "Mine Compliance Ranking",
                "x_key": "mine_name",
                "y_key": "compliance_rate",
                "sorted": True
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "KPI Summary",
                "type": "summary",
                "sections": [
                    {
                        "title": "Key Metrics",
                        "fields": [
                            ("Overall Compliance", "kpis.overall_compliance", "percentage"),
                            ("Compliance Change", "trends.compliance_change", "percentage_change"),
                            ("Total Entries", "kpis.total_entries"),
                            ("Total Violations", "kpis.total_violations"),
                            ("Violation Change", "trends.violation_change", "percentage_change")
                        ]
                    }
                ]
            },
            {
                "name": "Mine Rankings",
                "type": "data_table",
                "data_key": "mine_performance",
                "columns": [
                    {"key": "mine_name", "label": "Mine", "width": 25},
                    {"key": "compliance_rate", "label": "Compliance %", "width": 15},
                    {"key": "worker_count", "label": "Workers", "width": 12}
                ]
            }
        ]


class FinancialImpactTemplate(BaseReportTemplate):
    """Template for financial impact report."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.FINANCIAL_IMPACT

    @property
    def report_name(self) -> str:
        return "Financial Impact Report"

    @property
    def required_role(self) -> str:
        return "general_manager"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        from ..services.data_aggregator import DataAggregator

        aggregator = DataAggregator(self.db)

        financial = await aggregator.get_financial_impact(start_date, end_date)

        return {
            "mine_name": "Financial Analysis",
            "date_range": f"{self.format_date(start_date)} to {self.format_date(end_date)}",
            "generated_at": datetime.utcnow(),
            **financial
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "Financial Impact Report",
                "include_logo": True
            },
            {
                "type": "financial_summary",
                "metrics": [
                    {
                        "label": "Estimated Cost Savings",
                        "value_key": "estimated_cost_savings",
                        "format": "currency",
                        "color": "green"
                    },
                    {
                        "label": "Incidents Prevented",
                        "value_key": "incidents_prevented",
                        "color": "green"
                    },
                    {
                        "label": "Violations Caught",
                        "value_key": "violations_caught",
                        "color": "blue"
                    }
                ]
            },
            {
                "type": "insight_box",
                "title": "ROI Analysis",
                "content": "By catching violations at the gate and preventing potential incidents, the PPE detection system has contributed to significant cost avoidance."
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Financial Impact",
                "type": "summary",
                "sections": [
                    {
                        "title": "Cost Analysis",
                        "fields": [
                            ("Violations Caught at Gate", "violations_caught"),
                            ("Total Violations", "total_violations"),
                            ("Estimated Incidents Prevented", "incidents_prevented"),
                            ("Estimated Cost Savings", "estimated_cost_savings", "currency"),
                            ("Violation Processing Cost", "violation_processing_cost", "currency")
                        ]
                    }
                ]
            }
        ]
