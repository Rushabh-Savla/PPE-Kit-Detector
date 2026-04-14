"""
Report templates for Area Safety Officer role.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from .base import BaseReportTemplate
from ..schemas import ReportType


class AreaSafetyOfficerTemplate(BaseReportTemplate):
    """Template for Area Safety Officer multi-mine comparison reports."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.MINE_COMPARISON

    @property
    def report_name(self) -> str:
        return "Mine Comparison Report"

    @property
    def required_role(self) -> str:
        return "area_safety_officer"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        from ..services.data_aggregator import DataAggregator

        filters = filters or {}
        mine_ids = filters.get("mine_ids", [])

        # If no mine_ids provided, get all mines
        if not mine_ids:
            mines = await self.db.mines.find({"is_active": True}).to_list(length=None)
            mine_ids = [str(m["_id"]) for m in mines]

        aggregator = DataAggregator(self.db)

        comparison = await aggregator.get_mine_comparison(mine_ids, start_date, end_date)

        return {
            "mine_name": "Multi-Mine Overview",
            "date_range": f"{self.format_date(start_date)} to {self.format_date(end_date)}",
            "generated_at": datetime.utcnow(),
            "total_mines": len(mine_ids),
            **comparison
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "Mine Comparison Report",
                "include_logo": True
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "total_mines", "label": "Total Mines", "color": "blue"}
                ]
            },
            {
                "type": "chart",
                "chart_type": "bar",
                "data_key": "mines",
                "title": "Compliance Rate by Mine",
                "x_key": "mine_name",
                "y_key": "compliance_rate"
            },
            {
                "type": "table",
                "title": "Mine Performance Summary",
                "data_key": "mines",
                "columns": [
                    {"key": "mine_name", "label": "Mine"},
                    {"key": "location", "label": "Location"},
                    {"key": "worker_count", "label": "Workers"},
                    {"key": "compliance_rate", "label": "Compliance", "format": "percentage"},
                    {"key": "high_risk_workers", "label": "High Risk Workers"}
                ]
            },
            {
                "type": "highlight_box",
                "title": "Best Performing Mine",
                "data_key": "best_performing",
                "fields": ["mine_name", "compliance_rate"]
            },
            {
                "type": "table",
                "title": "Mines Requiring Attention",
                "data_key": "needs_attention",
                "columns": [
                    {"key": "mine_name", "label": "Mine"},
                    {"key": "compliance_rate", "label": "Compliance", "format": "percentage"},
                    {"key": "high_risk_workers", "label": "High Risk Workers"}
                ],
                "row_color": "warning"
            }
        ]

    def get_excel_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Summary",
                "type": "summary",
                "sections": [
                    {
                        "title": "Overview",
                        "fields": [
                            ("Total Mines", "total_mines"),
                            ("Best Performing", "best_performing.mine_name"),
                            ("Best Compliance", "best_performing.compliance_rate", "percentage")
                        ]
                    }
                ]
            },
            {
                "name": "Mine Comparison",
                "type": "data_table",
                "data_key": "mines",
                "columns": [
                    {"key": "mine_name", "label": "Mine Name", "width": 25},
                    {"key": "location", "label": "Location", "width": 20},
                    {"key": "worker_count", "label": "Total Workers", "width": 15},
                    {"key": "compliance_rate", "label": "Compliance %", "width": 15},
                    {"key": "high_risk_workers", "label": "High Risk Workers", "width": 18}
                ],
                "include_chart": True
            },
            {
                "name": "Needs Attention",
                "type": "data_table",
                "data_key": "needs_attention",
                "columns": [
                    {"key": "mine_name", "label": "Mine Name", "width": 25},
                    {"key": "compliance_rate", "label": "Compliance %", "width": 15},
                    {"key": "high_risk_workers", "label": "High Risk Workers", "width": 18}
                ]
            }
        ]

    def get_summary_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        best = data.get("best_performing", {})
        needs_attention = data.get("needs_attention", [])
        return [
            {
                "label": "Total Mines",
                "value": data.get("total_mines", 0),
                "color": "blue"
            },
            {
                "label": "Best Compliance",
                "value": f"{best.get('compliance_rate', 0)}%",
                "color": "green"
            },
            {
                "label": "Mines Need Attention",
                "value": len(needs_attention),
                "color": "red" if len(needs_attention) > 0 else "green"
            }
        ]


class RiskHeatmapTemplate(BaseReportTemplate):
    """Template for cross-mine risk heatmap report."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.RISK_HEATMAP

    @property
    def report_name(self) -> str:
        return "Risk Heatmap Report"

    @property
    def required_role(self) -> str:
        return "area_safety_officer"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        from ..services.data_aggregator import DataAggregator

        filters = filters or {}
        mine_ids = filters.get("mine_ids", [])

        if not mine_ids:
            mines = await self.db.mines.find({"is_active": True}).to_list(length=None)
            mine_ids = [str(m["_id"]) for m in mines]

        aggregator = DataAggregator(self.db)

        heatmap = await aggregator.get_risk_heatmap(mine_ids, start_date, end_date)

        # Group by risk level
        high_risk = [z for z in heatmap if z["risk_level"] == "high"]
        medium_risk = [z for z in heatmap if z["risk_level"] == "medium"]
        low_risk = [z for z in heatmap if z["risk_level"] == "low"]

        return {
            "mine_name": "Multi-Mine Risk Analysis",
            "date_range": f"{self.format_date(start_date)} to {self.format_date(end_date)}",
            "generated_at": datetime.utcnow(),
            "total_zones": len(heatmap),
            "high_risk_zones": high_risk,
            "medium_risk_zones": medium_risk,
            "low_risk_zones": low_risk,
            "all_zones": sorted(heatmap, key=lambda x: x["violation_rate"], reverse=True),
            "risk_summary": {
                "high": len(high_risk),
                "medium": len(medium_risk),
                "low": len(low_risk)
            }
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "Risk Heatmap Report",
                "include_logo": True
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "total_zones", "label": "Total Zones", "color": "blue"},
                    {"key": "risk_summary.high", "label": "High Risk", "color": "red"},
                    {"key": "risk_summary.medium", "label": "Medium Risk", "color": "orange"},
                    {"key": "risk_summary.low", "label": "Low Risk", "color": "green"}
                ]
            },
            {
                "type": "chart",
                "chart_type": "pie",
                "data_key": "risk_summary",
                "title": "Risk Distribution"
            },
            {
                "type": "table",
                "title": "High Risk Zones - Immediate Attention Required",
                "data_key": "high_risk_zones",
                "columns": [
                    {"key": "zone_name", "label": "Zone"},
                    {"key": "violation_count", "label": "Violations"},
                    {"key": "total_entries", "label": "Total Entries"},
                    {"key": "violation_rate", "label": "Violation %", "format": "percentage"}
                ],
                "row_color": "red"
            },
            {
                "type": "table",
                "title": "Medium Risk Zones",
                "data_key": "medium_risk_zones",
                "columns": [
                    {"key": "zone_name", "label": "Zone"},
                    {"key": "violation_count", "label": "Violations"},
                    {"key": "violation_rate", "label": "Violation %", "format": "percentage"}
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
                            ("Total Zones", "total_zones"),
                            ("High Risk Zones", "risk_summary.high"),
                            ("Medium Risk Zones", "risk_summary.medium"),
                            ("Low Risk Zones", "risk_summary.low")
                        ]
                    }
                ]
            },
            {
                "name": "All Zones",
                "type": "data_table",
                "data_key": "all_zones",
                "columns": [
                    {"key": "zone_name", "label": "Zone Name", "width": 25},
                    {"key": "mine_id", "label": "Mine ID", "width": 15},
                    {"key": "violation_count", "label": "Violations", "width": 12},
                    {"key": "total_entries", "label": "Total Entries", "width": 15},
                    {"key": "violation_rate", "label": "Violation %", "width": 15},
                    {"key": "risk_level", "label": "Risk Level", "width": 12}
                ],
                "conditional_format": {
                    "column": "risk_level",
                    "rules": [
                        {"condition": "== 'high'", "color": "red"},
                        {"condition": "== 'medium'", "color": "orange"},
                        {"condition": "== 'low'", "color": "green"}
                    ]
                }
            }
        ]


class CriticalIncidentsTemplate(BaseReportTemplate):
    """Template for critical incidents report."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.CRITICAL_INCIDENTS

    @property
    def report_name(self) -> str:
        return "Critical Incidents Report"

    @property
    def required_role(self) -> str:
        return "area_safety_officer"

    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        from ..services.data_aggregator import DataAggregator

        filters = filters or {}
        mine_ids = filters.get("mine_ids", [])

        if not mine_ids:
            mines = await self.db.mines.find({"is_active": True}).to_list(length=None)
            mine_ids = [str(m["_id"]) for m in mines]

        aggregator = DataAggregator(self.db)

        incidents = await aggregator.get_critical_incidents(mine_ids, start_date, end_date)

        # Separate by severity
        critical = [i for i in incidents if i["severity"] == "critical"]
        high = [i for i in incidents if i["severity"] == "high"]

        return {
            "mine_name": "Multi-Mine Incidents",
            "date_range": f"{self.format_date(start_date)} to {self.format_date(end_date)}",
            "generated_at": datetime.utcnow(),
            "total_incidents": len(incidents),
            "critical_count": len(critical),
            "high_count": len(high),
            "critical_incidents": critical,
            "high_incidents": high,
            "all_incidents": incidents
        }

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "Critical Incidents Report",
                "include_logo": True
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "total_incidents", "label": "Total Incidents", "color": "red"},
                    {"key": "critical_count", "label": "Critical", "color": "darkred"},
                    {"key": "high_count", "label": "High Severity", "color": "orange"}
                ]
            },
            {
                "type": "table",
                "title": "Critical Incidents",
                "data_key": "critical_incidents",
                "columns": [
                    {"key": "created_at", "label": "Date/Time", "format": "datetime"},
                    {"key": "mine_name", "label": "Mine"},
                    {"key": "type", "label": "Type"},
                    {"key": "message", "label": "Description"},
                    {"key": "status", "label": "Status"}
                ],
                "row_color": "red"
            },
            {
                "type": "table",
                "title": "High Severity Incidents",
                "data_key": "high_incidents",
                "columns": [
                    {"key": "created_at", "label": "Date/Time", "format": "datetime"},
                    {"key": "mine_name", "label": "Mine"},
                    {"key": "type", "label": "Type"},
                    {"key": "message", "label": "Description"},
                    {"key": "status", "label": "Status"}
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
                        "title": "Incident Summary",
                        "fields": [
                            ("Total Incidents", "total_incidents"),
                            ("Critical", "critical_count"),
                            ("High Severity", "high_count")
                        ]
                    }
                ]
            },
            {
                "name": "All Incidents",
                "type": "data_table",
                "data_key": "all_incidents",
                "columns": [
                    {"key": "created_at", "label": "Date/Time", "width": 20},
                    {"key": "severity", "label": "Severity", "width": 12},
                    {"key": "mine_name", "label": "Mine", "width": 20},
                    {"key": "type", "label": "Type", "width": 20},
                    {"key": "message", "label": "Description", "width": 40},
                    {"key": "status", "label": "Status", "width": 12}
                ]
            }
        ]
