"""
Emergency Incident Report Template.
Generates comprehensive PDF reports for gas emergencies and evacuations.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from .base import BaseReportTemplate
from ..schemas import ReportType


class EmergencyIncidentTemplate(BaseReportTemplate):
    """Template for Emergency Incident Reports (Gas Leaks, Evacuations, etc.)"""

    @property
    def report_type(self) -> ReportType:
        return ReportType.CRITICAL_INCIDENTS

    @property
    def report_name(self) -> str:
        return "Emergency Incident Report"

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
        """
        For the demo, we return hardcoded data for the Dec 12, 2024 gas emergency.
        In production, this would query the database.
        """
        filters = filters or {}

        # Demo data for the Dec 12 gas emergency
        return get_demo_emergency_data()

    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "header",
                "title": "EMERGENCY INCIDENT REPORT",
                "include_logo": True
            },
            # Executive Summary Section
            {
                "type": "alert_box",
                "title": "CRITICAL GAS EMERGENCY",
                "data_key": "incident_summary.status",
                "severity": "danger"
            },
            {
                "type": "metrics_row",
                "metrics": [
                    {"key": "incident_summary.severity", "label": "Severity Level", "color": "red"},
                    {"key": "incident_summary.workers_affected", "label": "Workers Affected", "color": "orange"},
                    {"key": "incident_summary.workers_evacuated", "label": "Safely Evacuated", "color": "green"},
                    {"key": "incident_summary.response_time", "label": "Response Time", "color": "blue"}
                ]
            },
            # Incident Details Table
            {
                "type": "table",
                "title": "Incident Details",
                "data_key": "incident_details",
                "columns": [
                    {"key": "field", "label": "Field"},
                    {"key": "value", "label": "Value"}
                ]
            },
            # Gas Readings Chart
            {
                "type": "chart",
                "chart_type": "line",
                "title": "Methane Level Timeline",
                "data_key": "gas_readings",
                "x_key": "time",
                "y_key": "ppm"
            },
            # Affected Workers Table
            {
                "type": "table",
                "title": "Affected Workers",
                "data_key": "affected_workers",
                "columns": [
                    {"key": "name", "label": "Worker Name"},
                    {"key": "employee_id", "label": "Employee ID"},
                    {"key": "zone", "label": "Zone"},
                    {"key": "evacuation_time", "label": "Evacuation Time"},
                    {"key": "status", "label": "Status"}
                ]
            },
            # Timeline of Events
            {
                "type": "table",
                "title": "Timeline of Events",
                "data_key": "timeline",
                "columns": [
                    {"key": "time", "label": "Time"},
                    {"key": "event", "label": "Event"},
                    {"key": "action", "label": "Action Taken"}
                ]
            },
            # Response Analysis Chart
            {
                "type": "chart",
                "chart_type": "pie",
                "title": "Response Time Distribution",
                "data_key": "response_breakdown"
            },
            # Notifications Sent
            {
                "type": "table",
                "title": "Notifications Sent",
                "data_key": "notifications",
                "columns": [
                    {"key": "recipient", "label": "Recipient"},
                    {"key": "type", "label": "Type"},
                    {"key": "time", "label": "Time Sent"},
                    {"key": "status", "label": "Status"}
                ]
            },
            # Recommendations
            {
                "type": "table",
                "title": "Recommendations",
                "data_key": "recommendations",
                "columns": [
                    {"key": "priority", "label": "Priority"},
                    {"key": "recommendation", "label": "Recommendation"},
                    {"key": "department", "label": "Department"}
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
                        "title": "Incident Summary",
                        "fields": [
                            ("Incident Type", "incident_summary.type"),
                            ("Severity", "incident_summary.severity"),
                            ("Location", "incident_summary.location"),
                            ("Workers Affected", "incident_summary.workers_affected"),
                            ("Workers Evacuated", "incident_summary.workers_evacuated"),
                            ("Response Time", "incident_summary.response_time"),
                            ("Outcome", "incident_summary.outcome")
                        ]
                    }
                ]
            },
            {
                "name": "Affected Workers",
                "type": "data_table",
                "data_key": "affected_workers",
                "columns": [
                    {"key": "name", "label": "Worker Name", "width": 25},
                    {"key": "employee_id", "label": "Employee ID", "width": 15},
                    {"key": "zone", "label": "Zone", "width": 20},
                    {"key": "evacuation_time", "label": "Evacuation Time", "width": 15},
                    {"key": "status", "label": "Status", "width": 15}
                ]
            },
            {
                "name": "Timeline",
                "type": "data_table",
                "data_key": "timeline",
                "columns": [
                    {"key": "time", "label": "Time", "width": 15},
                    {"key": "event", "label": "Event", "width": 40},
                    {"key": "action", "label": "Action Taken", "width": 40}
                ]
            }
        ]

    def get_summary_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        summary = data.get("incident_summary", {})
        return [
            {
                "label": "Severity",
                "value": summary.get("severity", "N/A"),
                "color": "red"
            },
            {
                "label": "Workers Affected",
                "value": summary.get("workers_affected", 0),
                "color": "orange"
            },
            {
                "label": "Safely Evacuated",
                "value": summary.get("workers_evacuated", 0),
                "color": "green"
            },
            {
                "label": "Response Time",
                "value": summary.get("response_time", "N/A"),
                "color": "blue"
            }
        ]


def get_demo_emergency_data() -> Dict[str, Any]:
    """
    Returns hardcoded demo data for the Dec 12, 2024 gas emergency.
    This is used for the SIH presentation demo.
    """
    return {
        "mine_name": "Jharia Coal Mine",
        "date_range": "December 12, 2025",
        "generated_at": datetime(2025, 12, 12, 4, 0, 0),

        "incident_summary": {
            "type": "Gas Emergency - Methane Spike",
            "severity": "CRITICAL",
            "status": "RESOLVED - All Workers Safely Evacuated",
            "location": "Zone A - Extraction",
            "gas_type": "Methane (CH4)",
            "peak_level": "15,200 PPM",
            "threshold": "5,000 PPM",
            "workers_affected": 2,
            "workers_evacuated": 2,
            "response_time": "3 min 25 sec",
            "outcome": "Zero Casualties",
            "incident_date": "December 12, 2024",
            "incident_time": "3:30 AM IST"
        },

        "incident_details": [
            {"field": "Incident Type", "value": "Gas Emergency - Methane Spike"},
            {"field": "Date & Time", "value": "December 12, 2024, 3:30 AM IST"},
            {"field": "Location", "value": "Zone A - Extraction, Jharia Coal Mine"},
            {"field": "Gas Detected", "value": "Methane (CH4)"},
            {"field": "Peak Reading", "value": "15,200 PPM (CRITICAL)"},
            {"field": "Safe Threshold", "value": "5,000 PPM"},
            {"field": "Sensor ID", "value": "HELMET-001 (Worker: Rushabh Savla)"},
            {"field": "Detection Method", "value": "Smart Helmet MQ-4 Sensor"},
            {"field": "Alarm Triggered", "value": "Automatic + Manual Evacuation"},
            {"field": "Evacuation Complete", "value": "3:33 AM IST"},
            {"field": "All Clear Given", "value": "4:15 AM IST"},
        ],

        "gas_readings": [
            {"time": "3:25", "ppm": 800},
            {"time": "3:26", "ppm": 1200},
            {"time": "3:27", "ppm": 2500},
            {"time": "3:28", "ppm": 5800},
            {"time": "3:29", "ppm": 9200},
            {"time": "3:30", "ppm": 15200},
            {"time": "3:31", "ppm": 14500},
            {"time": "3:32", "ppm": 12000},
            {"time": "3:33", "ppm": 8500},
            {"time": "3:35", "ppm": 5000},
            {"time": "3:40", "ppm": 2800},
            {"time": "3:50", "ppm": 1200},
            {"time": "4:00", "ppm": 600},
        ],

        "affected_workers": [
            {"name": "Rushabh Savla", "employee_id": "EMP-2024-001", "zone": "Zone A - Extraction", "evacuation_time": "3 min 25 sec", "status": "SAFE"},
            {"name": "Sourav Maity", "employee_id": "EMP-2024-002", "zone": "Zone A - Extraction", "evacuation_time": "3 min 30 sec", "status": "SAFE"},
        ],

        "timeline": [
            {"time": "3:25 AM", "event": "Elevated methane levels detected", "action": "Monitoring initiated"},
            {"time": "3:28 AM", "event": "Methane exceeds warning threshold (5,000 PPM)", "action": "Alert generated to Safety Officer"},
            {"time": "3:29 AM", "event": "Helmet sensor confirms gas spike", "action": "Worker Rushabh Savla notified via helmet alarm"},
            {"time": "3:30 AM", "event": "CRITICAL: Methane reaches 15,200 PPM", "action": "Mass evacuation triggered by Safety Officer"},
            {"time": "3:30 AM", "event": "All 2 helmet buzzers activated", "action": "ESP32 EVACUATE_ALL command sent"},
            {"time": "3:30 AM", "event": "SMS alert sent to Safety Officer", "action": "Emergency notification delivered"},
            {"time": "3:31 AM", "event": "Workers begin evacuation", "action": "Real-time tracking initiated"},
            {"time": "3:33 AM", "event": "All 2 workers evacuated from Zone A", "action": "Headcount verified at assembly point"},
            {"time": "3:35 AM", "event": "Ventilation systems activated", "action": "Gas dispersal in progress"},
            {"time": "3:50 AM", "event": "Methane levels dropping", "action": "Continued monitoring"},
            {"time": "4:00 AM", "event": "Methane below safe threshold", "action": "Preliminary all-clear assessment"},
            {"time": "4:15 AM", "event": "Zone declared safe", "action": "Workers cleared to return"},
        ],

        "response_breakdown": {
            "Detection to Alert": 15,
            "Alert to Evacuation Trigger": 8,
            "Evacuation Time": 25,
            "Verification": 12,
        },

        "notifications": [
            {"recipient": "Safety Officer (Tanush Maloo)", "type": "SMS", "time": "3:30:05 AM", "status": "Delivered"},
            {"recipient": "Safety Officer (Tanush Maloo)", "type": "Dashboard Alert", "time": "3:30:01 AM", "status": "Acknowledged"},
            {"recipient": "Mine Manager", "type": "Email", "time": "3:30:15 AM", "status": "Delivered"},
            {"recipient": "All Zone A Workers (2)", "type": "Helmet Buzzer", "time": "3:30:02 AM", "status": "Activated"},
            {"recipient": "Emergency Response Team", "type": "Radio Alert", "time": "3:30:20 AM", "status": "Acknowledged"},
        ],

        "recommendations": [
            {"priority": "HIGH", "recommendation": "Inspect Zone A ventilation systems for blockages", "department": "Maintenance"},
            {"priority": "HIGH", "recommendation": "Conduct geological survey of methane source", "department": "Mining Engineering"},
            {"priority": "MEDIUM", "recommendation": "Install additional methane sensors in Zone A", "department": "Safety"},
            {"priority": "MEDIUM", "recommendation": "Review and update evacuation drills frequency", "department": "Training"},
            {"priority": "LOW", "recommendation": "Document incident for regulatory compliance report", "department": "Compliance"},
        ],

        "compliance_status": {
            "dgms_notification": "Submitted",
            "incident_report_filed": True,
            "worker_statements_collected": True,
            "equipment_inspection_scheduled": True,
        }
    }
