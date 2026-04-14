"""
Report templates package.
"""

from .base import BaseReportTemplate
from .shift_incharge import ShiftInchargeTemplate
from .safety_officer import SafetyOfficerTemplate
from .manager import ManagerTemplate
from .area_safety_officer import AreaSafetyOfficerTemplate
from .general_manager import GeneralManagerTemplate
from .worker import WorkerTemplate
from .emergency_incident import EmergencyIncidentTemplate

__all__ = [
    "BaseReportTemplate",
    "ShiftInchargeTemplate",
    "SafetyOfficerTemplate",
    "ManagerTemplate",
    "AreaSafetyOfficerTemplate",
    "GeneralManagerTemplate",
    "WorkerTemplate",
    "EmergencyIncidentTemplate",
]
