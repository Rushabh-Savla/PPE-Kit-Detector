"""
Base template class for all report templates.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..schemas import ReportType, ReportFormat


class BaseReportTemplate(ABC):
    """
    Abstract base class for report templates.

    All role-specific templates must inherit from this class
    and implement the required methods.
    """

    def __init__(self, db):
        """
        Initialize the template with database connection.

        Args:
            db: MongoDB database instance
        """
        self.db = db

    @property
    @abstractmethod
    def report_type(self) -> ReportType:
        """Return the report type this template handles."""
        pass

    @property
    @abstractmethod
    def report_name(self) -> str:
        """Return the human-readable report name."""
        pass

    @property
    @abstractmethod
    def required_role(self) -> str:
        """Return the minimum role required to generate this report."""
        pass

    @abstractmethod
    async def aggregate_data(
        self,
        start_date: datetime,
        end_date: datetime,
        mine_id: Optional[str] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Aggregate data from database for the report.

        Args:
            start_date: Report start date
            end_date: Report end date
            mine_id: Optional mine ID filter
            filters: Additional filters

        Returns:
            Dictionary containing all report data
        """
        pass

    @abstractmethod
    def get_pdf_structure(self) -> List[Dict[str, Any]]:
        """
        Define the PDF layout structure.

        Returns:
            List of section definitions for PDF generation
        """
        pass

    @abstractmethod
    def get_excel_structure(self) -> List[Dict[str, Any]]:
        """
        Define the Excel sheet structure.

        Returns:
            List of sheet definitions for Excel generation
        """
        pass

    def get_report_title(self, data: Dict[str, Any]) -> str:
        """
        Generate the report title.

        Args:
            data: Report data

        Returns:
            Formatted report title
        """
        mine_name = data.get("mine_name", "All Mines")
        date_range = data.get("date_range", "")
        return f"{self.report_name} - {mine_name}\n{date_range}"

    def get_summary_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract summary metrics for the report header.

        Args:
            data: Report data

        Returns:
            List of metric dictionaries with label, value, and color
        """
        return []

    def format_date(self, dt: datetime) -> str:
        """Format datetime for display."""
        if isinstance(dt, str):
            return dt
        return dt.strftime("%Y-%m-%d")

    def format_datetime(self, dt: datetime) -> str:
        """Format datetime with time for display."""
        if isinstance(dt, str):
            return dt
        return dt.strftime("%Y-%m-%d %H:%M")

    def format_percentage(self, value: float) -> str:
        """Format percentage value."""
        return f"{value:.1f}%"

    def get_compliance_color(self, score: float) -> str:
        """
        Get color based on compliance score.

        Args:
            score: Compliance score (0-100)

        Returns:
            Color code string
        """
        if score >= 90:
            return "green"
        elif score >= 70:
            return "orange"
        else:
            return "red"

    def get_risk_color(self, risk_level: str) -> str:
        """
        Get color based on risk level.

        Args:
            risk_level: Risk level string

        Returns:
            Color code string
        """
        risk_colors = {
            "low": "green",
            "medium": "orange",
            "high": "red",
            "critical": "darkred"
        }
        return risk_colors.get(risk_level.lower(), "gray")

    async def get_mine_info(self, mine_id: str) -> Dict[str, Any]:
        """
        Get mine information.

        Args:
            mine_id: Mine ID

        Returns:
            Mine document or empty dict
        """
        from bson import ObjectId
        if not mine_id:
            return {"name": "All Mines"}

        mine = await self.db.mines.find_one({"_id": ObjectId(mine_id)})
        return mine or {"name": "Unknown Mine"}

    async def get_zone_info(self, zone_id: str) -> Dict[str, Any]:
        """
        Get zone information.

        Args:
            zone_id: Zone ID

        Returns:
            Zone document or empty dict
        """
        from bson import ObjectId
        if not zone_id:
            return {"name": "All Zones"}

        zone = await self.db.zones.find_one({"_id": ObjectId(zone_id)})
        return zone or {"name": "Unknown Zone"}

    def calculate_date_range(self, range_type: str) -> tuple:
        """
        Calculate start and end dates from range type.

        Args:
            range_type: One of previous_shift, previous_day, previous_week, previous_month

        Returns:
            Tuple of (start_date, end_date)
        """
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
            start_date = (end_date - timedelta(days=1)).replace(day=1)
        else:
            # Default to previous day
            end_date = now
            start_date = now - timedelta(days=1)

        return start_date, end_date
