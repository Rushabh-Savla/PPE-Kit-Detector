"""
Reports Package for Mine Safety PPE Detection System.

This package provides comprehensive report generation capabilities including:
- Role-specific report templates
- PDF and Excel export
- Scheduled email delivery
- Report history tracking
"""

from .router import router as reports_router

__all__ = ["reports_router"]
