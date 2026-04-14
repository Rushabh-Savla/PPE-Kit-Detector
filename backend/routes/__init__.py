"""
API Routes package.
"""
from .auth import router as auth_router
from .users import router as users_router
from .workers import router as workers_router
from .mines import router as mines_router
from .gate_entries import router as gate_entries_router
from .alerts import router as alerts_router
from .dashboards import router as dashboards_router

__all__ = [
    "auth_router",
    "users_router",
    "workers_router",
    "mines_router",
    "gate_entries_router",
    "alerts_router",
    "dashboards_router",
]
