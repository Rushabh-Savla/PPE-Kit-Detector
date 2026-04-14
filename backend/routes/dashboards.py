"""
Role-specific dashboard routes.
"""
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from bson import ObjectId
from database import get_database
from auth import (
    get_current_user, get_shift_incharge_or_above, get_safety_officer_or_above,
    get_manager_or_above, get_area_safety_officer_or_above, get_general_manager,
    get_worker, UserRole, check_mine_access
)
from schemas import (
    ShiftType, SHIFT_DEFINITIONS, AlertSeverity
)

router = APIRouter(prefix="/dashboard", tags=["Dashboards"])


# ==================== Super Admin Dashboard ====================

@router.get("/super-admin")
async def get_super_admin_dashboard(
    current_user: dict = Depends(get_current_user)
):
    """
    Dashboard for Super Admin:
    - System-wide statistics
    - All mines overview
    - All workers count
    - All users count
    - Active alerts
    - Today's entries and violations
    - Overall compliance rate
    """
    db = get_database()

    # Verify super admin role
    if current_user.get("role") != "super_admin":
        raise HTTPException(status_code=403, detail="Super admin access required")

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    week_ago = today - timedelta(days=7)

    # Get counts in parallel using aggregation for efficiency
    total_mines = await db.mines.count_documents({"is_active": True})
    total_workers = await db.workers.count_documents({"is_active": True})
    total_users = await db.users.count_documents({"is_active": True})

    # Today's entries (distinct workers who entered today)
    today_entries = await db.gate_entries.count_documents({
        "timestamp": {"$gte": today, "$lt": tomorrow},
        "entry_type": "entry"
    })

    # Today's violations
    today_violations = await db.gate_entries.count_documents({
        "timestamp": {"$gte": today, "$lt": tomorrow},
        "violations": {"$ne": []}
    })

    # Active alerts
    active_alerts = await db.alerts.count_documents({
        "status": "active"
    })

    # Week compliance rate
    week_entries = await db.gate_entries.count_documents({
        "timestamp": {"$gte": week_ago},
        "entry_type": "entry"
    })
    week_violations = await db.gate_entries.count_documents({
        "timestamp": {"$gte": week_ago},
        "violations": {"$ne": []}
    })

    compliance_rate = 100.0
    if week_entries > 0:
        compliance_rate = round((1 - week_violations / week_entries) * 100, 1)

    # System status based on alerts and compliance
    system_status = "healthy"
    if active_alerts > 10 or compliance_rate < 80:
        system_status = "critical"
    elif active_alerts > 5 or compliance_rate < 90:
        system_status = "warning"

    # Get recent alerts
    alerts_cursor = db.alerts.find({
        "status": "active"
    }).sort("created_at", -1).limit(5)

    recent_alerts = []
    async for alert in alerts_cursor:
        mine = await db.mines.find_one({"_id": alert.get("mine_id")})
        recent_alerts.append({
            "id": str(alert["_id"]),
            "alert_type": alert["alert_type"],
            "severity": alert["severity"],
            "status": alert["status"],
            "message": alert["message"],
            "mine_id": str(alert.get("mine_id", "")),
            "mine_name": mine["name"] if mine else "Unknown",
            "created_at": alert["created_at"].isoformat(),
        })

    # Get recent mines
    mines_cursor = db.mines.find({"is_active": True}).sort("created_at", -1).limit(5)
    recent_mines = []
    async for mine in mines_cursor:
        # Get zone and gate counts
        zone_count = await db.zones.count_documents({"mine_id": mine["_id"]})
        gate_count = await db.gates.count_documents({"mine_id": mine["_id"]})

        recent_mines.append({
            "id": str(mine["_id"]),
            "name": mine["name"],
            "location": mine.get("location", ""),
            "is_active": mine.get("is_active", True),
            "zones": [{"id": str(i)} for i in range(zone_count)],  # Placeholder for count
            "gates": [{"id": str(i)} for i in range(gate_count)],  # Placeholder for count
            "created_at": mine.get("created_at", datetime.utcnow()).isoformat(),
        })

    return {
        "stats": {
            "total_mines": total_mines,
            "total_workers": total_workers,
            "total_users": total_users,
            "active_alerts": active_alerts,
            "today_entries": today_entries,
            "today_violations": today_violations,
            "compliance_rate": compliance_rate,
            "system_status": system_status,
        },
        "recent_alerts": recent_alerts,
        "recent_mines": recent_mines,
        # Legacy format for compatibility
        "total_employees": total_workers,
        "present_today": today_entries,
        "absent_today": max(0, total_workers - today_entries),
        "violations_today": today_violations,
        "violations_this_week": week_violations,
    }


def get_current_shift() -> ShiftType:
    """Determine current shift based on time."""
    now = datetime.utcnow()
    hour = now.hour

    if 6 <= hour < 14:
        return ShiftType.DAY
    elif 14 <= hour < 22:
        return ShiftType.AFTERNOON
    else:
        return ShiftType.NIGHT


def get_shift_start_time(shift: ShiftType, date: datetime = None) -> datetime:
    """Get the start time of a shift."""
    if date is None:
        date = datetime.utcnow()

    if shift == ShiftType.DAY:
        return date.replace(hour=6, minute=0, second=0, microsecond=0)
    elif shift == ShiftType.AFTERNOON:
        return date.replace(hour=14, minute=0, second=0, microsecond=0)
    else:  # Night
        if date.hour < 6:
            return (date - timedelta(days=1)).replace(hour=22, minute=0, second=0, microsecond=0)
        return date.replace(hour=22, minute=0, second=0, microsecond=0)


# ==================== Shift Incharge Dashboard ====================

@router.get("/shift-incharge")
async def get_shift_incharge_dashboard(
    gate_id: Optional[str] = None,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """
    Dashboard for Shift Incharge:
    - Live workers entering/exiting
    - Real-time PPE compliance
    - Shift attendance
    - Active alerts
    """
    db = get_database()

    current_shift = get_current_shift()
    shift_start = get_shift_start_time(current_shift)
    shift_info = SHIFT_DEFINITIONS[current_shift]

    # Get user's mine
    mine_id = current_user.get("mine_id")
    if not mine_id:
        return {"error": "No mine assigned to user"}

    mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
    mine_name = mine["name"] if mine else "Unknown"

    query = {
        "mine_id": ObjectId(mine_id),
        "timestamp": {"$gte": shift_start},
        "shift": current_shift.value
    }

    if gate_id:
        query["gate_id"] = gate_id

    # Get entry/exit counts
    total_entries = await db.gate_entries.count_documents({
        **query,
        "entry_type": "entry"
    })

    total_exits = await db.gate_entries.count_documents({
        **query,
        "entry_type": "exit"
    })

    currently_inside = total_entries - total_exits

    # Get violation counts
    violations_this_shift = await db.gate_entries.count_documents({
        **query,
        "violations": {"$ne": []}
    })

    compliant_entries = total_entries - violations_this_shift

    # Get expected workers for this shift
    expected_workers = await db.workers.count_documents({
        "mine_id": ObjectId(mine_id),
        "assigned_shift": current_shift.value,
        "is_active": True
    })

    # Get pending alerts
    pending_alerts = await db.alerts.count_documents({
        "mine_id": ObjectId(mine_id),
        "status": {"$in": ["active", "acknowledged"]}
    })

    # Get recent entries
    recent_entries_cursor = db.gate_entries.find(query).sort("timestamp", -1).limit(10)
    recent_entries = []

    async for entry in recent_entries_cursor:
        recent_entries.append({
            "id": str(entry["_id"]),
            "worker_name": entry.get("worker_name", "Unknown"),
            "employee_id": entry.get("employee_id"),
            "entry_type": entry.get("entry_type"),
            "status": entry.get("status"),
            "violations": entry.get("violations", []),
            "timestamp": entry["timestamp"].isoformat(),
            "ppe_status": entry.get("ppe_status", {}),
        })

    # Get active alerts for this mine
    alerts_cursor = db.alerts.find({
        "mine_id": ObjectId(mine_id),
        "status": {"$in": ["active", "acknowledged"]}
    }).sort("created_at", -1).limit(5)

    active_alerts = []
    async for alert in alerts_cursor:
        active_alerts.append({
            "id": str(alert["_id"]),
            "alert_type": alert["alert_type"],
            "severity": alert["severity"],
            "message": alert["message"],
            "created_at": alert["created_at"].isoformat(),
        })

    return {
        "mine_id": mine_id,
        "mine_name": mine_name,
        "current_shift": current_shift.value,
        "shift_name": shift_info.name,
        "shift_start": shift_info.start_time,
        "shift_end": shift_info.end_time,
        "statistics": {
            "expected_workers": expected_workers,
            "workers_entered": total_entries,
            "workers_exited": total_exits,
            "currently_inside": currently_inside,
            "ppe_compliant": compliant_entries,
            "ppe_non_compliant": violations_this_shift,
            "violations_this_shift": violations_this_shift,
            "compliance_rate": round(
                (compliant_entries / total_entries * 100) if total_entries > 0 else 100,
                1
            ),
            "pending_alerts": pending_alerts,
        },
        "recent_entries": recent_entries,
        "active_alerts": active_alerts,
    }


# ==================== Safety Officer Dashboard ====================

@router.get("/safety-officer")
async def get_safety_officer_dashboard(
    current_user: dict = Depends(get_safety_officer_or_above)
):
    """
    Dashboard for Safety Officer:
    - Compliance analytics (daily/weekly/monthly)
    - Trend detection
    - Worker risk scores
    - Zone risk analysis
    """
    db = get_database()

    mine_id = current_user.get("mine_id")
    if not mine_id:
        return {"error": "No mine assigned to user"}

    mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
    mine_name = mine["name"] if mine else "Unknown"

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)

    # Compliance rates
    async def get_compliance_rate(start_date):
        total = await db.gate_entries.count_documents({
            "mine_id": ObjectId(mine_id),
            "timestamp": {"$gte": start_date},
            "entry_type": "entry"
        })
        violations = await db.gate_entries.count_documents({
            "mine_id": ObjectId(mine_id),
            "timestamp": {"$gte": start_date},
            "entry_type": "entry",
            "violations": {"$ne": []}
        })
        return round(((total - violations) / total * 100) if total > 0 else 100, 1)

    compliance_today = await get_compliance_rate(today)
    compliance_week = await get_compliance_rate(week_ago)
    compliance_month = await get_compliance_rate(month_ago)

    # Violation counts
    violations_today = await db.gate_entries.count_documents({
        "mine_id": ObjectId(mine_id),
        "timestamp": {"$gte": today},
        "violations": {"$ne": []}
    })

    violations_week = await db.gate_entries.count_documents({
        "mine_id": ObjectId(mine_id),
        "timestamp": {"$gte": week_ago},
        "violations": {"$ne": []}
    })

    # Violation trends (by type)
    violation_pipeline = [
        {"$match": {
            "mine_id": ObjectId(mine_id),
            "timestamp": {"$gte": week_ago},
            "violations": {"$ne": []}
        }},
        {"$unwind": "$violations"},
        {"$group": {"_id": "$violations", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]

    violation_trends = {}
    async for v in db.gate_entries.aggregate(violation_pipeline):
        violation_trends[v["_id"]] = v["count"]

    # High risk workers (low compliance scores)
    high_risk_cursor = db.workers.find({
        "mine_id": ObjectId(mine_id),
        "compliance_score": {"$lt": 70},
        "is_active": True
    }).sort("compliance_score", 1).limit(10)

    high_risk_workers = []
    async for worker in high_risk_cursor:
        high_risk_workers.append({
            "id": str(worker["_id"]),
            "employee_id": worker["employee_id"],
            "name": worker["name"],
            "compliance_score": worker.get("compliance_score", 100),
            "total_violations": worker.get("total_violations", 0),
        })

    # Zone risk analysis
    zones_cursor = db.zones.find({"mine_id": ObjectId(mine_id)})
    zone_risk_scores = []

    async for zone in zones_cursor:
        # Count violations in this zone (based on workers in zone)
        zone_workers = await db.workers.find({"zone_id": zone["_id"]}).to_list(length=None)
        zone_worker_ids = [str(w["_id"]) for w in zone_workers]

        zone_violations = await db.gate_entries.count_documents({
            "worker_id": {"$in": zone_worker_ids},
            "timestamp": {"$gte": week_ago},
            "violations": {"$ne": []}
        })

        zone_risk_scores.append({
            "zone_id": str(zone["_id"]),
            "zone_name": zone["name"],
            "risk_level": zone.get("risk_level", "normal"),
            "violations_this_week": zone_violations,
            "worker_count": len(zone_workers),
        })

    # Recent alerts
    alerts_cursor = db.alerts.find({
        "mine_id": ObjectId(mine_id)
    }).sort("created_at", -1).limit(10)

    recent_alerts = []
    async for alert in alerts_cursor:
        recent_alerts.append({
            "id": str(alert["_id"]),
            "alert_type": alert["alert_type"],
            "severity": alert["severity"],
            "status": alert["status"],
            "message": alert["message"],
            "created_at": alert["created_at"].isoformat(),
        })

    # ML-based at-risk workers (predictive)
    at_risk_predictions = await db.predictions.find({
        "risk_category": {"$in": ["high", "critical"]},
        "expires_at": {"$gt": datetime.utcnow()}
    }).sort("overall_risk_score", -1).limit(10).to_list(length=10)

    at_risk_workers_ml = []
    for pred in at_risk_predictions:
        worker = await db.workers.find_one({"_id": ObjectId(pred["worker_id"])})
        if worker and worker.get("mine_id") == ObjectId(mine_id):
            risk_factors = pred.get("risk_factors", [])
            at_risk_workers_ml.append({
                "worker_id": pred["worker_id"],
                "employee_id": worker.get("employee_id", ""),
                "name": worker.get("name", ""),
                "risk_score": round(pred["overall_risk_score"], 1),
                "risk_category": pred["risk_category"],
                "predicted_violations": pred.get("predicted_violations_count", 0),
                "attendance_rate": round(pred.get("attendance_rate_30d", 100), 1),
                "main_issue": risk_factors[0]["description"] if risk_factors else "Multiple factors",
                "requires_intervention": pred.get("requires_intervention", False)
            })

    # Risk distribution summary
    risk_distribution = {}
    for category in ["low", "medium", "high", "critical"]:
        count = await db.predictions.count_documents({
            "risk_category": category,
            "expires_at": {"$gt": datetime.utcnow()}
        })
        risk_distribution[category] = count

    return {
        "mine_id": mine_id,
        "mine_name": mine_name,
        "compliance_rates": {
            "today": compliance_today,
            "this_week": compliance_week,
            "this_month": compliance_month,
        },
        "violations": {
            "today": violations_today,
            "this_week": violations_week,
        },
        "violation_trends": violation_trends,
        "high_risk_workers": high_risk_workers,
        "at_risk_workers_predictive": at_risk_workers_ml,  # NEW: ML-based predictions
        "risk_distribution": risk_distribution,  # NEW: Risk category distribution
        "zone_risk_analysis": zone_risk_scores,
        "recent_alerts": recent_alerts,
    }


# ==================== Manager Dashboard ====================

@router.get("/manager")
async def get_manager_dashboard(
    current_user: dict = Depends(get_manager_or_above)
):
    """
    Dashboard for Manager:
    - Mine overview
    - Shift performance comparison
    - Worker compliance rankings
    - Escalations and approvals
    """
    db = get_database()

    mine_id = current_user.get("mine_id")
    if not mine_id:
        return {"error": "No mine assigned to user"}

    mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
    mine_name = mine["name"] if mine else "Unknown"

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today - timedelta(days=7)

    # Total workers
    total_workers = await db.workers.count_documents({
        "mine_id": ObjectId(mine_id),
        "is_active": True
    })

    # Active workers today
    active_today_pipeline = [
        {"$match": {
            "mine_id": ObjectId(mine_id),
            "timestamp": {"$gte": today},
            "entry_type": "entry"
        }},
        {"$group": {"_id": "$worker_id"}}
    ]
    active_cursor = db.gate_entries.aggregate(active_today_pipeline)
    active_today = len([x async for x in active_cursor])

    # Overall compliance rate
    total_entries = await db.gate_entries.count_documents({
        "mine_id": ObjectId(mine_id),
        "timestamp": {"$gte": week_ago},
        "entry_type": "entry"
    })
    violations = await db.gate_entries.count_documents({
        "mine_id": ObjectId(mine_id),
        "timestamp": {"$gte": week_ago},
        "violations": {"$ne": []}
    })
    compliance_rate = round(((total_entries - violations) / total_entries * 100) if total_entries > 0 else 100, 1)

    # Shift performance comparison
    shift_performance = {}
    for shift in ShiftType:
        shift_entries = await db.gate_entries.count_documents({
            "mine_id": ObjectId(mine_id),
            "timestamp": {"$gte": week_ago},
            "shift": shift.value,
            "entry_type": "entry"
        })
        shift_violations = await db.gate_entries.count_documents({
            "mine_id": ObjectId(mine_id),
            "timestamp": {"$gte": week_ago},
            "shift": shift.value,
            "violations": {"$ne": []}
        })
        shift_performance[shift.value] = round(
            ((shift_entries - shift_violations) / shift_entries * 100) if shift_entries > 0 else 100,
            1
        )

    # Pending escalations (critical alerts)
    pending_escalations = await db.alerts.count_documents({
        "mine_id": ObjectId(mine_id),
        "status": {"$in": ["active", "acknowledged"]},
        "severity": {"$in": ["high", "critical"]}
    })

    # Top compliant workers
    top_workers_cursor = db.workers.find({
        "mine_id": ObjectId(mine_id),
        "is_active": True
    }).sort("compliance_score", -1).limit(5)

    top_workers = []
    async for worker in top_workers_cursor:
        top_workers.append({
            "id": str(worker["_id"]),
            "employee_id": worker["employee_id"],
            "name": worker["name"],
            "compliance_score": worker.get("compliance_score", 100),
            "badges": worker.get("badges", []),
        })

    # Entry delay analysis (placeholder - would need more detailed tracking)
    avg_delay_minutes = 0  # Would calculate from actual entry time vs shift start

    # ML-based at-risk workers
    at_risk_predictions = await db.predictions.find({
        "risk_category": {"$in": ["high", "critical"]},
        "expires_at": {"$gt": datetime.utcnow()}
    }).sort("overall_risk_score", -1).limit(10).to_list(length=10)

    at_risk_workers = []
    for pred in at_risk_predictions:
        worker = await db.workers.find_one({"_id": ObjectId(pred["worker_id"])})
        if worker and worker.get("mine_id") == ObjectId(mine_id):
            risk_factors = pred.get("risk_factors", [])
            at_risk_workers.append({
                "worker_id": pred["worker_id"],
                "employee_id": worker.get("employee_id", ""),
                "name": worker.get("name", ""),
                "risk_score": round(pred["overall_risk_score"], 1),
                "risk_category": pred["risk_category"],
                "main_issue": risk_factors[0]["description"] if risk_factors else "Multiple factors",
                "requires_intervention": pred.get("requires_intervention", False)
            })

    # Predictive insights summary
    total_at_risk = len(at_risk_workers)
    critical_risk_count = len([w for w in at_risk_workers if w["risk_category"] == "critical"])
    intervention_needed = len([w for w in at_risk_workers if w["requires_intervention"]])

    return {
        "mine_id": mine_id,
        "mine_name": mine_name,
        "overview": {
            "total_workers": total_workers,
            "active_workers_today": active_today,
            "compliance_rate": compliance_rate,
            "pending_escalations": pending_escalations,
            "avg_entry_delay_minutes": avg_delay_minutes,
        },
        "shift_performance": shift_performance,
        "top_compliant_workers": top_workers,
        "at_risk_workers": at_risk_workers,  # NEW: ML-based predictions
        "predictive_insights": {  # NEW: Summary of prediction data
            "total_at_risk": total_at_risk,
            "critical_risk_count": critical_risk_count,
            "intervention_needed": intervention_needed
        }
    }


# ==================== Area Safety Officer Dashboard ====================

@router.get("/area-safety-officer")
async def get_area_safety_officer_dashboard(
    current_user: dict = Depends(get_area_safety_officer_or_above)
):
    """
    Dashboard for Area Safety Officer:
    - Multi-mine comparison
    - Risk heatmap data
    - Compliance leaderboard
    - Critical alerts across mines
    """
    db = get_database()

    mine_ids = current_user.get("mine_ids", [])
    if not mine_ids:
        return {"error": "No mines assigned to user"}

    week_ago = datetime.utcnow() - timedelta(days=7)

    # Mines overview
    mines_overview = []
    total_compliance = 0
    total_mines_with_data = 0

    for mine_id in mine_ids:
        mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
        if not mine:
            continue

        # Get compliance rate for this mine
        entries = await db.gate_entries.count_documents({
            "mine_id": ObjectId(mine_id),
            "timestamp": {"$gte": week_ago},
            "entry_type": "entry"
        })
        violations = await db.gate_entries.count_documents({
            "mine_id": ObjectId(mine_id),
            "timestamp": {"$gte": week_ago},
            "violations": {"$ne": []}
        })
        compliance = round(((entries - violations) / entries * 100) if entries > 0 else 100, 1)

        if entries > 0:
            total_compliance += compliance
            total_mines_with_data += 1

        worker_count = await db.workers.count_documents({
            "mine_id": ObjectId(mine_id),
            "is_active": True
        })

        active_alerts = await db.alerts.count_documents({
            "mine_id": ObjectId(mine_id),
            "status": {"$in": ["active", "acknowledged"]}
        })

        mines_overview.append({
            "mine_id": mine_id,
            "mine_name": mine["name"],
            "location": mine.get("location"),
            "compliance_rate": compliance,
            "worker_count": worker_count,
            "violations_this_week": violations,
            "active_alerts": active_alerts,
        })

    # Sort by compliance rate (worst first for attention)
    mines_overview.sort(key=lambda x: x["compliance_rate"])

    overall_compliance = round(total_compliance / total_mines_with_data, 1) if total_mines_with_data > 0 else 100

    # Critical alerts across all mines
    critical_alerts_cursor = db.alerts.find({
        "mine_id": {"$in": [ObjectId(mid) for mid in mine_ids]},
        "status": {"$in": ["active", "acknowledged"]},
        "severity": {"$in": ["high", "critical"]}
    }).sort("created_at", -1).limit(10)

    critical_alerts = []
    async for alert in critical_alerts_cursor:
        mine = await db.mines.find_one({"_id": alert["mine_id"]})
        critical_alerts.append({
            "id": str(alert["_id"]),
            "mine_id": str(alert["mine_id"]),
            "mine_name": mine["name"] if mine else "Unknown",
            "alert_type": alert["alert_type"],
            "severity": alert["severity"],
            "message": alert["message"],
            "created_at": alert["created_at"].isoformat(),
        })

    # Risk heatmap data (zones with high violations)
    risk_heatmap = []
    for mine_id in mine_ids:
        zones_cursor = db.zones.find({"mine_id": ObjectId(mine_id)})
        async for zone in zones_cursor:
            zone_workers = await db.workers.find({"zone_id": zone["_id"]}).to_list(length=None)
            zone_worker_ids = [str(w["_id"]) for w in zone_workers]

            zone_violations = await db.gate_entries.count_documents({
                "worker_id": {"$in": zone_worker_ids},
                "timestamp": {"$gte": week_ago},
                "violations": {"$ne": []}
            })

            if zone_violations > 0:
                mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
                risk_heatmap.append({
                    "mine_id": mine_id,
                    "mine_name": mine["name"] if mine else "Unknown",
                    "zone_id": str(zone["_id"]),
                    "zone_name": zone["name"],
                    "violations": zone_violations,
                    "risk_level": "high" if zone_violations > 10 else "medium" if zone_violations > 5 else "low",
                })

    risk_heatmap.sort(key=lambda x: x["violations"], reverse=True)

    return {
        "overall_compliance_rate": overall_compliance,
        "total_mines": len(mine_ids),
        "mines_overview": mines_overview,
        "critical_alerts": critical_alerts,
        "risk_heatmap": risk_heatmap[:10],  # Top 10 risky zones
    }


# ==================== General Manager Dashboard ====================

@router.get("/general-manager")
async def get_general_manager_dashboard(
    current_user: dict = Depends(get_general_manager)
):
    """
    Dashboard for General Manager:
    - Organization-wide KPIs
    - Regulatory compliance status
    - Strategic alerts
    - Cost/productivity analysis
    """
    db = get_database()

    week_ago = datetime.utcnow() - timedelta(days=7)
    month_ago = datetime.utcnow() - timedelta(days=30)

    # Get all mines
    mines_cursor = db.mines.find({"is_active": True})
    all_mines = [m async for m in mines_cursor]
    total_mines = len(all_mines)

    # Total workers
    total_workers = await db.workers.count_documents({"is_active": True})

    # Organization-wide compliance
    total_entries = await db.gate_entries.count_documents({
        "timestamp": {"$gte": week_ago},
        "entry_type": "entry"
    })
    total_violations = await db.gate_entries.count_documents({
        "timestamp": {"$gte": week_ago},
        "violations": {"$ne": []}
    })
    org_compliance = round(((total_entries - total_violations) / total_entries * 100) if total_entries > 0 else 100, 1)

    # KPI summary
    monthly_entries = await db.gate_entries.count_documents({
        "timestamp": {"$gte": month_ago},
        "entry_type": "entry"
    })
    monthly_violations = await db.gate_entries.count_documents({
        "timestamp": {"$gte": month_ago},
        "violations": {"$ne": []}
    })

    # Mine-wise performance
    mine_performance = []
    for mine in all_mines:
        mine_entries = await db.gate_entries.count_documents({
            "mine_id": mine["_id"],
            "timestamp": {"$gte": week_ago},
            "entry_type": "entry"
        })
        mine_violations = await db.gate_entries.count_documents({
            "mine_id": mine["_id"],
            "timestamp": {"$gte": week_ago},
            "violations": {"$ne": []}
        })
        compliance = round(((mine_entries - mine_violations) / mine_entries * 100) if mine_entries > 0 else 100, 1)

        mine_performance.append({
            "mine_id": str(mine["_id"]),
            "mine_name": mine["name"],
            "compliance_rate": compliance,
            "total_entries": mine_entries,
            "violations": mine_violations,
        })

    mine_performance.sort(key=lambda x: x["compliance_rate"])

    # Strategic alerts (critical incidents)
    strategic_alerts_cursor = db.alerts.find({
        "severity": "critical",
        "status": {"$in": ["active", "acknowledged"]}
    }).sort("created_at", -1).limit(5)

    strategic_alerts = []
    async for alert in strategic_alerts_cursor:
        mine = await db.mines.find_one({"_id": alert["mine_id"]})
        strategic_alerts.append({
            "id": str(alert["_id"]),
            "mine_name": mine["name"] if mine else "Unknown",
            "alert_type": alert["alert_type"],
            "message": alert["message"],
            "created_at": alert["created_at"].isoformat(),
        })

    # Cost savings estimate (placeholder calculation)
    # Assume each prevented violation saves $500 in potential incidents
    violations_prevented = total_entries - total_violations
    estimated_savings = violations_prevented * 500

    return {
        "organization_overview": {
            "total_mines": total_mines,
            "total_workers": total_workers,
            "compliance_rate": org_compliance,
        },
        "kpi_summary": {
            "monthly_entries": monthly_entries,
            "monthly_violations": monthly_violations,
            "monthly_compliance_rate": round(
                ((monthly_entries - monthly_violations) / monthly_entries * 100) if monthly_entries > 0 else 100,
                1
            ),
        },
        "mine_performance": mine_performance,
        "strategic_alerts": strategic_alerts,
        "financial_insights": {
            "estimated_cost_savings": estimated_savings,
            "violations_prevented_this_week": violations_prevented,
        },
        "regulatory_status": {
            "compliance_threshold": 95.0,
            "current_compliance": org_compliance,
            "status": "compliant" if org_compliance >= 95 else "needs_attention",
        },
    }


# ==================== Worker Dashboard ====================

@router.get("/worker")
async def get_worker_dashboard(
    current_user: dict = Depends(get_current_user)
):
    """
    Dashboard for Worker (PWA):
    - Personal compliance score
    - Recent violations
    - Badges earned
    - Shift info
    """
    db = get_database()

    # Check if worker
    if current_user.get("user_type") != "worker":
        raise HTTPException(status_code=403, detail="Worker access required")

    worker_id = current_user.get("worker_id")
    if not worker_id:
        raise HTTPException(status_code=400, detail="Worker ID not found in token")

    worker = await db.workers.find_one({"_id": ObjectId(worker_id)})
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Get mine and zone names
    mine_name = None
    zone_name = None
    if worker.get("mine_id"):
        mine = await db.mines.find_one({"_id": worker["mine_id"]})
        if mine:
            mine_name = mine["name"]
    if worker.get("zone_id"):
        zone = await db.zones.find_one({"_id": worker["zone_id"]})
        if zone:
            zone_name = zone["name"]

    # Get recent violations
    week_ago = datetime.utcnow() - timedelta(days=7)
    violations_cursor = db.gate_entries.find({
        "worker_id": worker_id,
        "violations": {"$ne": []},
        "timestamp": {"$gte": week_ago}
    }).sort("timestamp", -1).limit(5)

    recent_violations = []
    async for entry in violations_cursor:
        recent_violations.append({
            "date": entry["timestamp"].strftime("%Y-%m-%d"),
            "time": entry["timestamp"].strftime("%H:%M"),
            "violations": entry["violations"],
        })

    # Get total entries
    total_entries = await db.gate_entries.count_documents({
        "worker_id": worker_id,
        "entry_type": "entry"
    })

    # Calculate streak (consecutive days without violations)
    streak_days = 0
    check_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    while True:
        day_start = check_date
        day_end = check_date + timedelta(days=1)

        day_violations = await db.gate_entries.count_documents({
            "worker_id": worker_id,
            "timestamp": {"$gte": day_start, "$lt": day_end},
            "violations": {"$ne": []}
        })

        day_entries = await db.gate_entries.count_documents({
            "worker_id": worker_id,
            "timestamp": {"$gte": day_start, "$lt": day_end}
        })

        if day_entries == 0:
            # No entries this day, skip
            check_date -= timedelta(days=1)
            if streak_days > 30:  # Max check 30 days back
                break
            continue

        if day_violations > 0:
            break

        streak_days += 1
        check_date -= timedelta(days=1)

        if streak_days > 365:  # Max streak
            break

    # Get unacknowledged warnings
    warnings_cursor = db.warnings.find({
        "worker_id": worker_id,
        "acknowledged": False
    }).sort("issued_at", -1)

    notifications = []
    async for warning in warnings_cursor:
        notifications.append({
            "type": "warning",
            "id": str(warning["_id"]),
            "message": f"{warning['warning_type']}: {warning['description']}",
            "date": warning["issued_at"].isoformat(),
            "severity": warning["severity"],
        })

    # Get shift info
    current_shift = get_current_shift()
    worker_shift = ShiftType(worker.get("assigned_shift", "day"))
    shift_info = SHIFT_DEFINITIONS[worker_shift]

    return {
        "worker": {
            "id": worker_id,
            "employee_id": worker["employee_id"],
            "name": worker["name"],
            "department": worker.get("department"),
            "mine_name": mine_name,
            "zone_name": zone_name,
        },
        "compliance": {
            "score": worker.get("compliance_score", 100),
            "total_violations": worker.get("total_violations", 0),
            "current_streak_days": streak_days,
        },
        "badges": worker.get("badges", []),
        "statistics": {
            "total_entries": total_entries,
        },
        "recent_violations": recent_violations,
        "shift_info": {
            "assigned_shift": worker_shift.value,
            "shift_name": shift_info.name,
            "start_time": shift_info.start_time,
            "end_time": shift_info.end_time,
            "is_current_shift": worker_shift == current_shift,
        },
        "notifications": notifications,
    }
