"""
Data aggregation service for report generation.

Provides role-specific data queries from MongoDB collections.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from bson import ObjectId


class DataAggregator:
    """
    Service for aggregating data from various collections for reports.
    """

    def __init__(self, db):
        """
        Initialize with database connection.

        Args:
            db: MongoDB database instance
        """
        self.db = db

    # ==================== Shift Incharge Data ====================

    async def get_shift_summary(
        self,
        mine_id: str,
        shift: str,
        date: datetime,
        gate_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get shift summary data for shift incharge report.

        Args:
            mine_id: Mine ID
            shift: Shift name (day, afternoon, night)
            date: Date of shift
            gate_id: Optional specific gate

        Returns:
            Shift summary data
        """
        # Calculate shift time range
        shift_ranges = {
            "day": (6, 14),
            "afternoon": (14, 22),
            "night": (22, 6)
        }
        start_hour, end_hour = shift_ranges.get(shift, (6, 14))

        start_time = date.replace(hour=start_hour, minute=0, second=0)
        if shift == "night":
            end_time = (date + timedelta(days=1)).replace(hour=end_hour, minute=0, second=0)
        else:
            end_time = date.replace(hour=end_hour, minute=0, second=0)

        # Build query
        query = {
            "mine_id": ObjectId(mine_id) if mine_id else {"$exists": True},
            "timestamp": {"$gte": start_time, "$lt": end_time}
        }
        if gate_id:
            query["gate_id"] = ObjectId(gate_id)

        # Get gate entries
        entries = await self.db.gate_entries.find(query).to_list(length=None)

        # Calculate metrics
        workers_entered = len([e for e in entries if e.get("entry_type") == "entry"])
        workers_exited = len([e for e in entries if e.get("entry_type") == "exit"])

        compliant_entries = len([e for e in entries if e.get("ppe_compliant", False)])
        total_entries = len(entries)
        compliance_rate = (compliant_entries / total_entries * 100) if total_entries > 0 else 0

        violations = [e for e in entries if e.get("violations")]
        violations_count = sum(len(e.get("violations", [])) for e in entries)

        # Get alerts for this shift
        alert_query = {
            "mine_id": ObjectId(mine_id) if mine_id else {"$exists": True},
            "created_at": {"$gte": start_time, "$lt": end_time}
        }
        alerts = await self.db.alerts.find(alert_query).to_list(length=None)
        alerts_resolved = len([a for a in alerts if a.get("status") == "resolved"])
        alerts_pending = len([a for a in alerts if a.get("status") == "active"])

        # Hourly breakdown
        hourly_breakdown = await self._get_hourly_breakdown(entries, start_time, end_time)

        # Get workers currently inside
        currently_inside = await self._get_workers_inside(mine_id, end_time)

        return {
            "shift_info": {
                "shift": shift,
                "date": date.strftime("%Y-%m-%d"),
                "start_time": start_time.strftime("%H:%M"),
                "end_time": end_time.strftime("%H:%M")
            },
            "workers_entered": workers_entered,
            "workers_exited": workers_exited,
            "currently_inside": currently_inside,
            "compliance_rate": round(compliance_rate, 1),
            "violations_count": violations_count,
            "alerts_resolved": alerts_resolved,
            "alerts_pending": alerts_pending,
            "entry_exit_logs": [self._format_entry(e) for e in entries[:100]],
            "hourly_breakdown": hourly_breakdown
        }

    async def get_entry_exit_log(
        self,
        mine_id: str,
        start_date: datetime,
        end_date: datetime,
        gate_id: Optional[str] = None,
        worker_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get detailed entry/exit log."""
        query = {
            "timestamp": {"$gte": start_date, "$lt": end_date}
        }
        if mine_id:
            query["mine_id"] = ObjectId(mine_id)
        if gate_id:
            query["gate_id"] = ObjectId(gate_id)
        if worker_id:
            query["worker_id"] = ObjectId(worker_id)

        entries = await self.db.gate_entries.find(query).sort("timestamp", -1).to_list(length=1000)
        return [self._format_entry(e) for e in entries]

    # ==================== Safety Officer Data ====================

    async def get_compliance_data(
        self,
        mine_id: str,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "day"
    ) -> Dict[str, Any]:
        """
        Get compliance analytics for safety officer report.

        Args:
            mine_id: Mine ID
            start_date: Start date
            end_date: End date
            group_by: Grouping (day, week, month)

        Returns:
            Compliance analytics data
        """
        query = {"timestamp": {"$gte": start_date, "$lt": end_date}}
        if mine_id:
            query["mine_id"] = ObjectId(mine_id)

        entries = await self.db.gate_entries.find(query).to_list(length=None)

        # Overall compliance
        total = len(entries)
        compliant = len([e for e in entries if e.get("ppe_compliant", False)])
        overall_compliance = (compliant / total * 100) if total > 0 else 0

        # Compliance trend
        compliance_trend = await self._get_compliance_trend(entries, group_by)

        # Violations by type
        violations_by_type = {}
        for entry in entries:
            for violation in entry.get("violations", []):
                # Handle both string violations (e.g., "NO-Helmet") and dict violations
                if isinstance(violation, str):
                    v_type = violation
                elif isinstance(violation, dict):
                    v_type = violation.get("type", "unknown")
                else:
                    v_type = str(violation)
                violations_by_type[v_type] = violations_by_type.get(v_type, 0) + 1

        # Zone analysis
        zone_analysis = await self._get_zone_analysis(mine_id, start_date, end_date)

        # High risk workers
        high_risk_workers = await self.get_high_risk_workers(mine_id, threshold=70)

        return {
            "overall_compliance": round(overall_compliance, 1),
            "compliance_trend": compliance_trend,
            "violations_by_type": violations_by_type,
            "zone_analysis": zone_analysis,
            "high_risk_workers": high_risk_workers[:20]
        }

    async def get_high_risk_workers(
        self,
        mine_id: Optional[str],
        threshold: float = 70
    ) -> List[Dict[str, Any]]:
        """Get workers with compliance score below threshold."""
        query = {"compliance_score": {"$lt": threshold}, "is_active": True}
        if mine_id:
            query["mine_id"] = ObjectId(mine_id)

        workers = await self.db.workers.find(query).sort("compliance_score", 1).to_list(length=50)

        result = []
        for w in workers:
            # Get recent violations
            violations = await self.db.gate_entries.count_documents({
                "worker_id": w["_id"],
                "violations": {"$exists": True, "$ne": []},
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=30)}
            })

            result.append({
                "worker_id": str(w["_id"]),
                "employee_id": w.get("employee_id"),
                "name": w.get("name"),
                "compliance_score": w.get("compliance_score", 0),
                "total_violations": w.get("total_violations", 0),
                "violations_last_30_days": violations,
                "zone": w.get("zone_id")
            })

        return result

    async def get_violation_trends(
        self,
        mine_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get violation trends with month-over-month comparison."""
        # Current period
        current_violations = await self._count_violations(mine_id, start_date, end_date)

        # Previous period (same duration)
        duration = end_date - start_date
        prev_start = start_date - duration
        prev_end = start_date
        prev_violations = await self._count_violations(mine_id, prev_start, prev_end)

        # Repeat offenders
        repeat_offenders = await self._get_repeat_offenders(mine_id, start_date, end_date)

        return {
            "current_period": current_violations,
            "previous_period": prev_violations,
            "change_percentage": self._calculate_change(current_violations["total"], prev_violations["total"]),
            "repeat_offenders": repeat_offenders
        }

    # ==================== Manager Data ====================

    async def get_operations_data(
        self,
        mine_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get operations summary for manager report."""
        # Worker counts
        total_workers = await self.db.workers.count_documents({
            "mine_id": ObjectId(mine_id) if mine_id else {"$exists": True},
            "is_active": True
        })

        # Active workers (entered today)
        active_query = {
            "entry_type": "entry",
            "timestamp": {"$gte": start_date, "$lt": end_date}
        }
        if mine_id:
            active_query["mine_id"] = ObjectId(mine_id)

        active_entries = await self.db.gate_entries.distinct("worker_id", active_query)
        active_workers = len(active_entries)

        # Compliance
        compliance_data = await self.get_compliance_data(mine_id, start_date, end_date)

        # Shift performance
        shift_performance = await self._get_shift_performance(mine_id, start_date, end_date)

        # Worker rankings
        worker_rankings = await self._get_worker_rankings(mine_id)

        # Escalations
        escalations = await self._get_escalations(mine_id, start_date, end_date)

        return {
            "total_workers": total_workers,
            "active_workers": active_workers,
            "compliance_rate": compliance_data["overall_compliance"],
            "shift_performance": shift_performance,
            "worker_rankings": worker_rankings,
            "escalations": escalations,
            "violations_by_type": compliance_data["violations_by_type"]
        }

    async def get_monthly_summary(
        self,
        mine_id: str,
        month: int,
        year: int
    ) -> Dict[str, Any]:
        """Get monthly summary data."""
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)

        operations = await self.get_operations_data(mine_id, start_date, end_date)
        compliance = await self.get_compliance_data(mine_id, start_date, end_date, "week")
        trends = await self.get_violation_trends(mine_id, start_date, end_date)

        return {
            **operations,
            "compliance_trend": compliance["compliance_trend"],
            "violation_trends": trends,
            "month": month,
            "year": year
        }

    # ==================== Area Safety Officer Data ====================

    async def get_mine_comparison(
        self,
        mine_ids: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get comparison data across multiple mines."""
        mines_data = []

        for mine_id in mine_ids:
            mine_info = await self.db.mines.find_one({"_id": ObjectId(mine_id)})
            if not mine_info:
                continue

            compliance = await self.get_compliance_data(mine_id, start_date, end_date)

            # Get worker count
            worker_count = await self.db.workers.count_documents({
                "mine_id": ObjectId(mine_id),
                "is_active": True
            })

            mines_data.append({
                "mine_id": mine_id,
                "mine_name": mine_info.get("name"),
                "location": mine_info.get("location"),
                "worker_count": worker_count,
                "compliance_rate": compliance["overall_compliance"],
                "violations_by_type": compliance["violations_by_type"],
                "high_risk_workers": len(compliance["high_risk_workers"])
            })

        # Sort by compliance rate
        mines_data.sort(key=lambda x: x["compliance_rate"], reverse=True)

        return {
            "mines": mines_data,
            "best_performing": mines_data[0] if mines_data else None,
            "needs_attention": [m for m in mines_data if m["compliance_rate"] < 80]
        }

    async def get_risk_heatmap(
        self,
        mine_ids: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get zone-level risk data for heatmap visualization."""
        heatmap_data = []

        for mine_id in mine_ids:
            zones = await self.db.zones.find({"mine_id": ObjectId(mine_id)}).to_list(length=None)

            for zone in zones:
                zone_id = str(zone["_id"])

                # Count violations in zone
                violation_count = await self.db.gate_entries.count_documents({
                    "zone_id": zone["_id"],
                    "violations": {"$exists": True, "$ne": []},
                    "timestamp": {"$gte": start_date, "$lt": end_date}
                })

                # Count total entries
                total_entries = await self.db.gate_entries.count_documents({
                    "zone_id": zone["_id"],
                    "timestamp": {"$gte": start_date, "$lt": end_date}
                })

                # Calculate risk level
                violation_rate = (violation_count / total_entries * 100) if total_entries > 0 else 0
                risk_level = "low" if violation_rate < 10 else "medium" if violation_rate < 25 else "high"

                heatmap_data.append({
                    "mine_id": mine_id,
                    "zone_id": zone_id,
                    "zone_name": zone.get("name"),
                    "violation_count": violation_count,
                    "total_entries": total_entries,
                    "violation_rate": round(violation_rate, 1),
                    "risk_level": risk_level
                })

        return heatmap_data

    async def get_critical_incidents(
        self,
        mine_ids: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get critical incidents across mines."""
        query = {
            "severity": {"$in": ["high", "critical"]},
            "created_at": {"$gte": start_date, "$lt": end_date}
        }
        if mine_ids:
            query["mine_id"] = {"$in": [ObjectId(m) for m in mine_ids]}

        alerts = await self.db.alerts.find(query).sort("created_at", -1).to_list(length=100)

        incidents = []
        for alert in alerts:
            mine_info = await self.db.mines.find_one({"_id": alert.get("mine_id")})
            incidents.append({
                "id": str(alert["_id"]),
                "type": alert.get("alert_type"),
                "severity": alert.get("severity"),
                "message": alert.get("message"),
                "mine_name": mine_info.get("name") if mine_info else "Unknown",
                "status": alert.get("status"),
                "created_at": alert.get("created_at")
            })

        return incidents

    # ==================== General Manager Data ====================

    async def get_executive_summary(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get executive summary data for general manager."""
        # Get all mines
        mines = await self.db.mines.find({"is_active": True}).to_list(length=None)
        mine_ids = [str(m["_id"]) for m in mines]

        # Organization-wide metrics
        total_workers = await self.db.workers.count_documents({"is_active": True})

        # Aggregate compliance across all mines
        all_entries = await self.db.gate_entries.find({
            "timestamp": {"$gte": start_date, "$lt": end_date}
        }).to_list(length=None)

        total = len(all_entries)
        compliant = len([e for e in all_entries if e.get("ppe_compliant", False)])
        org_compliance = (compliant / total * 100) if total > 0 else 0

        # KPIs
        kpis = {
            "total_mines": len(mines),
            "total_workers": total_workers,
            "total_entries": total,
            "overall_compliance": round(org_compliance, 1),
            "total_violations": sum(len(e.get("violations", [])) for e in all_entries)
        }

        # Mine performance
        mine_comparison = await self.get_mine_comparison(mine_ids, start_date, end_date)

        # Regulatory status
        regulatory_threshold = 85  # Example threshold
        regulatory_status = {
            "threshold": regulatory_threshold,
            "current_compliance": org_compliance,
            "status": "compliant" if org_compliance >= regulatory_threshold else "needs_attention"
        }

        # Strategic alerts
        critical_incidents = await self.get_critical_incidents(mine_ids, start_date, end_date)

        return {
            "kpis": kpis,
            "mine_performance": mine_comparison["mines"],
            "regulatory_status": regulatory_status,
            "critical_incidents": critical_incidents[:10],
            "best_performing_mine": mine_comparison.get("best_performing"),
            "mines_needing_attention": mine_comparison.get("needs_attention", [])
        }

    async def get_financial_impact(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calculate financial impact metrics."""
        # Estimated costs (these would be configurable in real system)
        COST_PER_INCIDENT = 5000  # Average cost per safety incident
        COST_PER_VIOLATION = 100  # Administrative cost per violation

        # Count prevented incidents (violations caught at gate)
        violations_caught = await self.db.gate_entries.count_documents({
            "violations": {"$exists": True, "$ne": []},
            "status": "denied",  # Entry was denied due to violations
            "timestamp": {"$gte": start_date, "$lt": end_date}
        })

        # Total violations
        all_violations = await self.db.gate_entries.count_documents({
            "violations": {"$exists": True, "$ne": []},
            "timestamp": {"$gte": start_date, "$lt": end_date}
        })

        # Estimate savings
        incidents_prevented = violations_caught * 0.1  # Assume 10% could have been incidents
        estimated_savings = incidents_prevented * COST_PER_INCIDENT

        return {
            "violations_caught": violations_caught,
            "total_violations": all_violations,
            "incidents_prevented": int(incidents_prevented),
            "estimated_cost_savings": round(estimated_savings, 2),
            "violation_processing_cost": all_violations * COST_PER_VIOLATION
        }

    # ==================== Worker Data ====================

    async def get_worker_compliance_card(
        self,
        worker_id: str
    ) -> Dict[str, Any]:
        """Get compliance card data for a worker."""
        worker = await self.db.workers.find_one({"_id": ObjectId(worker_id)})
        if not worker:
            return {}

        # Get recent violations (last 30 days)
        recent_violations = await self.db.gate_entries.find({
            "worker_id": ObjectId(worker_id),
            "violations": {"$exists": True, "$ne": []},
            "timestamp": {"$gte": datetime.utcnow() - timedelta(days=30)}
        }).sort("timestamp", -1).to_list(length=10)

        # Calculate streak (consecutive days without violations)
        streak = await self._calculate_compliance_streak(worker_id)

        # Get attendance summary
        attendance = await self._get_worker_attendance(worker_id)

        return {
            "worker_info": {
                "id": str(worker["_id"]),
                "employee_id": worker.get("employee_id"),
                "name": worker.get("name"),
                "department": worker.get("department"),
                "assigned_shift": worker.get("assigned_shift")
            },
            "compliance_score": worker.get("compliance_score", 0),
            "total_violations": worker.get("total_violations", 0),
            "streak_days": streak,
            "badges": worker.get("badges", []),
            "recent_violations": [self._format_entry(v) for v in recent_violations],
            "attendance_summary": attendance
        }

    # ==================== Helper Methods ====================

    async def _get_hourly_breakdown(
        self,
        entries: List[Dict],
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Group entries by hour."""
        from collections import defaultdict
        hourly = defaultdict(lambda: {"entries": 0, "violations": 0, "compliance_rate": 0})

        for entry in entries:
            hour = entry.get("timestamp").hour
            hourly[hour]["entries"] += 1
            if entry.get("violations"):
                hourly[hour]["violations"] += len(entry.get("violations", []))

        result = []
        current = start_time
        while current < end_time:
            hour = current.hour
            data = hourly.get(hour, {"entries": 0, "violations": 0})
            entries_count = data["entries"]
            violations_count = data["violations"]
            compliance = ((entries_count - violations_count) / entries_count * 100) if entries_count > 0 else 100

            result.append({
                "hour": f"{hour:02d}:00",
                "entries": entries_count,
                "violations": violations_count,
                "compliance_rate": round(compliance, 1)
            })
            current += timedelta(hours=1)

        return result

    async def _get_workers_inside(self, mine_id: str, as_of: datetime) -> int:
        """Get count of workers currently inside the mine."""
        # This is a simplified version - real implementation would track entry/exit pairs
        pipeline = [
            {"$match": {
                "mine_id": ObjectId(mine_id) if mine_id else {"$exists": True},
                "timestamp": {"$lt": as_of}
            }},
            {"$sort": {"timestamp": -1}},
            {"$group": {
                "_id": "$worker_id",
                "last_entry_type": {"$first": "$entry_type"}
            }},
            {"$match": {"last_entry_type": "entry"}}
        ]

        result = await self.db.gate_entries.aggregate(pipeline).to_list(length=None)
        return len(result)

    async def _get_compliance_trend(
        self,
        entries: List[Dict],
        group_by: str
    ) -> List[Dict[str, Any]]:
        """Calculate compliance trend over time."""
        from collections import defaultdict

        grouped = defaultdict(lambda: {"total": 0, "compliant": 0})

        for entry in entries:
            ts = entry.get("timestamp")
            if group_by == "day":
                key = ts.strftime("%Y-%m-%d")
            elif group_by == "week":
                key = ts.strftime("%Y-W%W")
            else:
                key = ts.strftime("%Y-%m")

            grouped[key]["total"] += 1
            if entry.get("ppe_compliant", False):
                grouped[key]["compliant"] += 1

        result = []
        for date_key in sorted(grouped.keys()):
            data = grouped[date_key]
            compliance = (data["compliant"] / data["total"] * 100) if data["total"] > 0 else 0
            result.append({
                "date": date_key,
                "total_entries": data["total"],
                "compliant_entries": data["compliant"],
                "compliance_rate": round(compliance, 1)
            })

        return result

    async def _get_zone_analysis(
        self,
        mine_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get zone-level violation analysis."""
        query = {"timestamp": {"$gte": start_date, "$lt": end_date}}
        if mine_id:
            query["mine_id"] = ObjectId(mine_id)

        pipeline = [
            {"$match": query},
            {"$group": {
                "_id": "$zone_id",
                "total_entries": {"$sum": 1},
                "violations": {"$sum": {"$cond": [{"$gt": [{"$size": {"$ifNull": ["$violations", []]}}, 0]}, 1, 0]}}
            }}
        ]

        results = await self.db.gate_entries.aggregate(pipeline).to_list(length=None)

        zone_analysis = []
        for r in results:
            zone = await self.db.zones.find_one({"_id": r["_id"]})
            violation_rate = (r["violations"] / r["total_entries"] * 100) if r["total_entries"] > 0 else 0

            zone_analysis.append({
                "zone_id": str(r["_id"]) if r["_id"] else "unknown",
                "zone_name": zone.get("name") if zone else "Unknown Zone",
                "total_entries": r["total_entries"],
                "violations": r["violations"],
                "violation_rate": round(violation_rate, 1),
                "risk_level": "high" if violation_rate > 20 else "medium" if violation_rate > 10 else "low"
            })

        return sorted(zone_analysis, key=lambda x: x["violation_rate"], reverse=True)

    async def _count_violations(
        self,
        mine_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Count violations in a period."""
        query = {
            "violations": {"$exists": True, "$ne": []},
            "timestamp": {"$gte": start_date, "$lt": end_date}
        }
        if mine_id:
            query["mine_id"] = ObjectId(mine_id)

        entries = await self.db.gate_entries.find(query).to_list(length=None)

        by_type = {}
        for entry in entries:
            for v in entry.get("violations", []):
                # Handle both string violations and dict violations
                if isinstance(v, str):
                    v_type = v
                elif isinstance(v, dict):
                    v_type = v.get("type", "unknown")
                else:
                    v_type = str(v)
                by_type[v_type] = by_type.get(v_type, 0) + 1

        return {
            "total": len(entries),
            "by_type": by_type,
            "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        }

    async def _get_repeat_offenders(
        self,
        mine_id: str,
        start_date: datetime,
        end_date: datetime,
        threshold: int = 3
    ) -> List[Dict[str, Any]]:
        """Get workers with multiple violations in period."""
        query = {
            "violations": {"$exists": True, "$ne": []},
            "timestamp": {"$gte": start_date, "$lt": end_date}
        }
        if mine_id:
            query["mine_id"] = ObjectId(mine_id)

        pipeline = [
            {"$match": query},
            {"$group": {
                "_id": "$worker_id",
                "violation_count": {"$sum": 1}
            }},
            {"$match": {"violation_count": {"$gte": threshold}}},
            {"$sort": {"violation_count": -1}},
            {"$limit": 20}
        ]

        results = await self.db.gate_entries.aggregate(pipeline).to_list(length=None)

        offenders = []
        for r in results:
            worker = await self.db.workers.find_one({"_id": r["_id"]})
            if worker:
                offenders.append({
                    "worker_id": str(r["_id"]),
                    "employee_id": worker.get("employee_id"),
                    "name": worker.get("name"),
                    "violation_count": r["violation_count"]
                })

        return offenders

    async def _get_shift_performance(
        self,
        mine_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get performance comparison by shift."""
        shifts = ["day", "afternoon", "night"]
        performance = {}

        for shift in shifts:
            query = {
                "shift": shift,
                "timestamp": {"$gte": start_date, "$lt": end_date}
            }
            if mine_id:
                query["mine_id"] = ObjectId(mine_id)

            entries = await self.db.gate_entries.find(query).to_list(length=None)
            total = len(entries)
            compliant = len([e for e in entries if e.get("ppe_compliant", False)])

            performance[shift] = {
                "total_entries": total,
                "compliant_entries": compliant,
                "compliance_rate": round((compliant / total * 100) if total > 0 else 0, 1)
            }

        return performance

    async def _get_worker_rankings(
        self,
        mine_id: str,
        limit: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get top and bottom performing workers."""
        query = {"is_active": True}
        if mine_id:
            query["mine_id"] = ObjectId(mine_id)

        # Top performers
        top = await self.db.workers.find(query).sort("compliance_score", -1).limit(limit).to_list(length=limit)

        # Bottom performers
        bottom = await self.db.workers.find(query).sort("compliance_score", 1).limit(limit).to_list(length=limit)

        return {
            "top_performers": [
                {
                    "employee_id": w.get("employee_id"),
                    "name": w.get("name"),
                    "compliance_score": w.get("compliance_score", 0)
                }
                for w in top
            ],
            "needs_improvement": [
                {
                    "employee_id": w.get("employee_id"),
                    "name": w.get("name"),
                    "compliance_score": w.get("compliance_score", 0)
                }
                for w in bottom
            ]
        }

    async def _get_escalations(
        self,
        mine_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get escalation alerts."""
        query = {
            "alert_type": {"$in": ["escalation", "worker_risk_prediction"]},
            "created_at": {"$gte": start_date, "$lt": end_date}
        }
        if mine_id:
            query["mine_id"] = ObjectId(mine_id)

        alerts = await self.db.alerts.find(query).sort("created_at", -1).to_list(length=50)

        return [
            {
                "id": str(a["_id"]),
                "type": a.get("alert_type"),
                "severity": a.get("severity"),
                "message": a.get("message"),
                "status": a.get("status"),
                "created_at": a.get("created_at")
            }
            for a in alerts
        ]

    async def _calculate_compliance_streak(self, worker_id: str) -> int:
        """Calculate consecutive days without violations."""
        # Get entries sorted by date descending
        entries = await self.db.gate_entries.find({
            "worker_id": ObjectId(worker_id)
        }).sort("timestamp", -1).limit(90).to_list(length=90)

        if not entries:
            return 0

        streak = 0
        current_date = None

        for entry in entries:
            entry_date = entry["timestamp"].date()

            if current_date is None:
                current_date = entry_date

            if entry_date != current_date:
                if (current_date - entry_date).days > 1:
                    break
                current_date = entry_date

            if entry.get("violations"):
                break

            streak += 1

        return streak

    async def _get_worker_attendance(self, worker_id: str) -> Dict[str, Any]:
        """Get worker attendance summary."""
        # Last 30 days
        start = datetime.utcnow() - timedelta(days=30)

        entries = await self.db.gate_entries.find({
            "worker_id": ObjectId(worker_id),
            "entry_type": "entry",
            "timestamp": {"$gte": start}
        }).to_list(length=None)

        # Count unique days
        unique_days = len(set(e["timestamp"].date() for e in entries))

        return {
            "days_worked_last_30": unique_days,
            "total_entries_last_30": len(entries),
            "attendance_rate": round((unique_days / 30) * 100, 1)
        }

    def _format_entry(self, entry: Dict) -> Dict[str, Any]:
        """Format a gate entry for display."""
        return {
            "id": str(entry.get("_id", "")),
            "worker_id": str(entry.get("worker_id", "")),
            "employee_id": entry.get("employee_id"),
            "worker_name": entry.get("worker_name"),
            "entry_type": entry.get("entry_type"),
            "timestamp": entry.get("timestamp"),
            "ppe_compliant": entry.get("ppe_compliant", False),
            "violations": entry.get("violations", []),
            "status": entry.get("status")
        }

    def _calculate_change(self, current: int, previous: int) -> float:
        """Calculate percentage change."""
        if previous == 0:
            return 100 if current > 0 else 0
        return round(((current - previous) / previous) * 100, 1)
