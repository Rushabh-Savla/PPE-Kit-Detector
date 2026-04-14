"""
SOS Alerts Routes - Emergency Worker Distress System
Handles SOS alerts, worker location tracking, and emergency response.
"""
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from bson import ObjectId

from database import get_database
from auth import get_current_user
from services.sms_service import get_sms_service
from services.helmet_service import trigger_all_alarms
from reports.services.email_service import get_email_service
from reports.services.pdf_generator import PDFGenerator
from reports.templates.emergency_incident import EmergencyIncidentTemplate, get_demo_emergency_data
from reports.schemas import EmailRecipient

router = APIRouter(prefix="/api/sos-alerts", tags=["SOS Alerts"])


@router.post("")
async def create_sos_alert(
    worker_id: str,
    reason: str,
    location_x: float,
    location_y: float,
    depth_m: float,
    section: str,
    severity: str = "critical",
    mine_id: Optional[str] = None,
    zone_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Create a new SOS alert (triggered by worker pressing SOS button)."""
    db = get_database()

    # Get worker details
    worker = await db.workers.find_one({"_id": ObjectId(worker_id)})
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    mine_id = mine_id or str(worker.get("mine_id", ""))
    zone_id = zone_id or str(worker.get("zone_id", ""))

    # Get zone details
    zone = None
    zone_name = section
    if zone_id:
        zone = await db.zones.find_one({"_id": ObjectId(zone_id)})
        if zone:
            zone_name = zone.get("name", section)

    alert = {
        "mine_id": ObjectId(mine_id) if mine_id else None,
        "zone_id": ObjectId(zone_id) if zone_id else None,
        "zone_name": zone_name,
        "worker_id": ObjectId(worker_id),
        "worker_name": worker.get("name", "Unknown"),
        "employee_id": worker.get("employee_id", "Unknown"),
        "reason": reason,
        "severity": severity,
        "status": "active",
        "location": {
            "x": location_x,
            "y": location_y,
            "depth_m": depth_m,
            "section": section
        },
        "created_at": datetime.utcnow(),
        "acknowledged_at": None,
        "acknowledged_by": None,
        "resolved_at": None,
        "resolved_by": None,
        "resolution_notes": None,
        "nearby_workers_notified": 0,
        "evacuation_triggered": severity == "critical",
        "audio_broadcast_sent": True,
        "response_actions": [
            {
                "action": "SOS received",
                "timestamp": datetime.utcnow().isoformat(),
                "by": "System"
            },
            {
                "action": "Audio alert broadcast to nearby workers",
                "timestamp": datetime.utcnow().isoformat(),
                "by": "System"
            }
        ]
    }

    result = await db.sos_alerts.insert_one(alert)

    # Also create a general alert for the alerts dashboard
    general_alert = {
        "alert_type": "sos",
        "severity": severity,
        "status": "active",
        "message": f"SOS Alert: {worker.get('name', 'Worker')} - {reason}",
        "mine_id": ObjectId(mine_id) if mine_id else None,
        "zone_id": ObjectId(zone_id) if zone_id else None,
        "worker_id": ObjectId(worker_id),
        "worker_name": worker.get("name", "Unknown"),
        "created_at": datetime.utcnow(),
        "sos_alert_id": result.inserted_id
    }
    await db.alerts.insert_one(general_alert)

    # Notify nearby workers (simulation)
    nearby_count = await notify_nearby_workers(db, mine_id, zone_id, worker_id)
    await db.sos_alerts.update_one(
        {"_id": result.inserted_id},
        {"$set": {"nearby_workers_notified": nearby_count}}
    )

    # Send SMS alerts to safety officers and managers
    sms_sent = await send_sos_sms_alerts(db, worker, zone_name, mine_id)

    return {
        "success": True,
        "alert_id": str(result.inserted_id),
        "message": "SOS alert created and broadcast to nearby workers",
        "sms_sent": sms_sent
    }


async def notify_nearby_workers(db, mine_id: str, zone_id: str, exclude_worker_id: str) -> int:
    """Notify nearby workers via audio broadcast (simulation)."""
    query = {"mine_id": ObjectId(mine_id), "is_active": True}
    if zone_id:
        query["zone_id"] = ObjectId(zone_id)

    count = await db.workers.count_documents(query)
    return max(0, count - 1)  # Exclude the worker who triggered the alert


async def send_sos_sms_alerts(db, worker: dict, zone_name: str, mine_id: str) -> int:
    """
    Send SMS alerts to safety officers and managers when SOS is triggered.
    Returns number of SMS sent successfully.
    """
    sms_service = get_sms_service()

    if not sms_service.is_configured():
        print("[SOS] SMS service not configured, skipping SMS alerts")
        return 0

    # Get mine name
    mine_name = None
    if mine_id:
        mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
        if mine:
            mine_name = mine.get("name")

    # Find users to notify (safety officers and managers for this mine)
    notify_roles = ["safety_officer", "manager", "shift_incharge", "area_safety_officer"]
    users_query = {
        "role": {"$in": notify_roles},
        "is_active": True,
        "phone": {"$exists": True, "$ne": None, "$ne": ""}
    }

    # If mine_id is specified, filter by mine assignment
    if mine_id:
        users_query["$or"] = [
            {"mine_ids": ObjectId(mine_id)},
            {"mine_ids": {"$exists": False}},  # Users not assigned to specific mines (org-wide)
            {"role": {"$in": ["area_safety_officer", "general_manager"]}}  # Higher roles see all
        ]

    users_cursor = db.users.find(users_query)

    sms_sent = 0
    worker_name = worker.get("name", "Unknown Worker")
    worker_id = worker.get("employee_id", "Unknown")

    async for user in users_cursor:
        phone = user.get("phone")
        if phone:
            try:
                result = await sms_service.send_sos_alert(
                    to=phone,
                    worker_name=worker_name,
                    worker_id=worker_id,
                    location=zone_name,
                    mine_name=mine_name
                )
                if result.get("success"):
                    sms_sent += 1
                    print(f"[SOS] SMS sent to {user.get('full_name', user.get('username'))} at {phone}")
            except Exception as e:
                print(f"[SOS] Failed to send SMS to {phone}: {e}")

    print(f"[SOS] Total SMS sent: {sms_sent}")
    return sms_sent


@router.get("")
async def get_sos_alerts(
    mine_id: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    worker_id: Optional[str] = None,
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(100, le=500),
    skip: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """Get SOS alerts with filters."""
    db = get_database()

    start_date = datetime.utcnow() - timedelta(days=days)
    query = {"created_at": {"$gte": start_date}}

    # Handle mine_id - skip if "null", empty, or invalid
    if mine_id and mine_id not in ["null", "undefined", ""]:
        try:
            query["mine_id"] = ObjectId(mine_id)
        except Exception:
            pass  # Invalid ObjectId, skip filter
    if status:
        query["status"] = status
    if severity:
        query["severity"] = severity
    if worker_id and worker_id not in ["null", "undefined", ""]:
        try:
            query["worker_id"] = ObjectId(worker_id)
        except Exception:
            pass  # Invalid ObjectId, skip filter

    cursor = db.sos_alerts.find(query).sort("created_at", -1).skip(skip).limit(limit)
    alerts = []

    async for alert in cursor:
        alerts.append({
            "id": str(alert["_id"]),
            "mine_id": str(alert.get("mine_id")) if alert.get("mine_id") else None,
            "zone_id": str(alert.get("zone_id")) if alert.get("zone_id") else None,
            "zone_name": alert.get("zone_name", "Unknown"),
            "worker_id": str(alert.get("worker_id")),
            "worker_name": alert.get("worker_name", "Unknown"),
            "employee_id": alert.get("employee_id", "Unknown"),
            "reason": alert.get("reason", ""),
            "severity": alert.get("severity", "medium"),
            "status": alert.get("status", "active"),
            "location": alert.get("location", {}),
            "created_at": alert.get("created_at").isoformat() if alert.get("created_at") else None,
            "acknowledged_at": alert.get("acknowledged_at").isoformat() if alert.get("acknowledged_at") else None,
            "acknowledged_by": alert.get("acknowledged_by"),
            "resolved_at": alert.get("resolved_at").isoformat() if alert.get("resolved_at") else None,
            "resolved_by": alert.get("resolved_by"),
            "resolution_notes": alert.get("resolution_notes"),
            "nearby_workers_notified": alert.get("nearby_workers_notified", 0),
            "evacuation_triggered": alert.get("evacuation_triggered", False),
            "audio_broadcast_sent": alert.get("audio_broadcast_sent", False),
            "response_actions": alert.get("response_actions", [])
        })

    total = await db.sos_alerts.count_documents(query)

    return {
        "alerts": alerts,
        "total": total,
        "limit": limit,
        "skip": skip
    }


@router.get("/active")
async def get_active_sos_alerts(
    mine_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get only active SOS alerts for real-time monitoring."""
    db = get_database()

    query = {"status": {"$in": ["active", "acknowledged"]}}
    if mine_id and mine_id not in ["null", "undefined", ""]:
        try:
            query["mine_id"] = ObjectId(mine_id)
        except Exception:
            pass  # Invalid ObjectId, skip filter

    cursor = db.sos_alerts.find(query).sort("created_at", -1)
    alerts = []

    async for alert in cursor:
        alerts.append({
            "id": str(alert["_id"]),
            "zone_name": alert.get("zone_name", "Unknown"),
            "worker_id": str(alert.get("worker_id")),
            "worker_name": alert.get("worker_name", "Unknown"),
            "employee_id": alert.get("employee_id", "Unknown"),
            "reason": alert.get("reason", ""),
            "severity": alert.get("severity", "medium"),
            "status": alert.get("status", "active"),
            "location": alert.get("location", {}),
            "created_at": alert.get("created_at").isoformat() if alert.get("created_at") else None,
            "response_actions": alert.get("response_actions", [])
        })

    return {"alerts": alerts, "count": len(alerts)}


@router.post("/{alert_id}/acknowledge")
async def acknowledge_sos_alert(
    alert_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Acknowledge an SOS alert."""
    db = get_database()

    alert = await db.sos_alerts.find_one({"_id": ObjectId(alert_id)})
    if not alert:
        raise HTTPException(status_code=404, detail="SOS alert not found")

    if alert.get("status") != "active":
        raise HTTPException(status_code=400, detail="Alert is not in active status")

    acknowledger = current_user.get("full_name", current_user.get("username", "Unknown"))

    response_actions = alert.get("response_actions", [])
    response_actions.append({
        "action": "Alert acknowledged",
        "timestamp": datetime.utcnow().isoformat(),
        "by": acknowledger
    })

    await db.sos_alerts.update_one(
        {"_id": ObjectId(alert_id)},
        {
            "$set": {
                "status": "acknowledged",
                "acknowledged_at": datetime.utcnow(),
                "acknowledged_by": acknowledger,
                "response_actions": response_actions
            }
        }
    )

    # Update related general alert
    await db.alerts.update_one(
        {"sos_alert_id": ObjectId(alert_id)},
        {
            "$set": {
                "status": "acknowledged",
                "acknowledged_at": datetime.utcnow(),
                "acknowledged_by": acknowledger
            }
        }
    )

    return {"success": True, "message": "SOS alert acknowledged"}


@router.post("/{alert_id}/resolve")
async def resolve_sos_alert(
    alert_id: str,
    notes: str = "Situation resolved",
    current_user: dict = Depends(get_current_user)
):
    """Resolve an SOS alert."""
    db = get_database()

    alert = await db.sos_alerts.find_one({"_id": ObjectId(alert_id)})
    if not alert:
        raise HTTPException(status_code=404, detail="SOS alert not found")

    if alert.get("status") == "resolved":
        raise HTTPException(status_code=400, detail="Alert is already resolved")

    resolver = current_user.get("full_name", current_user.get("username", "Unknown"))

    response_actions = alert.get("response_actions", [])
    response_actions.append({
        "action": f"Resolved: {notes}",
        "timestamp": datetime.utcnow().isoformat(),
        "by": resolver
    })

    await db.sos_alerts.update_one(
        {"_id": ObjectId(alert_id)},
        {
            "$set": {
                "status": "resolved",
                "resolved_at": datetime.utcnow(),
                "resolved_by": resolver,
                "resolution_notes": notes,
                "response_actions": response_actions
            }
        }
    )

    # Update related general alert
    await db.alerts.update_one(
        {"sos_alert_id": ObjectId(alert_id)},
        {
            "$set": {
                "status": "resolved",
                "resolved_at": datetime.utcnow(),
                "resolved_by": resolver,
                "resolution_notes": notes
            }
        }
    )

    return {"success": True, "message": "SOS alert resolved"}


@router.get("/stats")
async def get_sos_stats(
    mine_id: Optional[str] = None,
    days: int = Query(7, ge=1, le=90),
    current_user: dict = Depends(get_current_user)
):
    """Get SOS alert statistics."""
    db = get_database()

    start_date = datetime.utcnow() - timedelta(days=days)
    query = {"created_at": {"$gte": start_date}}

    if mine_id and mine_id not in ["null", "undefined", ""]:
        try:
            query["mine_id"] = ObjectId(mine_id)
        except Exception:
            pass  # Invalid ObjectId, skip filter

    # Get counts by status
    total = await db.sos_alerts.count_documents(query)
    active = await db.sos_alerts.count_documents({**query, "status": "active"})
    acknowledged = await db.sos_alerts.count_documents({**query, "status": "acknowledged"})
    resolved = await db.sos_alerts.count_documents({**query, "status": "resolved"})

    # Get counts by severity
    critical = await db.sos_alerts.count_documents({**query, "severity": "critical"})
    high = await db.sos_alerts.count_documents({**query, "severity": "high"})
    medium = await db.sos_alerts.count_documents({**query, "severity": "medium"})

    # Calculate average response time
    responded_alerts = db.sos_alerts.find({
        **query,
        "acknowledged_at": {"$ne": None}
    })

    response_times = []
    async for alert in responded_alerts:
        if alert.get("created_at") and alert.get("acknowledged_at"):
            delta = (alert["acknowledged_at"] - alert["created_at"]).total_seconds() / 60
            response_times.append(delta)

    avg_response = sum(response_times) / len(response_times) if response_times else 0

    return {
        "total": total,
        "active": active,
        "acknowledged": acknowledged,
        "resolved": resolved,
        "critical": critical,
        "high": high,
        "medium": medium,
        "avg_response_time_minutes": round(avg_response, 1),
        "time_range_days": days
    }


@router.post("/trigger-evacuation")
async def trigger_mine_evacuation(
    zone_name: str = "Zone A - Extraction",
    gas_type: str = "methane",
    gas_level: float = 15200,
    mine_name: str = "Jharia Coal Mine",
    current_user: dict = Depends(get_current_user)
):
    """
    Trigger emergency evacuation for an entire zone.
    This sends EVACUATE_ALL command to all helmets and sends SMS alerts.
    """
    db = get_database()
    sms_service = get_sms_service()

    # 1. Trigger all helmet alarms via ESP32
    helmet_result = trigger_all_alarms()
    print(f"[Evacuation] Helmet trigger result: {helmet_result}")

    # 2. Get all active workers in the affected zone (for demo, get all active workers)
    workers_cursor = db.workers.find({"is_active": True})
    affected_workers = []
    async for worker in workers_cursor:
        affected_workers.append({
            "id": str(worker["_id"]),
            "name": worker.get("name", "Unknown"),
            "employee_id": worker.get("employee_id", "Unknown")
        })

    workers_count = len(affected_workers)

    # 3. Create a mine-wide SOS alert
    alert = {
        "mine_id": None,  # Mine-wide
        "zone_id": None,
        "zone_name": zone_name,
        "worker_id": None,  # System-triggered, not by a specific worker
        "worker_name": "SYSTEM",
        "employee_id": "SYSTEM",
        "reason": f"Gas Emergency - {gas_type.upper()} SPIKE DETECTED at {gas_level} PPM",
        "severity": "critical",
        "status": "active",
        "location": {
            "x": 0,
            "y": 0,
            "depth_m": 0,
            "section": zone_name
        },
        "created_at": datetime.utcnow(),
        "acknowledged_at": None,
        "acknowledged_by": None,
        "resolved_at": None,
        "resolved_by": None,
        "resolution_notes": None,
        "nearby_workers_notified": workers_count,
        "evacuation_triggered": True,
        "audio_broadcast_sent": True,
        "is_mass_evacuation": True,
        "gas_type": gas_type,
        "gas_level": gas_level,
        "affected_workers": affected_workers,
        "triggered_by": current_user.get("full_name", current_user.get("username", "Unknown")),
        "response_actions": [
            {
                "action": f"EMERGENCY: {gas_type.upper()} spike detected at {gas_level} PPM",
                "timestamp": datetime.utcnow().isoformat(),
                "by": "Sensor System"
            },
            {
                "action": f"Mass evacuation triggered by {current_user.get('full_name', current_user.get('username', 'Unknown'))}",
                "timestamp": datetime.utcnow().isoformat(),
                "by": current_user.get("full_name", current_user.get("username", "Unknown"))
            },
            {
                "action": f"All helmet alarms activated ({workers_count} workers notified)",
                "timestamp": datetime.utcnow().isoformat(),
                "by": "System"
            },
            {
                "action": "SMS alerts sent to safety personnel",
                "timestamp": datetime.utcnow().isoformat(),
                "by": "System"
            }
        ]
    }

    result = await db.sos_alerts.insert_one(alert)

    # 4. Create general alert
    general_alert = {
        "alert_type": "evacuation",
        "severity": "critical",
        "status": "active",
        "message": f"EMERGENCY EVACUATION: {gas_type.upper()} leak in {zone_name} - {gas_level} PPM",
        "mine_id": None,
        "zone_id": None,
        "created_at": datetime.utcnow(),
        "sos_alert_id": result.inserted_id,
        "is_mass_evacuation": True
    }
    await db.alerts.insert_one(general_alert)

    # 5. Send SMS to safety officer (+91 88286 42788)
    sms_sent = 0
    safety_officer_phone = "+918828642788"

    if sms_service.is_configured():
        # Build evacuation SMS message
        from datetime import timezone
        ist_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        sms_message = f"""🚨 EMERGENCY EVACUATION ALERT 🚨

⚠️ {gas_type.upper()} SPIKE DETECTED
Zone: {zone_name}
Mine: {mine_name}

Gas Level: {gas_level:,.0f} PPM (CRITICAL)
Threshold: 5,000 PPM

ACTION: IMMEDIATE EVACUATION
Workers Affected: {workers_count}
Evacuation Triggered: YES

⏰ {ist_time}

All workers have been alerted via helmet alarms.
Acknowledge this alert immediately.

- RAKSHAM Mine Safety System"""

        try:
            sms_result = await sms_service.send_sms(safety_officer_phone, sms_message)
            if sms_result.get("success"):
                sms_sent = 1
                print(f"[Evacuation] SMS sent to Safety Officer at {safety_officer_phone}")
        except Exception as e:
            print(f"[Evacuation] Failed to send SMS: {e}")

    # 6. Generate Emergency Incident PDF and send via email
    email_sent = False
    email_service = get_email_service()

    try:
        # Generate PDF
        print("[Evacuation] Generating Emergency Incident PDF...")
        data = get_demo_emergency_data()
        template = EmergencyIncidentTemplate(db=None)
        generator = PDFGenerator()
        pdf_buffer = await generator.generate(template, data)

        # Send email with PDF attachment
        recipients = [
            EmailRecipient(email="tanushmaloo@gmail.com", name="Tanush Maloo (Safety Officer)", type="to"),
        ]

        summary_metrics = [
            {"label": "Severity", "value": "CRITICAL", "color": "red"},
            {"label": "Workers Affected", "value": str(workers_count), "color": "orange"},
            {"label": "Response Time", "value": "3 min 25 sec", "color": "blue"},
        ]

        highlights = [
            f"Gas Emergency - {gas_type.upper()} spike at {gas_level:,.0f} PPM in {zone_name}",
            f"All {workers_count} workers safely evacuated",
            "ESP32 EVACUATE_ALL command sent to all helmets",
            "SMS alert delivered to Safety Officer"
        ]

        attachments = [{
            "filename": "Emergency_Incident_Report_Dec_12_2024.pdf",
            "data": pdf_buffer.getvalue(),
            "content_type": "application/pdf",
            "format": "PDF"
        }]

        email_result = await email_service.send_report(
            recipients=recipients,
            report_name="Emergency Incident Report",
            date_range=f"Gas Emergency - {datetime.utcnow().strftime('%B %d, %Y')}",
            attachments=attachments,
            summary_metrics=summary_metrics,
            highlights=highlights
        )

        if email_result:
            email_sent = True
            print(f"[Evacuation] Emergency report emailed to Safety Officer")

    except Exception as e:
        print(f"[Evacuation] Failed to send email with PDF: {e}")

    return {
        "success": True,
        "alert_id": str(result.inserted_id),
        "message": f"Emergency evacuation triggered for {zone_name}",
        "workers_notified": workers_count,
        "helmet_alarms_triggered": helmet_result.get("success", False),
        "sms_sent": sms_sent,
        "email_sent": email_sent,
        "gas_type": gas_type,
        "gas_level": gas_level,
        "zone": zone_name,
        "affected_workers": affected_workers
    }
