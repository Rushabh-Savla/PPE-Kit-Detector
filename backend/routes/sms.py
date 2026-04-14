"""
SMS Routes - Twilio SMS Integration
Handles SMS sending for alerts and notifications.
"""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel

from auth import get_current_user
from services.sms_service import get_sms_service

router = APIRouter(prefix="/api/sms", tags=["SMS"])


class SendSMSRequest(BaseModel):
    """Request model for sending SMS."""
    to: str
    message: str


class SendBulkSMSRequest(BaseModel):
    """Request model for sending bulk SMS."""
    recipients: List[str]
    message: str


class SendAlertSMSRequest(BaseModel):
    """Request model for sending alert SMS."""
    to: str
    alert_type: str
    severity: str
    message: str
    location: Optional[str] = None


class SendSOSAlertRequest(BaseModel):
    """Request model for SOS alert SMS."""
    to: str
    worker_name: str
    worker_id: str
    location: str
    mine_name: Optional[str] = None


class SendGasAlertRequest(BaseModel):
    """Request model for gas alert SMS."""
    to: str
    gas_type: str
    level_ppm: float
    zone_name: str
    severity: str
    mine_name: Optional[str] = None


@router.get("/status")
async def get_sms_status(current_user: dict = Depends(get_current_user)):
    """Check if SMS service is configured."""
    sms_service = get_sms_service()
    return {
        "configured": sms_service.is_configured(),
        "account_sid": sms_service.account_sid[:10] + "..." if sms_service.account_sid else None,
        "messaging_service_sid": sms_service.messaging_service_sid
    }


@router.post("/send")
async def send_sms(
    request: SendSMSRequest,
    current_user: dict = Depends(get_current_user)
):
    """Send a simple SMS message."""
    sms_service = get_sms_service()

    result = await sms_service.send_sms(
        to=request.to,
        message=request.message
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/send-bulk")
async def send_bulk_sms(
    request: SendBulkSMSRequest,
    current_user: dict = Depends(get_current_user)
):
    """Send SMS to multiple recipients."""
    sms_service = get_sms_service()

    result = await sms_service.send_bulk_sms(
        recipients=request.recipients,
        message=request.message
    )

    return result


@router.post("/send-alert")
async def send_alert_sms(
    request: SendAlertSMSRequest,
    current_user: dict = Depends(get_current_user)
):
    """Send a formatted alert SMS."""
    sms_service = get_sms_service()

    result = await sms_service.send_alert_sms(
        to=request.to,
        alert_type=request.alert_type,
        severity=request.severity,
        message=request.message,
        location=request.location
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/send-sos-alert")
async def send_sos_alert_sms(
    request: SendSOSAlertRequest,
    current_user: dict = Depends(get_current_user)
):
    """Send an SOS emergency alert SMS."""
    sms_service = get_sms_service()

    result = await sms_service.send_sos_alert(
        to=request.to,
        worker_name=request.worker_name,
        worker_id=request.worker_id,
        location=request.location,
        mine_name=request.mine_name
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/send-gas-alert")
async def send_gas_alert_sms(
    request: SendGasAlertRequest,
    current_user: dict = Depends(get_current_user)
):
    """Send a gas level alert SMS."""
    sms_service = get_sms_service()

    result = await sms_service.send_gas_alert(
        to=request.to,
        gas_type=request.gas_type,
        level_ppm=request.level_ppm,
        zone_name=request.zone_name,
        severity=request.severity,
        mine_name=request.mine_name
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/test")
async def test_sms(
    phone: str = Body(..., embed=True),
    current_user: dict = Depends(get_current_user)
):
    """Send a test SMS to verify configuration."""
    sms_service = get_sms_service()

    if not sms_service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="SMS service not configured. Please set TWILIO_AUTH_TOKEN in environment."
        )

    result = await sms_service.send_sms(
        to=phone,
        message="🔔 Test message from Raksham Mine Safety System. Your SMS alerts are working correctly!"
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "success": True,
        "message": f"Test SMS sent to {phone}",
        "message_sid": result["message_sid"]
    }


@router.get("/quick-test/{phone}")
async def quick_test_sms(phone: str):
    """Quick test endpoint (no auth required) - for development only."""
    sms_service = get_sms_service()

    if not sms_service.is_configured():
        return {"success": False, "error": "SMS not configured"}

    result = await sms_service.send_sms(
        to=phone,
        message="🔔 Test from Raksham Mine Safety System!"
    )

    return result
