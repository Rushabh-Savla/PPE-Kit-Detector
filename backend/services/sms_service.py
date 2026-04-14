"""
SMS Service for sending alerts and notifications via Twilio.
"""

import os
import httpx
from typing import Optional, List
from datetime import datetime


class SMSService:
    """
    Service for sending SMS messages via Twilio API.
    """

    def __init__(self):
        """Initialize SMS service with Twilio credentials from environment."""
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.messaging_service_sid = os.getenv("TWILIO_MESSAGING_SERVICE_SID", "")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER", "")  # Optional: direct from number

        self.base_url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"

        # Log configuration (without auth token)
        print(f"[SMSService] Initialized with:")
        print(f"  - Account SID: {self.account_sid[:10]}...")
        print(f"  - Messaging Service SID: {self.messaging_service_sid}")
        print(f"  - Auth Token: {'***' if self.auth_token else 'NOT CONFIGURED'}")

    def is_configured(self) -> bool:
        """Check if the service is properly configured."""
        return bool(self.account_sid and self.auth_token)

    async def send_sms(
        self,
        to: str,
        message: str,
        from_number: Optional[str] = None
    ) -> dict:
        """
        Send an SMS message.

        Args:
            to: Recipient phone number (with country code, e.g., +918828642788)
            message: Message body
            from_number: Optional sender number (uses messaging service if not provided)

        Returns:
            dict with 'success', 'message_sid', and 'error' keys
        """
        if not self.is_configured():
            print("[SMSService] Warning: Twilio credentials not configured. SMS not sent.")
            return {
                "success": False,
                "message_sid": None,
                "error": "Twilio credentials not configured"
            }

        # Ensure phone number has country code
        if not to.startswith('+'):
            to = f"+91{to}"  # Default to India country code

        print(f"[SMSService] Sending SMS to {to}...")

        try:
            # Prepare request data
            data = {
                "To": to,
                "Body": message,
            }

            # Use messaging service or direct from number
            if from_number or self.from_number:
                data["From"] = from_number or self.from_number
            else:
                data["MessagingServiceSid"] = self.messaging_service_sid

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    data=data,
                    auth=(self.account_sid, self.auth_token),
                    timeout=30.0
                )

                if response.status_code in [200, 201]:
                    result = response.json()
                    print(f"[SMSService] SMS sent successfully. SID: {result.get('sid')}")
                    return {
                        "success": True,
                        "message_sid": result.get("sid"),
                        "error": None
                    }
                else:
                    error_msg = response.text
                    print(f"[SMSService] Failed to send SMS: {error_msg}")
                    return {
                        "success": False,
                        "message_sid": None,
                        "error": error_msg
                    }

        except Exception as e:
            print(f"[SMSService] Error sending SMS: {e}")
            return {
                "success": False,
                "message_sid": None,
                "error": str(e)
            }

    async def send_bulk_sms(
        self,
        recipients: List[str],
        message: str
    ) -> dict:
        """
        Send SMS to multiple recipients.

        Args:
            recipients: List of phone numbers
            message: Message body

        Returns:
            dict with 'success_count', 'failed_count', and 'results'
        """
        results = []
        success_count = 0
        failed_count = 0

        for phone in recipients:
            result = await self.send_sms(phone, message)
            results.append({"phone": phone, **result})
            if result["success"]:
                success_count += 1
            else:
                failed_count += 1

        return {
            "success_count": success_count,
            "failed_count": failed_count,
            "results": results
        }

    async def send_alert_sms(
        self,
        to: str,
        alert_type: str,
        severity: str,
        message: str,
        location: Optional[str] = None
    ) -> dict:
        """
        Send a formatted alert SMS.

        Args:
            to: Recipient phone number
            alert_type: Type of alert (e.g., 'sos', 'gas', 'ppe_violation')
            severity: Alert severity (e.g., 'critical', 'high', 'medium')
            message: Alert message
            location: Optional location info

        Returns:
            dict with send result
        """
        # Format alert message
        severity_emoji = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "⚡",
            "low": "ℹ️"
        }.get(severity.lower(), "📢")

        alert_title = {
            "sos": "SOS EMERGENCY",
            "gas": "GAS ALERT",
            "ppe_violation": "PPE VIOLATION",
            "evacuation": "EVACUATION ORDER",
            "worker_risk": "WORKER AT RISK"
        }.get(alert_type.lower(), "ALERT")

        formatted_message = f"{severity_emoji} {alert_title} {severity_emoji}\n\n"
        formatted_message += f"{message}\n"

        if location:
            formatted_message += f"\n📍 Location: {location}"

        formatted_message += f"\n\n⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        formatted_message += "\n\n- Raksham Mine Safety System"

        return await self.send_sms(to, formatted_message)

    async def send_sos_alert(
        self,
        to: str,
        worker_name: str,
        worker_id: str,
        location: str,
        mine_name: Optional[str] = None
    ) -> dict:
        """
        Send an SOS emergency alert SMS.

        Args:
            to: Recipient phone number (safety officer/manager)
            worker_name: Name of worker who triggered SOS
            worker_id: Employee ID of the worker
            location: Location/zone where SOS was triggered
            mine_name: Name of the mine

        Returns:
            dict with send result
        """
        message = f"🚨 SOS EMERGENCY ALERT 🚨\n\n"
        message += f"Worker: {worker_name}\n"
        message += f"ID: {worker_id}\n"
        message += f"Location: {location}\n"

        if mine_name:
            message += f"Mine: {mine_name}\n"

        message += f"\n⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        message += "\n⚠️ IMMEDIATE ACTION REQUIRED\n"
        message += "\n- Raksham Mine Safety System"

        return await self.send_sms(to, message)

    async def send_gas_alert(
        self,
        to: str,
        gas_type: str,
        level_ppm: float,
        zone_name: str,
        severity: str,
        mine_name: Optional[str] = None
    ) -> dict:
        """
        Send a gas level alert SMS.

        Args:
            to: Recipient phone number
            gas_type: Type of gas (CH4, CO, etc.)
            level_ppm: Gas level in PPM
            zone_name: Zone where detected
            severity: Alert severity
            mine_name: Name of the mine

        Returns:
            dict with send result
        """
        severity_emoji = "🚨" if severity.lower() == "critical" else "⚠️"

        message = f"{severity_emoji} GAS ALERT - {severity.upper()} {severity_emoji}\n\n"
        message += f"Gas: {gas_type}\n"
        message += f"Level: {level_ppm:.1f} PPM\n"
        message += f"Zone: {zone_name}\n"

        if mine_name:
            message += f"Mine: {mine_name}\n"

        message += f"\n⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"

        if severity.lower() == "critical":
            message += "\n🚨 EVACUATE IMMEDIATELY\n"

        message += "\n- Raksham Mine Safety System"

        return await self.send_sms(to, message)


# Singleton instance
_sms_service: Optional[SMSService] = None


def get_sms_service() -> SMSService:
    """Get or create the SMS service singleton."""
    global _sms_service
    if _sms_service is None:
        _sms_service = SMSService()
    return _sms_service
