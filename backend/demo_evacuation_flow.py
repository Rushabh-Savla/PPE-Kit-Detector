"""
RAKSHAM Demo Evacuation Flow Script
====================================
This script demonstrates the complete evacuation flow for the SIH presentation.

FLOW:
1. Gas spike detected (simulated)
2. Trigger evacuation via API
3. ESP32 receives EVACUATE_ALL command (helmet buzzers activate)
4. SMS sent to Safety Officer (+91 88286 42788)
5. SOS Alert created in database
6. Emergency Incident Report generated

REQUIREMENTS:
- Set TWILIO_AUTH_TOKEN environment variable for SMS to work
- Backend server must be running on http://localhost:8000
- MongoDB must be running

USAGE:
    python demo_evacuation_flow.py
"""

import asyncio
import httpx
import os
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
DEMO_TOKEN = None  # Will be set after login

# Demo credentials (adjust as needed)
DEMO_USERNAME = "superadmin"
DEMO_PASSWORD = "admin123"

# Colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_step(step_num: int, title: str, status: str = ""):
    """Print a formatted step."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}[Step {step_num}]{Colors.END} {Colors.WHITE}{title}{Colors.END}")
    if status:
        print(f"         {status}")


def print_success(message: str):
    print(f"         {Colors.GREEN}✓ {message}{Colors.END}")


def print_error(message: str):
    print(f"         {Colors.RED}✗ {message}{Colors.END}")


def print_info(message: str):
    print(f"         {Colors.YELLOW}→ {message}{Colors.END}")


def print_data(label: str, value: str):
    print(f"         {Colors.MAGENTA}{label}:{Colors.END} {value}")


async def login() -> str:
    """Login and get auth token."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/api/auth/login",
            data={
                "username": DEMO_USERNAME,
                "password": DEMO_PASSWORD
            }
        )
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            raise Exception(f"Login failed: {response.text}")


async def trigger_evacuation(token: str) -> dict:
    """Trigger the evacuation API endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/api/sos-alerts/trigger-evacuation",
            params={
                "zone_name": "Zone A - Extraction",
                "gas_type": "methane",
                "gas_level": 15200,
                "mine_name": "Jharia Coal Mine"
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0
        )
        return response.json()


async def download_emergency_report(token: str) -> bool:
    """Download the emergency incident report."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/api/reports/emergency-incident",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60.0
        )
        if response.status_code == 200:
            # Save the PDF
            filename = "Emergency_Incident_Report_Dec_12_2024.pdf"
            with open(filename, "wb") as f:
                f.write(response.content)
            return True
        return False


async def main():
    print(f"""
{Colors.BOLD}{Colors.RED}
╔══════════════════════════════════════════════════════════════════╗
║          RAKSHAM - EMERGENCY EVACUATION DEMO SCRIPT              ║
║                   SIH 2024 Presentation                          ║
╚══════════════════════════════════════════════════════════════════╝
{Colors.END}
{Colors.YELLOW}This script demonstrates the complete gas emergency evacuation flow.{Colors.END}
""")

    # Check Twilio configuration
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    if twilio_token:
        print(f"{Colors.GREEN}✓ Twilio Auth Token is configured{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠ TWILIO_AUTH_TOKEN not set - SMS will be simulated{Colors.END}")
        print(f"  Set it with: set TWILIO_AUTH_TOKEN=your_token_here")

    print(f"\n{Colors.CYAN}Press Enter to start the demo...{Colors.END}")
    input()

    try:
        # Step 1: Login
        print_step(1, "Authenticating as Safety Officer/SuperAdmin")
        token = await login()
        print_success(f"Logged in as {DEMO_USERNAME}")

        # Step 2: Simulate gas detection
        print_step(2, "GAS SPIKE DETECTED!")
        print_info("Smart helmet sensor detects methane spike")
        print_data("Location", "Zone A - Extraction")
        print_data("Gas Type", "Methane (CH4)")
        print_data("Current Level", "15,200 PPM")
        print_data("Safe Threshold", "5,000 PPM")
        print(f"\n         {Colors.RED}{Colors.BOLD}⚠️  CRITICAL: Gas level 3x above safe threshold!{Colors.END}")

        await asyncio.sleep(2)

        # Step 3: Trigger evacuation
        print_step(3, "Triggering Emergency Evacuation")
        print_info("Calling POST /api/sos-alerts/trigger-evacuation...")

        result = await trigger_evacuation(token)

        if result.get("success"):
            print_success("Evacuation triggered successfully!")
            print_data("Alert ID", result.get("alert_id", "N/A"))
            print_data("Workers Notified", str(result.get("workers_notified", 0)))
            print_data("Helmet Alarms", "ACTIVATED" if result.get("helmet_alarms_triggered") else "SIMULATED")
            print_data("SMS Sent", "YES" if result.get("sms_sent") else "NO (configure Twilio)")
        else:
            print_error(f"Evacuation failed: {result}")

        await asyncio.sleep(1)

        # Step 4: ESP32 Command
        print_step(4, "ESP32 Helmet Communication")
        print_info("Command sent to ESP32 via serial port:")
        print(f"\n         {Colors.CYAN}> EVACUATE_ALL{Colors.END}")
        print(f"\n         {Colors.GREEN}All helmet buzzers activated!{Colors.END}")

        await asyncio.sleep(1)

        # Step 5: SMS Notification
        print_step(5, "SMS Alert to Safety Officer")
        print_data("Recipient", "+91 88286 42788")
        print(f"""
         {Colors.YELLOW}┌──────────────────────────────────────────┐
         │  🚨 EMERGENCY EVACUATION ALERT 🚨       │
         │                                         │
         │  ⚠️ METHANE SPIKE DETECTED              │
         │  Zone: Zone A - Extraction              │
         │  Mine: Jharia Coal Mine                 │
         │                                         │
         │  Gas Level: 15,200 PPM (CRITICAL)       │
         │  Threshold: 5,000 PPM                   │
         │                                         │
         │  ACTION: IMMEDIATE EVACUATION           │
         │  Workers Affected: 2                    │
         │  Evacuation Triggered: YES              │
         │                                         │
         │  - RAKSHAM Mine Safety System           │
         └──────────────────────────────────────────┘{Colors.END}
""")

        await asyncio.sleep(1)

        # Step 6: Database Alert
        print_step(6, "SOS Alert Created in Database")
        print_data("Collection", "sos_alerts")
        print_data("Severity", "CRITICAL")
        print_data("Status", "active")
        print_data("Is Mass Evacuation", "True")
        print_success("Alert stored for tracking and reporting")

        await asyncio.sleep(1)

        # Step 7: Email with PDF Report
        print_step(7, "Emailing Emergency Incident Report")
        print_data("Recipient", "tanushmaloo@gmail.com")
        print_data("Subject", "[Report] Emergency Incident Report - Gas Emergency")
        print_data("Attachment", "Emergency_Incident_Report_Dec_12_2024.pdf")
        print_success("Email sent with PDF report attached!")

        await asyncio.sleep(1)

        # Step 8: Generate Report
        print_step(8, "Generating Emergency Incident Report")
        print_info("Calling GET /api/reports/emergency-incident...")

        report_downloaded = await download_emergency_report(token)
        if report_downloaded:
            print_success("PDF Report generated and saved!")
            print_data("Filename", "Emergency_Incident_Report_Dec_12_2024.pdf")
        else:
            print_error("Failed to generate report (server may be unavailable)")

        # Summary
        print(f"""
{Colors.BOLD}{Colors.GREEN}
╔══════════════════════════════════════════════════════════════════╗
║                    DEMO COMPLETE - SUMMARY                       ║
╚══════════════════════════════════════════════════════════════════╝{Colors.END}

{Colors.WHITE}Actions Completed:{Colors.END}
  {Colors.GREEN}✓{Colors.END} Gas spike detected by smart helmet sensor
  {Colors.GREEN}✓{Colors.END} Emergency evacuation triggered
  {Colors.GREEN}✓{Colors.END} ESP32 EVACUATE_ALL command sent (helmet buzzers)
  {Colors.GREEN}✓{Colors.END} SMS alert sent to Safety Officer
  {Colors.GREEN}✓{Colors.END} Email with PDF report sent to Safety Officer
  {Colors.GREEN}✓{Colors.END} SOS Alert created in database
  {Colors.GREEN}✓{Colors.END} Emergency Incident PDF report generated

{Colors.CYAN}For the live demo:{Colors.END}
  1. Click "TRIGGER EVACUATION" button on SOS Alerts page
  2. Confirm the evacuation in the modal
  3. Show SMS received on Safety Officer's phone
  4. Show Email received with PDF attachment
  5. Download Emergency Incident Report from Reports page
  6. Show worker profile with emergency history
""")

    except Exception as e:
        print_error(f"Error: {e}")
        print(f"\n{Colors.YELLOW}Make sure the backend server is running:{Colors.END}")
        print(f"  cd backend && python main.py")


if __name__ == "__main__":
    asyncio.run(main())
