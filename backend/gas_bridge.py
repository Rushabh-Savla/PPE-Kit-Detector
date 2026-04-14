"""
Gas Sensor Bridge - Connects your Arduino gas sensors to the main backend
Reads from your existing gas.py sensor script and forwards to main API
"""
import asyncio
import aiohttp
import sys
from datetime import datetime
from typing import Optional

# ==================== CONFIGURATION ====================
# Change these values to match your setup

# Gas sensor endpoint (your existing gas.py script)
GAS_SENSOR_URL = "http://localhost:9000/data"

# Main backend API endpoint (running on different port to avoid conflict)
MAIN_API_URL = "http://localhost:8000/api/gas-sensors/readings"

# Login credentials (to get auth token)
USERNAME = "shift_day1"  # Change to your preferred user
PASSWORD = "shift123"
LOGIN_URL = "http://localhost:8000/auth/login"

# Mine/Zone/Gate configuration
# Run this to get your IDs:
#   mongo
#   use your_database_name
#   db.mines.find()
#   db.zones.find()
#   db.gates.find()
MINE_ID = "69374aa89607f42863c06cb7"
# Set to your mine ObjectId as string, e.g., "507f1f77bcf86cd799439011"
ZONE_ID = None  # Optional - set if sensor is in specific zone
GATE_ID = None  # Optional - set if sensor is at specific gate
SENSOR_ID = "GAS_SENSOR_001"  # Unique identifier for this sensor

# Polling interval (seconds)
POLL_INTERVAL = 10  # Read from sensors every 10 seconds

# CO sensor multiplier (if you have CO sensor on different analog pin)
# If you don't have a separate CO sensor, it will use 0
HAS_CO_SENSOR = False  # Set to True if you have CO sensor connected
CO_MULTIPLIER = 0.5  # Adjust based on your CO sensor calibration

# ==================== COLOR OUTPUT ====================
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_status(message: str, status: str = "info"):
    """Print colored status messages"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if status == "success":
        print(f"{Colors.GREEN}✓{Colors.RESET} [{timestamp}] {message}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠{Colors.RESET} [{timestamp}] {message}")
    elif status == "error":
        print(f"{Colors.RED}✗{Colors.RESET} [{timestamp}] {message}")
    elif status == "info":
        print(f"{Colors.CYAN}ℹ{Colors.RESET} [{timestamp}] {message}")
    else:
        print(f"  [{timestamp}] {message}")

# ==================== AUTHENTICATION ====================
async def get_auth_token(session: aiohttp.ClientSession) -> Optional[str]:
    """Login and get authentication token"""
    try:
        print_status("Logging in to get authentication token...", "info")
        async with session.post(
            LOGIN_URL,
            json={"username": USERNAME, "password": PASSWORD}
        ) as response:
            if response.status == 200:
                data = await response.json()
                token = data.get("access_token")
                print_status(f"Logged in as {USERNAME}", "success")
                return token
            else:
                error = await response.text()
                print_status(f"Login failed: {error}", "error")
                return None
    except Exception as e:
        print_status(f"Login error: {e}", "error")
        return None

# ==================== DATA FETCHING & SENDING ====================
async def fetch_sensor_data(session: aiohttp.ClientSession) -> Optional[dict]:
    """Fetch data from your gas sensor script"""
    try:
        async with session.get(GAS_SENSOR_URL, timeout=aiohttp.ClientTimeout(total=5)) as response:
            if response.status == 200:
                data = await response.json()
                if data.get("status") == "connected":
                    return data
                else:
                    return None
            else:
                return None
    except asyncio.TimeoutError:
        print_status("Timeout connecting to gas sensor", "warning")
        return None
    except Exception as e:
        print_status(f"Error fetching sensor data: {e}", "error")
        return None

async def send_to_backend(
    session: aiohttp.ClientSession,
    token: str,
    sensor_data: dict
) -> bool:
    """Send gas reading to main backend API"""
    try:
        # Extract data from sensor
        methane_ppm = sensor_data.get("methane_ppm", 0.0)

        # Calculate CO from methane if no separate sensor
        # (You can modify this based on your actual CO sensor)
        if HAS_CO_SENSOR:
            co_ppm = sensor_data.get("co_ppm", 0.0)
        else:
            # Estimate CO from methane (very rough approximation)
            # Replace with actual CO sensor reading when available
            co_ppm = methane_ppm * CO_MULTIPLIER if methane_ppm > 0 else 0.0

        env = sensor_data.get("env", {})

        # Prepare payload
        payload = {
            "methane_ppm": methane_ppm,
            "co_ppm": co_ppm,
            "pressure_hpa": env.get("pressure_hpa", 0.0),
            "altitude_m": env.get("altitude_m", 0.0),
            "sensor_id": SENSOR_ID,
        }

        # Add mine/zone/gate if configured
        if MINE_ID:
            payload["mine_id"] = MINE_ID
        if ZONE_ID:
            payload["zone_id"] = ZONE_ID
        if GATE_ID:
            payload["gate_id"] = GATE_ID

        # Send to API
        headers = {"Authorization": f"Bearer {token}"}
        async with session.post(
            MAIN_API_URL,
            params=payload,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                severity = result.get("severity", "normal")
                alert_created = result.get("alert_created", False)

                # Color-code based on severity
                status_color = {
                    "normal": "success",
                    "medium": "warning",
                    "high": "warning",
                    "critical": "error"
                }.get(severity, "info")

                alert_msg = f" | {Colors.RED}ALERT CREATED{Colors.RESET}" if alert_created else ""
                print_status(
                    f"CH₄: {methane_ppm:6.1f} PPM | CO: {co_ppm:5.1f} PPM | "
                    f"Severity: {severity.upper():8s}{alert_msg}",
                    status_color
                )
                return True
            else:
                error = await response.text()
                print_status(f"API error: {error}", "error")
                return False

    except Exception as e:
        print_status(f"Error sending data: {e}", "error")
        return False

# ==================== MAIN LOOP ====================
async def main_loop():
    """Main monitoring loop"""
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.CYAN}  GAS SENSOR BRIDGE - Main Backend Integration{Colors.RESET}")
    print("=" * 70)
    print(f"\n{Colors.BOLD}Configuration:{Colors.RESET}")
    print(f"  Gas Sensor URL:  {GAS_SENSOR_URL}")
    print(f"  Main API URL:    {MAIN_API_URL}")
    print(f"  Username:        {USERNAME}")
    print(f"  Mine ID:         {MINE_ID or 'Not configured - PLEASE SET THIS!'}")
    print(f"  Zone ID:         {ZONE_ID or 'Not set (optional)'}")
    print(f"  Gate ID:         {GATE_ID or 'Not set (optional)'}")
    print(f"  Sensor ID:       {SENSOR_ID}")
    print(f"  Poll Interval:   {POLL_INTERVAL} seconds")
    print(f"  CO Sensor:       {'Yes' if HAS_CO_SENSOR else 'No (using estimation)'}")
    print("\n" + "=" * 70 + "\n")

    # Check if MINE_ID is configured
    if not MINE_ID:
        print_status(
            "WARNING: MINE_ID not configured! Please set it in the script.",
            "error"
        )
        print(f"\n{Colors.YELLOW}To get your Mine ID:{Colors.RESET}")
        print("  1. Open MongoDB shell: mongo")
        print("  2. Use your database: use your_database_name")
        print("  3. Find mines: db.mines.find()")
        print("  4. Copy the _id value and paste it as MINE_ID in this script")
        print(f"\n{Colors.RED}Script will exit in 10 seconds...{Colors.RESET}\n")
        await asyncio.sleep(10)
        return

    print_status("Starting gas sensor bridge...", "info")
    print_status(f"Make sure your gas.py script is running on port 8000!", "info")
    print_status(f"Polling every {POLL_INTERVAL} seconds. Press Ctrl+C to stop.\n", "info")

    consecutive_failures = 0
    max_failures = 5

    async with aiohttp.ClientSession() as session:
        # Get authentication token
        token = await get_auth_token(session)
        if not token:
            print_status("Failed to authenticate. Exiting.", "error")
            return

        # Main monitoring loop
        while True:
            try:
                # Fetch sensor data
                sensor_data = await fetch_sensor_data(session)

                if sensor_data:
                    # Send to backend
                    success = await send_to_backend(session, token, sensor_data)

                    if success:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1

                # Check for too many failures
                if consecutive_failures >= max_failures:
                    print_status(
                        f"Failed {max_failures} times in a row. Checking sensor connection...",
                        "warning"
                    )
                    print_status(
                        "Is your gas.py script running on port 8000?",
                        "warning"
                    )
                    consecutive_failures = 0  # Reset counter

                # Wait before next poll
                await asyncio.sleep(POLL_INTERVAL)

            except KeyboardInterrupt:
                print(f"\n\n{Colors.YELLOW}Shutting down gracefully...{Colors.RESET}")
                break
            except Exception as e:
                print_status(f"Unexpected error: {e}", "error")
                await asyncio.sleep(5)

    print_status("Gas sensor bridge stopped.", "info")

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    try:
        # Check if running on Windows and set appropriate event loop policy
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}Goodbye!{Colors.RESET}\n")
    except Exception as e:
        print_status(f"Fatal error: {e}", "error")
        sys.exit(1)
