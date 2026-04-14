"""
Smart Helmet Service - Serial Reader and Data Processing
Reads sensor data from ESP32 helmet via serial port and provides REST API access.
Updated for SIH1_BIDIRECTIONAL_SENSOR.ino
"""
import serial
import asyncio
import re
import random
from datetime import datetime
from typing import Dict, Optional, Any
# Assuming 'database' module exists and provides get_database
try:
    from database import get_database
except ImportError:
    # Mocking for standalone testing if database.py is not provided
    print("[WARN] 'database' module not found. Alert storage will be disabled.")
    def get_database():
        class MockDB:
            async def insert_one(self, doc):
                # Mock result object with inserted_id
                return type('MockResult', (object,), {'inserted_id': 'mock_id_' + str(random.randint(1000, 9999))})()
            async def __aenter__(self): return self
            async def __aexit__(self, exc_type, exc_val, exc_tb): pass
        return MockDB()


# ================= CONFIGURATION =================
SERIAL_PORT = "/dev/ttyACM0"  # Change to your ESP32 port
BAUD_RATE = 115200
SERIAL_ENABLED = True  # Set to False to use mock data for testing

# ================= GLOBAL STATE (Restored) =================
# Store latest readings per helmet/worker
latest_helmet_data: Dict[str, Dict[str, Any]] = {}

# Global serial connection for bidirectional communication
_serial_connection: Optional[serial.Serial] = None

# Worker ID to Name mapping for demo helmets
DEMO_WORKER_NAMES = {
    "worker_1": "Rushabh Savla",
    "worker_2": "Tanush Maloo",
}

# For faking heart rate and SpO2 oscillation
_fake_hr_direction = 1
_fake_hr_value = 90
_fake_spo2_direction = 1
_fake_spo2_value = 91.0

# ================= ALERT THRESHOLDS (Updated to match Arduino ALARM_PPM) =================
THRESHOLDS = {
    "methane_ppm": {"high": 1000, "severity": "critical"}, 
    "battery_voltage": {"low": 3.32, "severity": "high"},
    "heart_rate_low": {"low": 50, "severity": "medium"},
    "heart_rate_high": {"high": 120, "severity": "medium"},
    "spo2": {"low": 90, "severity": "high"},
}


def get_fake_vitals() -> tuple:
    """
    Generate fake heart rate and SpO2 values that oscillate. (Restored)
    Heart rate: oscillates between 88-94 BPM
    SpO2: oscillates between 90.0-92.5%
    """
    global _fake_hr_direction, _fake_hr_value, _fake_spo2_direction, _fake_spo2_value

    # Heart rate oscillation (88-94)
    _fake_hr_value += _fake_hr_direction * random.randint(1, 2)
    if _fake_hr_value >= 94:
        _fake_hr_direction = -1
    elif _fake_hr_value <= 88:
        _fake_hr_direction = 1

    # SpO2 oscillation (90.0-92.5)
    _fake_spo2_value += _fake_spo2_direction * random.uniform(0.1, 0.3)
    if _fake_spo2_value >= 92.5:
        _fake_spo2_direction = -1
    elif _fake_spo2_value <= 90.0:
        _fake_spo2_direction = 1

    return _fake_hr_value, round(_fake_spo2_value, 1)


def parse_arduino_array(arr: list) -> Optional[Dict[str, Any]]:
    """
    Converts the raw integer list from Arduino into a named dictionary.
    
    New Packet Structure (Index 0 to 20):
    [0]sdMillis, [1-3]Acc, [4-6]Gyro, [7-9]RPY, [10]CH4*10, [11]CH4%*1e4, [12]Batt*1000, 
    [13]FSR, [14]FSRSt, [15]State, [16]SOS, [17]Man, [18]HR*10, [19]SpO2*10, [20]MQ7, [21]Dout
    """
    if len(arr) < 21:
        return None

    # --- Power ---
    # Index 12: Battery (mV -> divide by 1000)
    battery_voltage = round(arr[12] / 1000.0, 2)
    battery_low = battery_voltage < THRESHOLDS["battery_voltage"]["low"]

    # --- Gas ---
    # Index 10: CH4 (ppm * 10 -> divide by 10)
    methane_ppm = round(arr[10] / 10.0, 1)

    # --- Vitals ---
    # Index 18: Heart Rate (val * 10 -> divide by 10)
    # Index 19: SpO2 (val * 10 -> divide by 10)
    raw_hr = arr[18] / 10.0
    raw_spo2 = arr[19] / 10.0
    
    # Use real data if > 0, otherwise use faked data for stability in UI
    if raw_hr <= 0 or raw_spo2 <= 0:
        heart_rate, spo2 = get_fake_vitals()
    else:
        heart_rate = raw_hr
        spo2 = raw_spo2

    # --- Status ---
    # Index 16: SOS Latched (0 or 1)
    sos_active = arr[16] == 1
    system_state = arr[15]

    # --- Determine severity (Restored logic) ---
    severity = "normal"
    if sos_active or system_state == 5: # State 5 is SOS
        severity = "critical"
    elif methane_ppm > THRESHOLDS["methane_ppm"]["high"] or system_state == 2: # State 2 is Methane
        severity = "critical"
    elif battery_low or system_state == 3: # State 3 is Low Power
        severity = "high"
    elif spo2 < THRESHOLDS["spo2"]["low"]:
        severity = "high"
    elif heart_rate < THRESHOLDS["heart_rate_low"]["low"] or heart_rate > THRESHOLDS["heart_rate_high"]["high"]:
        severity = "medium"
    
    data_dict = {
        # Metadata
        "timestamp_ms": arr[0], # Arduino millis
        
        # Motion data
        "accel_x": arr[1],
        "accel_y": arr[2],
        "accel_z": arr[3],
        "gyro_x": arr[4],
        "gyro_y": arr[5],
        "gyro_z": arr[6],

        # Orientation (centi-degrees -> divide by 100)
        "roll": round(arr[7] / 100.0, 2),
        "pitch": round(arr[8] / 100.0, 2),
        "yaw": round(arr[9] / 100.0, 2),

        # Gas sensors
        "methane_ppm": methane_ppm,
        "co_raw": arr[20], # MQ7 Raw

        # Power
        "battery_voltage": battery_voltage,
        "battery_low": battery_low,

        # FSR sensor
        "fsr_force": arr[13],
        "fsr_state": "Touch" if arr[14] == 2 else "None",

        # Status
        "system_state": system_state,
        "sos_active": sos_active,
        "manual_override": arr[17] == 1,

        # Health vitals (faked/real)
        "heart_rate": heart_rate,
        "spo2": spo2,

        # API Metadata
        "severity": severity,
        "timestamp": datetime.utcnow(),
    }

    return data_dict


async def check_and_store_alerts(worker_id: str, data: Dict[str, Any], db) -> bool:
    """
    Check if any thresholds are exceeded and store alert in database. (Restored)
    """
    alert_messages = []
    severity = "normal"

    # Check methane
    if data["methane_ppm"] > THRESHOLDS["methane_ppm"]["high"]:
        alert_messages.append(f"CRITICAL: Methane level {data['methane_ppm']} PPM exceeds threshold ({THRESHOLDS['methane_ppm']['high']} PPM)")
        severity = "critical"

    # Check SOS
    if data["sos_active"]:
        alert_messages.append(f"CRITICAL: SOS activated by worker")
        severity = "critical"

    # Check battery
    if data["battery_low"]:
        alert_messages.append(f"WARNING: Low battery ({data['battery_voltage']}V < {THRESHOLDS['battery_voltage']['low']}V)")
        if severity == "normal": severity = "high"

    # Check SpO2
    if data["spo2"] < THRESHOLDS["spo2"]["low"]:
        alert_messages.append(f"WARNING: Low oxygen saturation ({data['spo2']}% < {THRESHOLDS['spo2']['low']}%)")
        if severity == "normal": severity = "high"

    # Check heart rate
    if data["heart_rate"] < THRESHOLDS["heart_rate_low"]["low"]:
        alert_messages.append(f"CAUTION: Low heart rate ({data['heart_rate']} BPM < {THRESHOLDS['heart_rate_low']['low']})")
        if severity == "normal": severity = "medium"
    elif data["heart_rate"] > THRESHOLDS["heart_rate_high"]["high"]:
        alert_messages.append(f"CAUTION: High heart rate ({data['heart_rate']} BPM > {THRESHOLDS['heart_rate_high']['high']})")
        if severity == "normal": severity = "medium"

    if alert_messages:
        # Store helmet reading in database
        reading_doc = {
            "worker_id": worker_id,
            "timestamp": data["timestamp"],
            "severity": severity,
            **{k: v for k, v in data.items() if k not in ["timestamp", "severity"]}
        }
        reading_result = await db.helmet_readings.insert_one(reading_doc)

        # Create alert
        alert_doc = {
            "alert_type": "helmet_sensor",
            "severity": severity,
            "status": "active",
            "message": " | ".join(alert_messages),
            "worker_id": worker_id,
            "helmet_reading_id": reading_result.inserted_id,
            "created_at": datetime.utcnow(),
            "details": {
                "methane_ppm": data["methane_ppm"],
                "battery_voltage": data["battery_voltage"],
                "heart_rate": data["heart_rate"],
                "spo2": data["spo2"],
                "sos_active": data["sos_active"],
            }
        }
        await db.alerts.insert_one(alert_doc)
        return True

    return False


def get_mock_helmet_data(worker_id: str = "worker_1") -> Dict[str, Any]:
    """Generate mock helmet data for testing without hardware. (Restored)"""
    hr, spo2 = get_fake_vitals()
    
    return {
        "worker_id": worker_id,
        "timestamp_ms": 0,
        "accel_x": random.randint(-100, 100),
        "accel_y": random.randint(-100, 100),
        "accel_z": random.randint(900, 1100),
        "gyro_x": random.randint(-50, 50),
        "gyro_y": random.randint(-50, 50),
        "gyro_z": random.randint(-50, 50),
        "roll": round(random.uniform(-5, 5), 2),
        "pitch": round(random.uniform(-5, 5), 2),
        "yaw": round(random.uniform(0, 360), 2),
        "methane_ppm": round(random.uniform(100, 500), 1),
        "co_raw": random.randint(50, 150),
        "battery_voltage": round(random.uniform(3.5, 4.2), 2),
        "battery_low": False,
        "fsr_force": random.randint(0, 100),
        "fsr_state": "None",
        "system_state": 1,
        "sos_active": False,
        "manual_override": False,
        "heart_rate": hr,
        "spo2": spo2,
        "severity": "normal",
        "timestamp": datetime.utcnow(),
    }


async def serial_reader():
    """Background task that reads from serial port and updates latest_helmet_data. (Restored)"""
    global latest_helmet_data, _serial_connection

    print(f"[Helmet Service] Starting serial reader on {SERIAL_PORT}...")
    ser = None
    last_error_time = 0
    import time

    # For demo: currently hardcoded to worker_1
    current_worker_id = "worker_1"

    while True:
        try:
            if not SERIAL_ENABLED:
                mock_data = get_mock_helmet_data(current_worker_id)
                mock_data["worker_id"] = current_worker_id
                latest_helmet_data[current_worker_id] = mock_data
                await asyncio.sleep(1)
                continue

            if ser is None:
                try:
                    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                    _serial_connection = ser  # Store globally for bidirectional communication
                    print(f"[Helmet Service] Serial connected to {SERIAL_PORT}")
                except Exception as e:
                    # Throttle error logging to once every 60 seconds
                    current_time = time.time()
                    if current_time - last_error_time > 60:
                        print(f"[Helmet Service] Serial connection failed (will retry): {e}")
                        last_error_time = current_time
                    
                    # Fall back to mock data
                    mock_data = get_mock_helmet_data(current_worker_id)
                    latest_helmet_data[current_worker_id] = mock_data
                    await asyncio.sleep(2)
                    continue

            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()

                # Filter for packet
                if line.startswith("INT32_PACKET:"):
                    match = re.search(r"\[(.*?)\]", line)
                    if match:
                        content = match.group(1)
                        raw_values = [int(float(x)) for x in content.split(',') if x.strip()]

                        # Parse to dictionary
                        parsed_data = parse_arduino_array(raw_values)

                        if parsed_data:
                            parsed_data["worker_id"] = current_worker_id
                            latest_helmet_data[current_worker_id] = parsed_data

                            # Check thresholds and store alerts if needed
                            try:
                                db = get_database()
                                # Only store alerts if it's the right worker ID or if the system needs it
                                await check_and_store_alerts(current_worker_id, parsed_data, db)
                            except Exception as e:
                                print(f"[Helmet Service] DB error: {e}")

            await asyncio.sleep(0.01)

        except Exception as e:
            print(f"[Helmet Service] Error: {e}")
            if ser:
                ser.close()
            ser = None
            await asyncio.sleep(1)


def get_latest_helmet_data(worker_id: Optional[str] = None) -> Dict[str, Any]:
    """Get the latest helmet data. (Restored)"""
    if worker_id:
        return latest_helmet_data.get(worker_id, {})
    # For NextJS page, usually returns all worker data
    return latest_helmet_data


def get_all_helmet_data() -> list:
    """Get latest data from all helmets as a list. (Restored)"""
    return list(latest_helmet_data.values())


async def start_helmet_reader():
    """Start the helmet serial reader as a background task. (Restored)"""
    # This is called once during FastAPI startup
    asyncio.create_task(serial_reader())


# ================= EVACUATION TRIGGER =================

def trigger_all_alarms() -> dict:
    """
    Send EVACUATE_ALL command to ESP32 via serial port.
    This triggers all helmet buzzers for emergency evacuation.

    Returns:
        dict with success status and message
    """
    global _serial_connection

    try:
        if _serial_connection and _serial_connection.is_open:
            _serial_connection.write(b"EVACUATE_ALL\n")
            print("[Helmet Service] EVACUATE_ALL command sent to ESP32")
            return {"success": True, "message": "Evacuation command sent to all helmets"}
        else:
            # Try to open a new connection if not available
            try:
                temp_ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                temp_ser.write(b"EVACUATE_ALL\n")
                temp_ser.close()
                print("[Helmet Service] EVACUATE_ALL command sent via new connection")
                return {"success": True, "message": "Evacuation command sent (new connection)"}
            except Exception as e:
                print(f"[Helmet Service] Failed to send evacuation command: {e}")
                # For demo purposes, return success even if serial is not connected
                return {"success": True, "message": "Evacuation command simulated (no serial connection)", "simulated": True}
    except Exception as e:
        print(f"[Helmet Service] Error triggering alarms: {e}")
        return {"success": False, "message": f"Failed to trigger alarms: {str(e)}"}