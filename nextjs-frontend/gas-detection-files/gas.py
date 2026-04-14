import serial
import threading
import time
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ================= CONFIGURATION =================
# Linux: "/dev/ttyUSB0" or "/dev/ttyACM0" | Windows: "COM3"
SERIAL_PORT = "COM3"
BAUD_RATE = 115200

# ================= GLOBAL VARIABLES =================
latest_data = {
    "status": "waiting_for_data",
    "timestamp": 0,
    "methane_ppm": 0.0,
    "battery_v": 0.0,
    "accel": {"x_g": 0, "y_g": 0, "z_g": 0},
    "gyro": {"x_dps": 0, "y_dps": 0, "z_dps": 0},
    "fsr": {"raw": 0, "is_wearing_helmet": False},
    "system_state": 0,
    # NEW FIELDS
    "env": {
        "pressure_hpa": 0.0,
        "altitude_m": 0.0,
        "delta_alt_m": 0.0
    }
}
data_lock = threading.Lock()
ser = None

# ================= SERIAL READER =================
def serial_reader_loop():
    global ser, latest_data
    print(f"--- Attempting to open {SERIAL_PORT} ---")
    
    # Regex for the packet
    packet_pattern = re.compile(r"INT32_PACKET:\s*\[(.*?)\]")
    
    # Regex for environmental data (Matches "P=1013.25hPa", "ALT=12.50m", etc)
    # The Δ char might vary by encoding, so we match "ALT=" followed by "m" at the end of line
    pres_pattern = re.compile(r"P=([\d\.]+)hPa")
    alt_pattern = re.compile(r"ALT=([\d\.-]+)m")
    # Matches the last "m" value which is delta altitude
    delta_pattern = re.compile(r"(?:Δ|D)ALT=([\d\.-]+)m") 

    while True:
        try:
            if ser is None or not ser.is_open:
                try:
                    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                    print(f"SUCCESS: Connected to {SERIAL_PORT}")
                    ser.reset_input_buffer()
                except serial.SerialException:
                    time.sleep(2)
                    continue

            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                except:
                    continue
                
                # 1. Parse INT32 Packet (Sensors)
                match_pkt = packet_pattern.search(line)
                if match_pkt:
                    raw_str = match_pkt.group(1)
                    try:
                        val = [int(x.strip()) for x in raw_str.split(',')]
                        if len(val) == 12:
                            with data_lock:
                                latest_data["timestamp_ms"] = val[0]
                                latest_data["accel"] = {"x_g": val[1]/1000.0, "y_g": val[2]/1000.0, "z_g": val[3]/1000.0}
                                latest_data["gyro"] = {"x_dps": val[4]/1000.0, "y_dps": val[5]/1000.0, "z_dps": val[6]/1000.0}
                                latest_data["methane_ppm"] = val[7]/10.0
                                latest_data["battery_v"] = val[8]/1000.0
                                latest_data["fsr"] = {"raw": val[9], "is_wearing_helmet": val[10] == 2}
                                latest_data["system_state"] = val[11]
                                latest_data["status"] = "connected"
                    except ValueError: pass

                # 2. Parse Text Line (Pressure/Altitude)
                # Line looks like: AX=... P=1013.25hPa T=25.00C ALT=10.00m ΔALT=1.20m
                if "P=" in line and "ALT=" in line:
                    p_match = pres_pattern.search(line)
                    a_match = alt_pattern.search(line)
                    d_match = delta_pattern.search(line)
                    
                    with data_lock:
                        if p_match: latest_data["env"]["pressure_hpa"] = float(p_match.group(1))
                        if a_match: latest_data["env"]["altitude_m"] = float(a_match.group(1))
                        if d_match: latest_data["env"]["delta_alt_m"] = float(d_match.group(1))

        except Exception as e:
            print(f"Serial Error: {e}")
            if ser: ser.close()
            time.sleep(1)

# ================= SERVER SETUP =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=serial_reader_loop, daemon=True)
    t.start()
    yield
    if ser and ser.is_open: ser.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/data")
def get_data():
    with data_lock: return latest_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)