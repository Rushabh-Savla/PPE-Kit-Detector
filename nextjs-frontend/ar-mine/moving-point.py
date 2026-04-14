import serial
import threading
import time
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ================= CONFIGURATION =================
# ESP32-S3 usually appears as /dev/ttyACM0 on Linux
SERIAL_PORT = "/dev/ttyACM0" 
BAUD_RATE = 115200

# ================= GLOBAL VARIABLES =================
latest_data = {
    "status": "waiting_for_data",
    "timestamp_ms": 0,
    "accel": {"x_g": 0, "y_g": 0, "z_g": 0},
    "gyro": {"x_dps": 0, "y_dps": 0, "z_dps": 0},
    "orientation": {"roll": 0, "pitch": 0, "yaw": 0}, 
    "gas": {
        "methane_ppm": 0.0,
        "co_raw": 0,
        "co_alert": False
    },
    "health": {
        "hr_bpm": 0.0,
        "spo2_pct": 0.0
    },
    "battery_v": 0.0,
    "fsr": {"raw": 0, "is_wearing_helmet": False},
    "system_state": 1,
    "flags": {"sos": False, "manual": False},
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
    
    # Regex to extract the array content: INT32_PACKET: [ ... ]
    packet_pattern = re.compile(r"INT32_PACKET:\s*\[(.*?)\]")
    
    # Regex for environmental data line (P=... ALT=...) if still present
    # (The new Arduino code doesn't explicitly print the env line separately in loop, 
    # but we will keep this just in case, or rely on internal sensors if merged)
    # NOTE: Your new Arduino code sends P/ALT/Temp only via Serial in printCombinedBlock? 
    # Actually, looking at your Arduino code, it sends "INT32_PACKET" AND prints "P=" lines?
    # No, the new code ONLY prints INT32_PACKET. 
    # WAIT! The Arduino code calculates Alt/Pres but does NOT put them in the INT32_PACKET.
    # It only puts them in `latest` struct. 
    # **Correction based on your Arduino code**: You removed the text print of P= and ALT=. 
    # However, you *need* them. I will assume for now we only have the INT32 packet.
    # If you need P/Alt, you must add them to the INT32 packet in Arduino or print the text line again.
    # **Fix for now**: I will leave fields 0, but they won't update unless you add them to the packet.
    
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
                
                # Parse INT32 Packet
                match_pkt = packet_pattern.search(line)
                if match_pkt:
                    raw_str = match_pkt.group(1)
                    try:
                        val = [int(x.strip()) for x in raw_str.split(',')]
                        # The new packet has 22 values
                        if len(val) >= 22:
                            with data_lock:
                                latest_data["timestamp_ms"] = val[0]
                                # 1-3 Accel
                                latest_data["accel"] = {"x_g": val[1]/1000.0, "y_g": val[2]/1000.0, "z_g": val[3]/1000.0}
                                # 4-6 Gyro
                                latest_data["gyro"] = {"x_dps": val[4]/1000.0, "y_dps": val[5]/1000.0, "z_dps": val[6]/1000.0}
                                # 7-9 Orientation
                                latest_data["orientation"] = {"roll": val[7]/100.0, "pitch": val[8]/100.0, "yaw": val[9]/100.0}
                                # 10 Methane PPM
                                latest_data["gas"]["methane_ppm"] = val[10]/10.0
                                # 11 is Percent (skipped)
                                # 12 Battery
                                latest_data["battery_v"] = val[12]/1000.0
                                # 13-14 FSR
                                latest_data["fsr"] = {"raw": val[13], "is_wearing_helmet": val[14] == 2}
                                # 15 State
                                latest_data["system_state"] = val[15]
                                # 16-17 Flags
                                latest_data["flags"]["sos"] = (val[16] == 1)
                                latest_data["flags"]["manual"] = (val[17] == 1)
                                # 18-19 Health (MAX30105)
                                latest_data["health"]["hr_bpm"] = val[18]/10.0
                                latest_data["health"]["spo2_pct"] = val[19]/10.0
                                # 20-21 MQ7 (CO)
                                latest_data["gas"]["co_raw"] = val[20]
                                latest_data["gas"]["co_alert"] = (val[21] == 0) # usually digital low = alarm
                                
                                latest_data["status"] = "connected"
                    except ValueError: pass

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

@app.post("/command/{cmd}")
def send_command(cmd: str):
    """
    'c': Calibrate
    '0': Auto
    '1'-'5': Manual States (5=SOS)
    """
    if ser and ser.is_open:
        try:
            ser.write(cmd.encode('utf-8'))
            return {"status": "sent", "command": cmd}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write: {e}")
    return {"status": "error", "detail": "Serial not open"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)