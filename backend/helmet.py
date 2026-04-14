import serial
import asyncio
import re
import uvicorn
import json
import time # Added for throttling logic
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# ================= CONFIGURATION =================
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
# Throttle WebSocket broadcasts to prevent flooding the client (e.g., 500ms)
BROADCAST_INTERVAL_MS = 500

app = FastAPI()

# Add CORS middleware for REST API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ================= HTML DASHBOARD (No change needed) =================
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>SIH Bidirectional Helmet</title>
        <style>
            body { font-family: 'Segoe UI', monospace; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }
            h1 { text-align: center; color: #38bdf8; }
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; max-width: 1400px; margin: 0 auto; }
            .card { background: #1e293b; border-left: 5px solid #3b82f6; border-radius: 8px; padding: 20px; display: flex; justify-content: space-between; align-items: center; }
            .label { font-size: 14px; color: #94a3b8; text-transform: uppercase; }
            .value { font-size: 24px; font-weight: bold; color: #f8fafc; }
            .card.alert { border-left-color: #ef4444; } 
            .card.safe { border-left-color: #22c55e; }
            #json-viewer { background: #000; color: #0f0; padding: 20px; margin-top: 30px; border-radius: 8px; font-family: monospace; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>Helmet Sensor Live (New Protocol)</h1>
        <div id="dashboard" class="grid"></div>
        <div id="json-viewer">Waiting for JSON data...</div>

        <script>
            var ws = new WebSocket("ws://" + window.location.host + "/ws");
            ws.onmessage = function(event) {
                var data = JSON.parse(event.data);
                document.getElementById("json-viewer").innerText = JSON.stringify(data, null, 4);
                var container = document.getElementById("dashboard");
                container.innerHTML = "";
                
                for (const [key, value] of Object.entries(data)) {
                    let card = document.createElement("div");
                    card.className = "card";
                    if (key === "SOS" && value === true) card.classList.add("alert");
                    // Using the new, lower methane alarm threshold of 1000
                    else if (key.includes("Methane") && value > 1000) card.classList.add("alert"); 
                    else card.classList.add("safe");

                    let labelSpan = document.createElement("span");
                    labelSpan.className = "label";
                    labelSpan.innerText = key;
                    let valueSpan = document.createElement("span");
                    valueSpan.className = "value";
                    valueSpan.innerText = value;

                    card.appendChild(labelSpan);
                    card.appendChild(valueSpan);
                    container.appendChild(card);
                }
            };
        </script>
    </body>
</html>
"""

# ================= BACKEND LOGIC =================
class ConnectionManager:
    def __init__(self): self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket): await websocket.accept(); self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket): self.active_connections.remove(websocket)
    async def broadcast(self, message: str): 
        for connection in self.active_connections: 
            try:
                await connection.send_text(message)
            except Exception:
                # Handle connection loss gracefully
                self.disconnect(connection)

manager = ConnectionManager()

# Store latest sensor data for REST API
latest_sensor_data = {}

@app.get("/")
async def get(): return HTMLResponse(html)

@app.get("/api/sensor-data")
async def get_sensor_data():
    """REST endpoint to get latest helmet sensor data"""
    return latest_sensor_data

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: manager.disconnect(websocket)

def parse_arduino_array(arr):
    """
    Parses the array from the new SIH1_BIDIRECTIONAL_SENSOR.ino sketch.
    """
    if len(arr) < 21: 
        return {"error": "Incomplete packet", "len": len(arr), "raw_array": arr}

    # MAPPING FOR: SIH1_BIDIRECTIONAL_SENSOR.ino
    # [0]sdMillis, [1-3]Acc, [4-6]Gyro, [7-9]RPY, [10]CH4*10, [11]CH4%*1e4, [12]Batt*1000, 
    # [13]FSR, [14]FSRSt, [15]State, [16]SOS, [17]Man, [18]HR*10, [19]SpO2*10, [20]MQ7_raw, [21]MQ7_dout
    
    data_dict = {
        "Timestamp (ms)": arr[0],
        "Accel X (mg)": arr[1], "Accel Y (mg)": arr[2], "Accel Z (mg)": arr[3],
        "Gyro X (mdps)":  arr[4], "Gyro Y (mdps)":  arr[5], "Gyro Z (mdps)":  arr[6],
        
        # Angles divided by 100
        "Roll (Deg)":  round(arr[7] / 100.0, 2),
        "Pitch (Deg)": round(arr[8] / 100.0, 2),
        "Yaw (Deg)":   round(arr[9] / 100.0, 2),
        
        # Methane divided by 10
        "Methane (PPM)": round(arr[10] / 10.0, 1),
        
        # Battery divided by 1000
        "Battery (V)":   round(arr[12] / 1000.0, 2),
        
        "FSR Force": arr[13],
        "FSR State": "Touch" if arr[14] == 2 else "None",
        "System State": arr[15],
        "SOS": True if arr[16] == 1 else False,
        "Manual Override": True if arr[17] == 1 else False,
        
        # HR and SpO2 divided by 10
        "Heart Rate (BPM)": round(arr[18] / 10.0, 1),
        "SpO2 (%)":         round(arr[19] / 10.0, 1),
        
        "CO Sensor (MQ7)": arr[20],
        "MQ7 Dout": arr[21]
    }
    return data_dict

async def serial_reader():
    print(f"Connecting to {SERIAL_PORT}...")
    ser = None
    last_broadcast_time = 0 # Throttling timestamp

    while True:
        try:
            if ser is None:
                try:
                    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                    print("Serial Connected!")
                except Exception:
                    await asyncio.sleep(2)
                    continue

            # Check for data in buffer
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Check for the start of your packet
                if line.startswith("INT32_PACKET:"):
                    match = re.search(r"\[(.*?)\]", line)
                    if match:
                        content = match.group(1)
                        # Split and filter empty strings
                        raw_values = [int(float(x)) for x in content.split(',') if x.strip()]
                        
                        final_data = parse_arduino_array(raw_values)
                        
                        # --- THROTTLING LOGIC ---
                        current_time_ms = time.time() * 1000
                        
                        if current_time_ms - last_broadcast_time >= BROADCAST_INTERVAL_MS:
                            # Store latest data for REST API
                            global latest_sensor_data
                            latest_sensor_data = final_data
                            # Broadcast the latest data to all connected WebSockets
                            await manager.broadcast(json.dumps(final_data))
                            last_broadcast_time = current_time_ms
            
            # Use a very short sleep to keep the loop responsive
            await asyncio.sleep(0.001)
            
        except Exception as e:
            print(f"Serial Error: {e}")
            if ser: ser.close()
            ser = None
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    # Start the serial reader in the background
    asyncio.create_task(serial_reader())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)