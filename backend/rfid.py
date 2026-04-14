import serial
import threading
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import List

# ================= CONFIGURATION =================
# 🔴 CHANGE THIS to your ESP32 Port
SERIAL_PORT = "COM5"  
BAUD_RATE = 115200

# ================= GLOBAL STATE =================
current_state = {
    "scanning": False,
    "gate": "CLOSED",
    "result": "WAITING",
    "ppe": {"helmet": False, "vest": False, "boots": False},
    "log": "System Ready"
}

# Global reference to the main event loop
main_loop = None

# ================= SERIAL CONNECTION =================
ser = None
try:
    ser = serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=1,
        write_timeout=1,  # Add write timeout to prevent hanging
        rtscts=False,     # Disable hardware flow control
        dsrdtr=False      # Disable DSR/DTR flow control
    )
    print(f"✅ Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"❌ Error connecting to serial: {e}")

# ================= WEBSOCKET MANAGER =================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()
app = FastAPI()

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= BACKGROUND SERIAL READER =================
def read_from_serial():
    global current_state, main_loop
    print("🧵 Serial Listener Thread Started")
    
    while True:
        if ser and ser.in_waiting > 0:
            try:
                raw = ser.readline()
                line = raw.decode('utf-8', errors='ignore').strip()
                if not line: continue
                
                print(f"📥 ESP32: {line}")
                
                updated = False
                current_state["log"] = line

                # --- PARSING LOGIC ---
                if "Scanning Started" in line:
                    current_state["scanning"] = True
                    current_state["result"] = "SCANNING"
                    current_state["ppe"] = {"helmet": False, "vest": False, "boots": False}
                    updated = True
                elif "Helmet Scanned" in line:
                    current_state["ppe"]["helmet"] = True
                    updated = True
                elif "Vest Scanned" in line:
                    current_state["ppe"]["vest"] = True
                    updated = True
                elif "Boots Scanned" in line:
                    current_state["ppe"]["boots"] = True
                    updated = True
                elif "COMPLIANT" in line:
                    current_state["scanning"] = False
                    current_state["result"] = "PASS"
                    current_state["gate"] = "OPEN"
                    updated = True
                elif "NON-COMPLIANT" in line:
                    current_state["scanning"] = False
                    current_state["result"] = "FAIL"
                    current_state["gate"] = "CLOSED"
                    updated = True
                elif "Press '1'" in line:
                    current_state["scanning"] = False
                    current_state["result"] = "WAITING"
                    current_state["gate"] = "CLOSED"
                    updated = True
                
                # Push update to UI via WebSockets
                if updated and main_loop and main_loop.is_running():
                    asyncio.run_coroutine_threadsafe(manager.broadcast(current_state), main_loop)

            except Exception as e:
                print(f"Error reading serial: {e}")

# ================= FASTAPI EVENTS & ENDPOINTS =================

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    # Start the serial thread
    t = threading.Thread(target=read_from_serial, daemon=True)
    t.start()

@app.get("/")
async def get():
    # Reads the separate HTML file and serves it
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    await websocket.send_json(current_state)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/start-scan")
async def start_scan():
    global current_state
    print("📥 /start-scan endpoint called")

    # 1. Immediate UI Feedback
    current_state["scanning"] = True
    current_state["result"] = "SCANNING"
    current_state["ppe"] = {"helmet": False, "vest": False, "boots": False}
    current_state["log"] = "Command Sent..."

    print("📡 Broadcasting state update...")
    await manager.broadcast(current_state)
    print("✅ Broadcast complete")

    # 2. Send Command to ESP32
    print(f"🔍 Serial check: ser={ser is not None}, is_open={ser.is_open if ser else 'N/A'}")

    if ser and ser.is_open:
        try:
            bytes_written = ser.write(b"1")  # Send just "1" with no line ending
            ser.flush()      # Force send immediately
            print(f"📤 Sent '1' to ESP32 ({bytes_written} bytes written)")
        except Exception as e:
            print(f"❌ Error writing to serial: {e}")
            return {"status": "error", "message": str(e)}
    else:
        print("❌ Serial port not connected")
        return {"status": "error", "message": "Serial port not connected"}

    print("✅ Returning OK")
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)