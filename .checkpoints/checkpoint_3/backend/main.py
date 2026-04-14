"""
Main FastAPI application for PPE detection, face recognition, and attendance system.
With Role-Based Access Control (RBAC) for mine safety management.
"""
import os
import base64
from io import BytesIO
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from bson import ObjectId
from dotenv import load_dotenv
import asyncio
import json

from detector import PersonDetector
from video_stream import get_video_processor
from database import connect_to_mongodb, close_mongodb_connection, get_database
from auth import (
    get_password_hash, verify_password, create_access_token,
    get_current_user, get_current_admin, verify_token, UserRole
)
from schemas import (
    EmployeeCreate, EmployeeResponse, AttendanceRecord,
    PPEViolation, AdminLogin, DashboardStats, ShiftType
)

# Import route modules
from routes.auth import router as auth_router
from routes.users import router as users_router
from routes.workers import router as workers_router
from routes.mines import router as mines_router
from routes.gate_entries import router as gate_entries_router
from routes.alerts import router as alerts_router
from routes.dashboards import router as dashboards_router
from routes.gas_sensors import router as gas_sensors_router
from routes.predictions import router as predictions_router
from routes.sos_alerts import router as sos_alerts_router
from routes.danger_zones import router as danger_zones_router
from routes.sms import router as sms_router
from routes.helmet import router as helmet_router
from reports import reports_router
from services.helmet_service import start_helmet_reader

# Load .env from backend directory
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)
DEFAULT_CAMERA = int(os.getenv("CAMERA_SOURCE", 0))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    await connect_to_mongodb()
    await initialize_default_superadmin()

    # Start ML prediction scheduler
    try:
        from ml.scheduler import start_scheduler, scheduler
        db = get_database()
        start_scheduler(db)

        # Register report scheduler
        try:
            from reports.report_scheduler import register_report_scheduler
            register_report_scheduler(scheduler, db)
        except Exception as e:
            print(f"Warning: Failed to register report scheduler: {e}")
    except Exception as e:
        print(f"Warning: Failed to start ML scheduler: {e}")

    # Start helmet serial reader
    try:
        asyncio.create_task(start_helmet_reader())
        print("Helmet serial reader started")
    except Exception as e:
        print(f"Warning: Failed to start helmet reader: {e}")

    yield
    await close_mongodb_connection()


app = FastAPI(
    title="Mine Safety PPE & Attendance System API",
    description="Role-based access control system for mine safety management with PPE detection",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


# Custom exception handler to ensure CORS headers on errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with CORS headers."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={"Access-Control-Allow-Origin": "*"}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all exceptions and ensure CORS headers are included."""
    import traceback
    print(f"Unhandled exception: {exc}")
    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"}
    )

# Initialize detector
detector = PersonDetector()


async def initialize_default_superadmin():
    """Create default super admin if none exists."""
    db = get_database()

    # Check if any super admin exists
    admin = await db.users.find_one({"role": UserRole.SUPER_ADMIN.value})
    if not admin:
        default_password = os.getenv("ADMIN_PASSWORD", "admin123")
        await db.users.insert_one({
            "username": "superadmin",
            "password_hash": get_password_hash(default_password),
            "full_name": "System Administrator",
            "email": "admin@system.local",
            "role": UserRole.SUPER_ADMIN.value,
            "is_active": True,
            "created_at": datetime.utcnow()
        })
        print("Default super admin created (username: superadmin)")

    # Create test mine and gate if none exists
    await initialize_test_mine_and_gate(db)


async def initialize_test_mine_and_gate(db):
    """Create a test mine and gate for development/testing."""
    # Check if any mine exists
    existing_mine = await db.mines.find_one({})
    if existing_mine:
        return  # Already have mines, don't create test data

    print("Creating test mine and gate for development...")

    # Create test mine
    mine_doc = {
        "name": "Test Mine",
        "location": "Development Environment",
        "description": "A test mine for development and testing purposes",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "created_by": "system",
    }
    mine_result = await db.mines.insert_one(mine_doc)
    mine_id = mine_result.inserted_id
    print(f"Created test mine: {mine_id}")

    # Create a test zone
    zone_doc = {
        "mine_id": mine_id,
        "name": "Main Zone",
        "description": "Primary work zone",
        "risk_level": "normal",
        "coordinates": {"x": 0, "y": 0, "width": 100, "height": 100},
        "created_at": datetime.utcnow(),
        "created_by": "system",
    }
    zone_result = await db.zones.insert_one(zone_doc)
    zone_id = zone_result.inserted_id
    print(f"Created test zone: {zone_id}")

    # Create test gate
    gate_doc = {
        "mine_id": mine_id,
        "name": "Main Entrance Gate",
        "gate_type": "both",
        "zone_id": zone_id,
        "location": "Main entrance of the test mine",
        "has_camera": True,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "created_by": "system",
    }
    gate_result = await db.gates.insert_one(gate_doc)
    print(f"Created test gate: {gate_result.inserted_id}")

    # Create a second gate (exit only)
    gate_doc_2 = {
        "mine_id": mine_id,
        "name": "Emergency Exit Gate",
        "gate_type": "exit",
        "zone_id": zone_id,
        "location": "Emergency exit point",
        "has_camera": True,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "created_by": "system",
    }
    gate_result_2 = await db.gates.insert_one(gate_doc_2)
    print(f"Created test gate 2: {gate_result_2.inserted_id}")

    # Create a test worker
    worker_doc = {
        "employee_id": "TEST001",
        "name": "Test Worker",
        "password_hash": get_password_hash("worker123"),
        "mine_id": mine_id,
        "zone_id": zone_id,
        "assigned_shift": "day",
        "face_registered": False,
        "is_active": True,
        "compliance_score": 100.0,
        "total_violations": 0,
        "badges": [],
        "created_at": datetime.utcnow(),
    }
    worker_result = await db.workers.insert_one(worker_doc)
    print(f"Created test worker: {worker_result.inserted_id} (employee_id: TEST001)")

    print("Test mine, zone, gates, and worker created successfully!")


# ==================== Include Route Modules ====================

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(workers_router)
app.include_router(mines_router)
app.include_router(gate_entries_router)
app.include_router(alerts_router)
app.include_router(dashboards_router)
app.include_router(gas_sensors_router)
app.include_router(predictions_router)
app.include_router(sos_alerts_router)
app.include_router(danger_zones_router)
app.include_router(sms_router)
app.include_router(helmet_router)
app.include_router(reports_router)


# ==================== WebSocket Test ====================

@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    """Simple test WebSocket endpoint."""
    await websocket.accept()
    await websocket.send_json({"message": "Hello from WebSocket!"})
    await websocket.close()


# ==================== Health Check ====================

@app.get("/")
def root():
    return {
        "message": "Mine Safety PPE & Attendance System API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Role-based access control (6 roles)",
            "Mine/Zone/Gate management",
            "PPE detection at gates",
            "Real-time compliance monitoring",
            "Worker management with compliance scores",
            "Alert and warning system",
            "Role-specific dashboards"
        ]
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/roles")
def get_roles():
    """Get available roles and their descriptions."""
    return {
        "roles": [
            {
                "value": UserRole.SUPER_ADMIN.value,
                "label": "Super Admin",
                "description": "Full system access, manages all configurations"
            },
            {
                "value": UserRole.GENERAL_MANAGER.value,
                "label": "General Manager",
                "description": "Organization-wide view, KPIs, regulatory compliance"
            },
            {
                "value": UserRole.AREA_SAFETY_OFFICER.value,
                "label": "Area Safety Officer",
                "description": "Oversees multiple mines, comparison analytics"
            },
            {
                "value": UserRole.MANAGER.value,
                "label": "Manager",
                "description": "Mine-level management, shift performance, approvals"
            },
            {
                "value": UserRole.SAFETY_OFFICER.value,
                "label": "Safety Officer",
                "description": "Safety enforcement, compliance analytics, PPE rules"
            },
            {
                "value": UserRole.SHIFT_INCHARGE.value,
                "label": "Shift Incharge",
                "description": "Live gate monitoring, entry overrides, shift reports"
            },
            {
                "value": UserRole.WORKER.value,
                "label": "Worker",
                "description": "Personal compliance, violations, shift timing"
            },
        ]
    }


@app.get("/api/shifts")
def get_shifts():
    """Get shift definitions."""
    return {
        "shifts": [
            {
                "value": ShiftType.DAY.value,
                "label": "Day Shift",
                "start_time": "06:00",
                "end_time": "14:00"
            },
            {
                "value": ShiftType.AFTERNOON.value,
                "label": "Afternoon Shift",
                "start_time": "14:00",
                "end_time": "22:00"
            },
            {
                "value": ShiftType.NIGHT.value,
                "label": "Night Shift",
                "start_time": "22:00",
                "end_time": "06:00"
            },
        ]
    }


# ==================== Legacy Endpoints (for backward compatibility) ====================

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    mark_attendance: bool = Form(default=True),
    location: Optional[str] = Form(default=None)
):
    """
    Upload an image and detect:
    - PPE equipment (helmet, vest, etc.)
    - Faces and identify known persons
    - Mark attendance if face is recognized (optional)
    """
    try:
        contents = await file.read()
        result_image, detections = detector.process_image(contents)

        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        attendance_marked = False
        attendance_record = None
        db = get_database()

        # Mark attendance if face is identified and mark_attendance is True
        if mark_attendance:
            identified_persons = detections.get("summary", {}).get("identified_persons", [])
            identified_names = detections.get("summary", {}).get("identified_names", [])

            if identified_persons:
                employee_id = identified_persons[0]
                employee_name = identified_names[0] if identified_names else None

                # Try to find worker
                worker = await db.workers.find_one({"employee_id": employee_id})
                if not worker:
                    worker = await db.employees.find_one({"employee_id": employee_id})

                if worker:
                    employee_name = worker.get("name", employee_name)

                # Get violations from detection
                violations = detections.get("violations", [])
                violation_labels = [v.get("label", "Unknown") for v in violations]

                # Create attendance record
                attendance_doc = {
                    "employee_id": employee_id,
                    "employee_name": employee_name,
                    "worker_id": str(worker["_id"]) if worker and "_id" in worker else None,
                    "type": "check_in",
                    "timestamp": datetime.utcnow(),
                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "location": location,
                    "ppe_compliant": len(violations) == 0,
                    "violations": violation_labels,
                    "image": f"data:image/png;base64,{img_base64}"
                }

                await db.attendance.insert_one(attendance_doc)
                attendance_marked = True
                attendance_record = {
                    "employee_id": employee_id,
                    "employee_name": employee_name,
                    "timestamp": attendance_doc["timestamp"].isoformat(),
                    "ppe_compliant": attendance_doc["ppe_compliant"],
                    "violations": violation_labels
                }

        return JSONResponse({
            "success": True,
            "image": f"data:image/png;base64,{img_base64}",
            "detections": detections,
            "attendance_marked": attendance_marked,
            "attendance": attendance_record
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.post("/detect-and-log")
async def detect_and_log(
    file: UploadFile = File(...),
    log_violations: bool = Form(default=True),
    mark_attendance: bool = Form(default=True),
    location: Optional[str] = Form(default=None)
):
    """
    Detect PPE and faces, automatically log violations and mark attendance.
    """
    try:
        contents = await file.read()
        result_image, detections = detector.process_image(contents)

        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        violations_logged = False
        attendance_marked = False
        attendance_record = None
        db = get_database()

        # Get identified person info
        identified_persons = detections.get("summary", {}).get("identified_persons", [])
        identified_names = detections.get("summary", {}).get("identified_names", [])
        employee_id = identified_persons[0] if identified_persons else None
        employee_name = identified_names[0] if identified_names else None
        worker = None
        worker_id = None

        if employee_id:
            # Try workers collection first
            worker = await db.workers.find_one({"employee_id": employee_id})
            if worker:
                employee_name = worker["name"]
                worker_id = str(worker["_id"])
            else:
                # Fall back to employees collection
                emp = await db.employees.find_one({"employee_id": employee_id})
                if emp:
                    employee_name = emp["name"]
                    worker_id = str(emp["_id"])

        # Get violations
        violations = detections.get("violations", [])
        violation_labels = [v.get("label", "Unknown") for v in violations]

        # Log violations if any exist and logging is enabled
        if log_violations and violations:
            violation_record = {
                "employee_id": employee_id,
                "employee_name": employee_name,
                "worker_id": worker_id,
                "violations": violations,
                "timestamp": datetime.utcnow(),
                "location": location,
                "image": f"data:image/png;base64,{img_base64}"
            }

            await db.ppe_violations.insert_one(violation_record)
            violations_logged = True

        # Mark attendance if face is identified
        if mark_attendance and employee_id:
            attendance_doc = {
                "employee_id": employee_id,
                "employee_name": employee_name,
                "worker_id": worker_id,
                "type": "check_in",
                "timestamp": datetime.utcnow(),
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "location": location,
                "ppe_compliant": len(violations) == 0,
                "violations": violation_labels,
                "image": f"data:image/png;base64,{img_base64}"
            }

            await db.attendance.insert_one(attendance_doc)
            attendance_marked = True
            attendance_record = {
                "employee_id": employee_id,
                "employee_name": employee_name,
                "timestamp": attendance_doc["timestamp"].isoformat(),
                "ppe_compliant": attendance_doc["ppe_compliant"],
                "violations": violation_labels
            }

        return JSONResponse({
            "success": True,
            "image": f"data:image/png;base64,{img_base64}",
            "detections": detections,
            "violations_logged": violations_logged,
            "attendance_marked": attendance_marked,
            "attendance": attendance_record
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


# ==================== Legacy Employee Management ====================

@app.post("/employees")
async def create_employee(
    name: str = Form(...),
    employee_id: str = Form(...),
    department: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(default=None),
    current_admin: dict = Depends(get_current_admin)
):
    """Create a new employee (legacy endpoint)."""
    db = get_database()

    existing = await db.employees.find_one({"employee_id": employee_id})
    if existing:
        raise HTTPException(status_code=400, detail="Worker ID already exists.")

    employee_data = {
        "name": name,
        "employee_id": employee_id,
        "department": department,
        "created_at": datetime.utcnow(),
        "face_registered": False
    }

    if file and file.filename:
        contents = await file.read()
        if contents:
            # Pass the worker's actual name as display_name for face recognition
            success = detector.register_face(employee_id, contents, name)
            if success:
                employee_data["face_registered"] = True
            else:
                raise HTTPException(status_code=400, detail="No face detected in the uploaded image.")

    result = await db.employees.insert_one(employee_data)

    return {
        "success": True,
        "employee": {
            "id": str(result.inserted_id),
            "name": name,
            "employee_id": employee_id,
            "department": department,
            "created_at": employee_data["created_at"].isoformat(),
            "face_registered": employee_data["face_registered"]
        },
        "message": f"Worker {name} registered successfully"
    }


@app.get("/employees")
async def list_employees(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = None
):
    """List all employees (legacy endpoint)."""
    db = get_database()

    query = {}
    if search:
        query = {
            "$or": [
                {"name": {"$regex": search, "$options": "i"}},
                {"employee_id": {"$regex": search, "$options": "i"}},
                {"department": {"$regex": search, "$options": "i"}}
            ]
        }

    cursor = db.employees.find(query).skip(skip).limit(limit).sort("name", 1)
    employees = []
    async for emp in cursor:
        employees.append({
            "id": str(emp["_id"]),
            "name": emp["name"],
            "employee_id": emp["employee_id"],
            "department": emp.get("department"),
            "created_at": emp["created_at"],
            "face_registered": emp.get("face_registered", False)
        })

    total = await db.employees.count_documents(query)

    return {"employees": employees, "total": total}


@app.get("/employees/{employee_id}")
async def get_employee(employee_id: str):
    """Get employee details (legacy endpoint)."""
    db = get_database()
    emp = await db.employees.find_one({"employee_id": employee_id})

    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    return {
        "id": str(emp["_id"]),
        "name": emp["name"],
        "employee_id": emp["employee_id"],
        "department": emp.get("department"),
        "created_at": emp["created_at"],
        "face_registered": emp.get("face_registered", False)
    }


@app.delete("/employees/{employee_id}")
async def delete_employee(
    employee_id: str,
    current_admin: dict = Depends(get_current_admin)
):
    """Delete an employee (legacy endpoint)."""
    db = get_database()

    result = await db.employees.delete_one({"employee_id": employee_id})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Employee not found")

    if employee_id in detector.known_faces:
        del detector.known_faces[employee_id]
        detector._save_known_faces()

    return {"success": True, "message": "Employee deleted successfully"}


@app.put("/employees/{employee_id}")
async def update_employee(
    employee_id: str,
    name: str = Form(...),
    department: Optional[str] = Form(None),
    current_admin: dict = Depends(get_current_admin)
):
    """Update employee (legacy endpoint)."""
    db = get_database()

    emp = await db.employees.find_one({"employee_id": employee_id})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    update_data = {"name": name}
    if department is not None:
        update_data["department"] = department

    await db.employees.update_one(
        {"employee_id": employee_id},
        {"$set": update_data}
    )

    return {"success": True, "message": "Employee updated successfully"}


@app.post("/employees/{employee_id}/register-face")
async def register_employee_face(
    employee_id: str,
    file: UploadFile = File(...),
    angle: Optional[str] = Form(None)
):
    """Register face for employee (legacy endpoint).

    Supports multi-angle face registration:
    - No angle: Primary face registration (center view)
    - angle=angle_1: Left view
    - angle=angle_2: Right view
    """
    db = get_database()

    emp = await db.employees.find_one({"employee_id": employee_id})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    contents = await file.read()

    # Determine the face registration key
    if angle:
        face_key = f"{employee_id}_{angle}"
        display_name = f"{emp['name']} ({angle})"
    else:
        face_key = employee_id
        display_name = emp['name']

    success = detector.register_face(face_key, contents, display_name)

    if not success:
        raise HTTPException(status_code=400, detail="No face detected in image")

    # Update employee's face registration status
    update_data = {"face_registered": True}

    if angle:
        registered_angles = emp.get("face_angles", [])
        if angle not in registered_angles:
            registered_angles.append(angle)
        update_data["face_angles"] = registered_angles

    await db.employees.update_one(
        {"employee_id": employee_id},
        {"$set": update_data}
    )

    angle_msg = f" ({angle})" if angle else ""
    return {"success": True, "message": f"Face registered for {emp['name']}{angle_msg}"}


# ==================== Legacy Attendance ====================

@app.post("/attendance/check-in")
async def check_in_attendance(
    file: UploadFile = File(...),
    location: Optional[str] = Form(default=None)
):
    """
    Mark attendance check-in via face recognition.
    Used by gate verification screen for attendance marking.
    """
    try:
        contents = await file.read()
        result_image, detections = detector.process_image(contents)

        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        db = get_database()

        # Get identified person info
        identified_persons = detections.get("summary", {}).get("identified_persons", [])
        identified_names = detections.get("summary", {}).get("identified_names", [])

        if not identified_persons:
            return JSONResponse({
                "success": False,
                "message": "No face identified in image",
                "attendance_marked": False
            })

        employee_id = identified_persons[0]
        employee_name = identified_names[0] if identified_names else employee_id

        # Try to find worker
        worker = await db.workers.find_one({"employee_id": employee_id})
        if not worker:
            worker = await db.employees.find_one({"employee_id": employee_id})

        if worker:
            employee_name = worker.get("name", employee_name)

        # Get violations from detection
        violations = detections.get("violations", [])
        violation_labels = [v.get("label", "Unknown") for v in violations]

        # Check if already checked in today
        today = datetime.utcnow().strftime("%Y-%m-%d")
        existing = await db.attendance.find_one({
            "employee_id": employee_id,
            "date": today,
            "type": "check_in"
        })

        if existing:
            return JSONResponse({
                "success": True,
                "message": f"Already checked in today",
                "attendance_marked": True,
                "attendance": {
                    "employee_id": employee_id,
                    "employee_name": employee_name,
                    "timestamp": existing["timestamp"].isoformat(),
                    "already_checked_in": True
                }
            })

        # Create attendance record
        attendance_doc = {
            "employee_id": employee_id,
            "employee_name": employee_name,
            "worker_id": str(worker["_id"]) if worker and "_id" in worker else None,
            "type": "check_in",
            "timestamp": datetime.utcnow(),
            "date": today,
            "location": location,
            "ppe_compliant": len(violations) == 0,
            "violations": violation_labels,
            "image": f"data:image/png;base64,{img_base64}"
        }

        await db.attendance.insert_one(attendance_doc)

        return JSONResponse({
            "success": True,
            "message": f"Attendance marked for {employee_name}",
            "attendance_marked": True,
            "attendance": {
                "employee_id": employee_id,
                "employee_name": employee_name,
                "timestamp": attendance_doc["timestamp"].isoformat(),
                "ppe_compliant": attendance_doc["ppe_compliant"],
                "violations": violation_labels
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "attendance_marked": False
        }, status_code=500)


@app.post("/attendance/check-out")
async def check_out_attendance(
    file: UploadFile = File(...),
    location: Optional[str] = Form(default=None)
):
    """
    Mark attendance check-out via face recognition.
    """
    try:
        contents = await file.read()
        result_image, detections = detector.process_image(contents)

        db = get_database()

        # Get identified person info
        identified_persons = detections.get("summary", {}).get("identified_persons", [])
        identified_names = detections.get("summary", {}).get("identified_names", [])

        if not identified_persons:
            return JSONResponse({
                "success": False,
                "message": "No face identified in image",
                "attendance_marked": False
            })

        employee_id = identified_persons[0]
        employee_name = identified_names[0] if identified_names else employee_id

        # Try to find worker
        worker = await db.workers.find_one({"employee_id": employee_id})
        if not worker:
            worker = await db.employees.find_one({"employee_id": employee_id})

        if worker:
            employee_name = worker.get("name", employee_name)

        # Create check-out record
        today = datetime.utcnow().strftime("%Y-%m-%d")
        attendance_doc = {
            "employee_id": employee_id,
            "employee_name": employee_name,
            "worker_id": str(worker["_id"]) if worker and "_id" in worker else None,
            "type": "check_out",
            "timestamp": datetime.utcnow(),
            "date": today,
            "location": location
        }

        await db.attendance.insert_one(attendance_doc)

        return JSONResponse({
            "success": True,
            "message": f"Check-out marked for {employee_name}",
            "attendance_marked": True,
            "attendance": {
                "employee_id": employee_id,
                "employee_name": employee_name,
                "timestamp": attendance_doc["timestamp"].isoformat()
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "attendance_marked": False
        }, status_code=500)


@app.get("/attendance")
async def get_attendance_records(
    date: Optional[str] = None,
    employee_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    """Get attendance records with filters."""
    db = get_database()

    query = {}

    if date:
        query["date"] = date
    elif start_date or end_date:
        query["date"] = {}
        if start_date:
            query["date"]["$gte"] = start_date
        if end_date:
            query["date"]["$lte"] = end_date

    if employee_id:
        query["employee_id"] = employee_id

    cursor = db.attendance.find(query).skip(skip).limit(limit).sort("timestamp", -1)
    records = []

    async for record in cursor:
        records.append({
            "id": str(record["_id"]),
            "employee_id": record.get("employee_id"),
            "employee_name": record.get("employee_name"),
            "type": record.get("type"),
            "timestamp": record["timestamp"].isoformat() if record.get("timestamp") else None,
            "date": record.get("date"),
            "location": record.get("location"),
            "ppe_compliant": record.get("ppe_compliant"),
            "violations": record.get("violations", [])
        })

    total = await db.attendance.count_documents(query)

    return {"records": records, "total": total}


@app.get("/attendance/today")
async def get_today_attendance():
    """Get today's attendance summary."""
    db = get_database()

    today = datetime.utcnow().strftime("%Y-%m-%d")

    # Get unique check-ins
    pipeline = [
        {"$match": {"date": today, "type": "check_in"}},
        {"$group": {
            "_id": "$employee_id",
            "employee_name": {"$first": "$employee_name"},
            "check_in_time": {"$first": "$timestamp"},
            "ppe_compliant": {"$first": "$ppe_compliant"}
        }}
    ]

    check_ins = []
    async for record in db.attendance.aggregate(pipeline):
        check_ins.append({
            "employee_id": record["_id"],
            "employee_name": record.get("employee_name"),
            "check_in_time": record["check_in_time"].isoformat() if record.get("check_in_time") else None,
            "ppe_compliant": record.get("ppe_compliant", True)
        })

    # Get total workers for attendance rate
    total_workers = await db.workers.count_documents({"is_active": True})
    total_employees = await db.employees.count_documents({})
    total_all = total_workers + total_employees

    return {
        "date": today,
        "present": len(check_ins),
        "total": total_all,
        "attendance_rate": round((len(check_ins) / total_all * 100) if total_all > 0 else 0, 1),
        "records": check_ins
    }


# ==================== Legacy Violations ====================

@app.get("/violations")
async def list_violations(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    """List all PPE violations (legacy endpoint)."""
    db = get_database()

    cursor = db.ppe_violations.find().skip(skip).limit(limit).sort("timestamp", -1)
    violations = []
    async for v in cursor:
        violations.append({
            "id": str(v["_id"]),
            "employee_id": v.get("employee_id"),
            "employee_name": v.get("employee_name"),
            "violations": v.get("violations", []),
            "timestamp": v.get("timestamp"),
            "location": v.get("location")
        })

    total = await db.ppe_violations.count_documents({})
    return {"violations": violations, "total": total}


@app.get("/violations/today")
async def get_today_violations():
    """Get today's PPE violations (legacy endpoint)."""
    db = get_database()

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)

    cursor = db.ppe_violations.find({
        "timestamp": {"$gte": today, "$lt": tomorrow}
    }).sort("timestamp", -1)

    violations = []
    async for v in cursor:
        violations.append({
            "id": str(v["_id"]),
            "employee_id": v.get("employee_id"),
            "employee_name": v.get("employee_name"),
            "violations": v.get("violations", []),
            "timestamp": v.get("timestamp"),
            "location": v.get("location")
        })

    return {"violations": violations, "total": len(violations)}


# ==================== Legacy Dashboard ====================

@app.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics (legacy endpoint)."""
    db = get_database()

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    week_ago = today - timedelta(days=7)

    # Total employees (from both collections)
    total_employees = await db.employees.count_documents({})
    total_workers = await db.workers.count_documents({"is_active": True})
    total_all = total_employees + total_workers

    # Present today
    today_str = today.strftime("%Y-%m-%d")
    present_pipeline = [
        {"$match": {"date": today_str, "type": "check_in"}},
        {"$group": {"_id": "$employee_id"}}
    ]
    present_cursor = db.attendance.aggregate(present_pipeline)
    present_today = len([x async for x in present_cursor])

    # Gate entries today
    gate_entries_today = await db.gate_entries.count_documents({
        "timestamp": {"$gte": today, "$lt": tomorrow},
        "entry_type": "entry"
    })

    # Violations today
    violations_today = await db.ppe_violations.count_documents({
        "timestamp": {"$gte": today, "$lt": tomorrow}
    })
    gate_violations_today = await db.gate_entries.count_documents({
        "timestamp": {"$gte": today, "$lt": tomorrow},
        "violations": {"$ne": []}
    })

    # Violations this week
    violations_week = await db.ppe_violations.count_documents({
        "timestamp": {"$gte": week_ago}
    })

    # Compliance rate
    total_checks_week = gate_entries_today + await db.attendance.count_documents({
        "timestamp": {"$gte": week_ago}
    })
    compliance_rate = 100.0
    if total_checks_week > 0:
        compliance_rate = max(0, 100 - ((violations_week + gate_violations_today) / total_checks_week * 100))

    return {
        "total_employees": total_all,
        "present_today": present_today + gate_entries_today,
        "absent_today": max(0, total_all - present_today - gate_entries_today),
        "violations_today": violations_today + gate_violations_today,
        "violations_this_week": violations_week,
        "compliance_rate": round(compliance_rate, 1),
    }


# ==================== Legacy Face Registration ====================

@app.post("/register-face")
async def register_face(name: str, file: UploadFile = File(...), display_name: str = None):
    """Register a new face for recognition (legacy endpoint).

    Args:
        name: Employee ID or unique identifier
        file: Image file with face
        display_name: Optional actual name to display (defaults to name if not provided)
    """
    try:
        contents = await file.read()
        success = detector.register_face(name, contents, display_name or name)

        if success:
            return {"success": True, "message": f"Face registered for {display_name or name} ({name})"}
        else:
            return {"success": False, "message": "No face detected in image"}
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.get("/known-faces")
def get_known_faces():
    """Get list of registered faces."""
    return {"faces": detector.get_known_faces(), "count": len(detector.known_faces)}


@app.post("/reload-faces")
def reload_faces():
    """Reload faces from disk. Call this after new face registration if faces aren't being detected."""
    detector.reload_faces()
    return {
        "success": True,
        "message": "Faces reloaded from disk",
        "faces": detector.get_known_faces(),
        "count": len(detector.known_faces)
    }


# ==================== Live Video Streaming ====================

@app.get("/video/status")
def get_video_status():
    """Get the status of video streaming capability."""
    processor = get_video_processor(detector)
    return {
        "available": True,
        "running": processor.is_running,
        "message": "Local YOLOv8 video pipeline ready"
    }


@app.post("/video/start")
async def start_video_stream(
    video_source: int = Form(default=DEFAULT_CAMERA),
    max_fps: int = Form(default=15)
):
    """Start the video streaming pipeline."""
    processor = get_video_processor(detector)

    if processor.is_running:
        return {"success": True, "message": "Video stream already running"}

    # Update max_fps if different
    processor.max_fps = max_fps

    success = processor.start(video_source=video_source)

    if success:
        return {"success": True, "message": "Video stream started"}
    else:
        raise HTTPException(status_code=500, detail="Failed to start video stream")


@app.post("/video/stop")
async def stop_video_stream():
    """Stop the video streaming pipeline."""
    processor = get_video_processor(detector)
    processor.stop()
    return {"success": True, "message": "Video stream stopped"}


@app.websocket("/ws/video")
async def websocket_video_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming video frames with detections."""
    print("Incoming WebSocket connection request to /ws/video...")
    try:
        await websocket.accept()
        print("WebSocket connection ACCEPTED")
    except Exception as e:
        print(f"WebSocket accept failed: {e}")
        return

    processor = get_video_processor(detector)

    # Auto-start if not running
    if not processor.is_running:
        print("Starting video processor...")
        success = processor.start(video_source=DEFAULT_CAMERA)
        if not success:
            await websocket.send_json({
                "error": "Failed to start video stream. Check if camera is available."
            })
            await websocket.close()
            return

        # Give the camera time to initialize
        await asyncio.sleep(2.0)

    try:
        heartbeat_count = 0
        frame_count = 0
        while True:
            # Get frame from queue (non-blocking)
            frame_data = processor.get_frame_nowait()

            if frame_data:
                frame_count += 1
                if frame_count % 15 == 0:
                    print(f"Streaming frame {frame_count} to client...")
                
                if frame_count % 30 == 0:
                    print(f"DEBUG [WS]: Sending frame={frame_count}, detections={frame_data['result'].get('detections', {}).get('simplified', {})}")

                await websocket.send_json({
                    "type": "frame",
                    "frame": frame_data["frame"],
                    "detections": frame_data["result"].get("detections", {}).get("simplified", {}),
                    "result": frame_data["result"]
                })
                heartbeat_count = 0
            else:
                heartbeat_count += 1
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.01)

                # Send heartbeat if no frame available for a while
                if heartbeat_count % 100 == 0:
                    await websocket.send_json({"type": "heartbeat"})
                    heartbeat_count = 0

                # If too many heartbeats without frames, check if pipeline is still running
                if heartbeat_count > 500 and not processor.is_running:
                    await websocket.send_json({
                        "error": "Video pipeline stopped unexpectedly"
                    })
                    break

            # Small delay to prevent overwhelming the connection
            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("WebSocket connection closed")
        pass


@app.get("/video/frame")
async def get_video_frame():
    """Get the latest video frame (for polling-based clients)."""
    processor = get_video_processor(detector)

    if not processor.is_running:
        raise HTTPException(status_code=400, detail="Video stream not running")

    frame_data = processor.get_latest_frame()

    if frame_data:
        return JSONResponse({
            "success": True,
            "frame": frame_data["frame"],
            "result": frame_data["result"]
        })
    else:
        return JSONResponse({
            "success": False,
            "message": "No frame available yet"
        })


async def generate_mjpeg_stream():
    """Generator for MJPEG stream using the latest frame buffer."""
    processor = get_video_processor(detector)

    while processor.is_running:
        frame_data = processor.get_frame_nowait()

        if frame_data:
            frame_bytes = base64.b64decode(frame_data["frame"])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        else:
            # If no new frame, wait a bit
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.01)


@app.get("/video/mjpeg")
async def mjpeg_stream():
    """MJPEG stream endpoint for simple video display."""
    processor = get_video_processor(detector)

    if not processor.is_running:
        processor.start(video_source=DEFAULT_CAMERA)

    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
