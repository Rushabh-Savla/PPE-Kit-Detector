"""
Mine, Zone, and Gate management routes.
"""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, status, Depends, Query
from bson import ObjectId
from database import get_database
from auth import (
    get_current_user, get_manager_or_above, get_area_safety_officer_or_above,
    get_general_manager, UserRole, check_mine_access
)
from schemas import (
    MineCreate, MineResponse, MineList, ZoneCreate, ZoneResponse,
    GateCreate, GateResponse, GateType, MineVisualizationData
)

router = APIRouter(prefix="/mines", tags=["Mine Management"])


# ==================== Mine Endpoints ====================

@router.post("", response_model=MineResponse)
async def create_mine(
    mine_data: MineCreate,
    current_user: dict = Depends(get_general_manager)
):
    """Create a new mine. General Manager and above only."""
    db = get_database()

    # Check if mine name already exists
    existing = await db.mines.find_one({"name": mine_data.name})
    if existing:
        raise HTTPException(status_code=400, detail="Mine with this name already exists")

    mine_doc = {
        "name": mine_data.name,
        "location": mine_data.location,
        "description": mine_data.description,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "created_by": current_user.get("user_id"),
    }

    result = await db.mines.insert_one(mine_doc)

    return MineResponse(
        id=str(result.inserted_id),
        name=mine_doc["name"],
        location=mine_doc["location"],
        description=mine_doc["description"],
        zones=[],
        gates=[],
        created_at=mine_doc["created_at"],
        is_active=True
    )


@router.get("", response_model=MineList)
async def list_mines(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = None,
    is_active: Optional[bool] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    List mines. Access based on role:
    - Super Admin, General Manager: All mines
    - Area Safety Officer: Assigned mines only
    - Others: Their assigned mine only
    """
    db = get_database()

    query = {}
    user_role = UserRole(current_user.get("role"))

    if user_role in [UserRole.SUPER_ADMIN, UserRole.GENERAL_MANAGER]:
        pass  # Can see all
    elif user_role == UserRole.AREA_SAFETY_OFFICER:
        mine_ids = current_user.get("mine_ids", [])
        query["_id"] = {"$in": [ObjectId(mid) for mid in mine_ids]}
    elif current_user.get("user_type") == "worker":
        # Workers can only see their mine
        worker_mine_id = current_user.get("mine_id")
        if worker_mine_id:
            query["_id"] = ObjectId(worker_mine_id)
        else:
            return MineList(mines=[], total=0)
    else:
        # Manager, Safety Officer, Shift Incharge
        user_mine_id = current_user.get("mine_id")
        if user_mine_id:
            query["_id"] = ObjectId(user_mine_id)
        else:
            return MineList(mines=[], total=0)

    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"location": {"$regex": search, "$options": "i"}}
        ]

    if is_active is not None:
        query["is_active"] = is_active

    cursor = db.mines.find(query).skip(skip).limit(limit).sort("name", 1)
    mines = []

    async for mine in cursor:
        # Get zones and gates for this mine
        zones = []
        zones_cursor = db.zones.find({"mine_id": mine["_id"]})
        async for zone in zones_cursor:
            # Count workers in zone
            worker_count = await db.workers.count_documents({
                "zone_id": zone["_id"],
                "is_active": True
            })
            zones.append(ZoneResponse(
                id=str(zone["_id"]),
                name=zone["name"],
                description=zone.get("description"),
                risk_level=zone.get("risk_level", "normal"),
                coordinates=zone.get("coordinates"),
                worker_count=worker_count
            ))

        gates = []
        gates_cursor = db.gates.find({"mine_id": mine["_id"]})
        async for gate in gates_cursor:
            gates.append(GateResponse(
                id=str(gate["_id"]),
                name=gate["name"],
                gate_type=GateType(gate.get("gate_type", "both")),
                zone_id=str(gate["zone_id"]) if gate.get("zone_id") else None,
                location=gate.get("location"),
                has_camera=gate.get("has_camera", True),
                is_active=gate.get("is_active", True)
            ))

        mines.append(MineResponse(
            id=str(mine["_id"]),
            name=mine["name"],
            location=mine["location"],
            description=mine.get("description"),
            zones=zones,
            gates=gates,
            created_at=mine["created_at"],
            is_active=mine.get("is_active", True)
        ))

    total = await db.mines.count_documents(query)

    return MineList(mines=mines, total=total)


@router.get("/{mine_id}", response_model=MineResponse)
async def get_mine(
    mine_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get mine details by ID."""
    db = get_database()

    try:
        mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid mine ID")

    if not mine:
        raise HTTPException(status_code=404, detail="Mine not found")

    # Check access
    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    # Get zones and gates
    zones = []
    zones_cursor = db.zones.find({"mine_id": mine["_id"]})
    async for zone in zones_cursor:
        worker_count = await db.workers.count_documents({
            "zone_id": zone["_id"],
            "is_active": True
        })
        zones.append(ZoneResponse(
            id=str(zone["_id"]),
            name=zone["name"],
            description=zone.get("description"),
            risk_level=zone.get("risk_level", "normal"),
            coordinates=zone.get("coordinates"),
            worker_count=worker_count
        ))

    gates = []
    gates_cursor = db.gates.find({"mine_id": mine["_id"]})
    async for gate in gates_cursor:
        gates.append(GateResponse(
            id=str(gate["_id"]),
            name=gate["name"],
            gate_type=GateType(gate.get("gate_type", "both")),
            zone_id=str(gate["zone_id"]) if gate.get("zone_id") else None,
            location=gate.get("location"),
            has_camera=gate.get("has_camera", True),
            is_active=gate.get("is_active", True)
        ))

    return MineResponse(
        id=str(mine["_id"]),
        name=mine["name"],
        location=mine["location"],
        description=mine.get("description"),
        zones=zones,
        gates=gates,
        created_at=mine["created_at"],
        is_active=mine.get("is_active", True)
    )


@router.put("/{mine_id}", response_model=MineResponse)
async def update_mine(
    mine_id: str,
    mine_data: MineCreate,
    current_user: dict = Depends(get_general_manager)
):
    """Update mine details. General Manager and above only."""
    db = get_database()

    try:
        mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid mine ID")

    if not mine:
        raise HTTPException(status_code=404, detail="Mine not found")

    update_doc = {
        "name": mine_data.name,
        "location": mine_data.location,
        "description": mine_data.description,
        "updated_at": datetime.utcnow(),
        "updated_by": current_user.get("user_id"),
    }

    await db.mines.update_one({"_id": ObjectId(mine_id)}, {"$set": update_doc})

    return await get_mine(mine_id, current_user)


@router.delete("/{mine_id}")
async def delete_mine(
    mine_id: str,
    current_user: dict = Depends(get_general_manager)
):
    """Deactivate mine. General Manager and above only."""
    db = get_database()

    try:
        mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid mine ID")

    if not mine:
        raise HTTPException(status_code=404, detail="Mine not found")

    await db.mines.update_one(
        {"_id": ObjectId(mine_id)},
        {"$set": {
            "is_active": False,
            "deleted_at": datetime.utcnow(),
            "deleted_by": current_user.get("user_id")
        }}
    )

    return {"success": True, "message": "Mine deactivated successfully"}


# ==================== Zone Endpoints ====================

@router.post("/{mine_id}/zones", response_model=ZoneResponse)
async def create_zone(
    mine_id: str,
    zone_data: ZoneCreate,
    current_user: dict = Depends(get_manager_or_above)
):
    """Create a new zone in a mine."""
    db = get_database()

    # Check mine exists and access
    try:
        mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid mine ID")

    if not mine:
        raise HTTPException(status_code=404, detail="Mine not found")

    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    zone_doc = {
        "mine_id": ObjectId(mine_id),
        "name": zone_data.name,
        "description": zone_data.description,
        "risk_level": zone_data.risk_level,
        "coordinates": zone_data.coordinates,
        "created_at": datetime.utcnow(),
        "created_by": current_user.get("user_id"),
    }

    result = await db.zones.insert_one(zone_doc)

    return ZoneResponse(
        id=str(result.inserted_id),
        name=zone_doc["name"],
        description=zone_doc["description"],
        risk_level=zone_doc["risk_level"],
        coordinates=zone_doc["coordinates"],
        worker_count=0
    )


@router.get("/{mine_id}/zones")
async def list_zones(
    mine_id: str,
    current_user: dict = Depends(get_current_user)
):
    """List zones in a mine."""
    db = get_database()

    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    zones = []
    cursor = db.zones.find({"mine_id": ObjectId(mine_id)})

    async for zone in cursor:
        worker_count = await db.workers.count_documents({
            "zone_id": zone["_id"],
            "is_active": True
        })
        zones.append(ZoneResponse(
            id=str(zone["_id"]),
            name=zone["name"],
            description=zone.get("description"),
            risk_level=zone.get("risk_level", "normal"),
            coordinates=zone.get("coordinates"),
            worker_count=worker_count
        ))

    return {"zones": zones, "total": len(zones)}


@router.put("/{mine_id}/zones/{zone_id}", response_model=ZoneResponse)
async def update_zone(
    mine_id: str,
    zone_id: str,
    zone_data: ZoneCreate,
    current_user: dict = Depends(get_manager_or_above)
):
    """Update zone details."""
    db = get_database()

    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    try:
        zone = await db.zones.find_one({"_id": ObjectId(zone_id), "mine_id": ObjectId(mine_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid zone ID")

    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")

    update_doc = {
        "name": zone_data.name,
        "description": zone_data.description,
        "risk_level": zone_data.risk_level,
        "coordinates": zone_data.coordinates,
        "updated_at": datetime.utcnow(),
        "updated_by": current_user.get("user_id"),
    }

    await db.zones.update_one({"_id": ObjectId(zone_id)}, {"$set": update_doc})

    worker_count = await db.workers.count_documents({
        "zone_id": ObjectId(zone_id),
        "is_active": True
    })

    return ZoneResponse(
        id=zone_id,
        name=update_doc["name"],
        description=update_doc["description"],
        risk_level=update_doc["risk_level"],
        coordinates=update_doc["coordinates"],
        worker_count=worker_count
    )


@router.delete("/{mine_id}/zones/{zone_id}")
async def delete_zone(
    mine_id: str,
    zone_id: str,
    current_user: dict = Depends(get_manager_or_above)
):
    """Delete a zone."""
    db = get_database()

    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    result = await db.zones.delete_one({
        "_id": ObjectId(zone_id),
        "mine_id": ObjectId(mine_id)
    })

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Zone not found")

    # Remove zone_id from workers
    await db.workers.update_many(
        {"zone_id": ObjectId(zone_id)},
        {"$set": {"zone_id": None}}
    )

    return {"success": True, "message": "Zone deleted successfully"}


# ==================== Gate Endpoints ====================

@router.post("/{mine_id}/gates", response_model=GateResponse)
async def create_gate(
    mine_id: str,
    gate_data: GateCreate,
    current_user: dict = Depends(get_manager_or_above)
):
    """Create a new gate in a mine."""
    db = get_database()

    # Check mine exists and access
    try:
        mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid mine ID")

    if not mine:
        raise HTTPException(status_code=404, detail="Mine not found")

    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    gate_doc = {
        "mine_id": ObjectId(mine_id),
        "name": gate_data.name,
        "gate_type": gate_data.gate_type.value,
        "zone_id": ObjectId(gate_data.zone_id) if gate_data.zone_id else None,
        "location": gate_data.location,
        "has_camera": gate_data.has_camera,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "created_by": current_user.get("user_id"),
    }

    result = await db.gates.insert_one(gate_doc)

    return GateResponse(
        id=str(result.inserted_id),
        name=gate_doc["name"],
        gate_type=gate_data.gate_type,
        zone_id=gate_data.zone_id,
        location=gate_doc["location"],
        has_camera=gate_doc["has_camera"],
        is_active=True
    )


@router.get("/{mine_id}/gates")
async def list_gates(
    mine_id: str,
    current_user: dict = Depends(get_current_user)
):
    """List gates in a mine."""
    db = get_database()

    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    gates = []
    cursor = db.gates.find({"mine_id": ObjectId(mine_id)})

    async for gate in cursor:
        gates.append(GateResponse(
            id=str(gate["_id"]),
            name=gate["name"],
            gate_type=GateType(gate.get("gate_type", "both")),
            zone_id=str(gate["zone_id"]) if gate.get("zone_id") else None,
            location=gate.get("location"),
            has_camera=gate.get("has_camera", True),
            is_active=gate.get("is_active", True)
        ))

    return {"gates": gates, "total": len(gates)}


@router.put("/{mine_id}/gates/{gate_id}", response_model=GateResponse)
async def update_gate(
    mine_id: str,
    gate_id: str,
    gate_data: GateCreate,
    current_user: dict = Depends(get_manager_or_above)
):
    """Update gate details."""
    db = get_database()

    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    try:
        gate = await db.gates.find_one({"_id": ObjectId(gate_id), "mine_id": ObjectId(mine_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid gate ID")

    if not gate:
        raise HTTPException(status_code=404, detail="Gate not found")

    update_doc = {
        "name": gate_data.name,
        "gate_type": gate_data.gate_type.value,
        "zone_id": ObjectId(gate_data.zone_id) if gate_data.zone_id else None,
        "location": gate_data.location,
        "has_camera": gate_data.has_camera,
        "updated_at": datetime.utcnow(),
        "updated_by": current_user.get("user_id"),
    }

    await db.gates.update_one({"_id": ObjectId(gate_id)}, {"$set": update_doc})

    return GateResponse(
        id=gate_id,
        name=update_doc["name"],
        gate_type=gate_data.gate_type,
        zone_id=gate_data.zone_id,
        location=update_doc["location"],
        has_camera=update_doc["has_camera"],
        is_active=gate.get("is_active", True)
    )


@router.delete("/{mine_id}/gates/{gate_id}")
async def delete_gate(
    mine_id: str,
    gate_id: str,
    current_user: dict = Depends(get_manager_or_above)
):
    """Deactivate a gate."""
    db = get_database()

    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    result = await db.gates.update_one(
        {"_id": ObjectId(gate_id), "mine_id": ObjectId(mine_id)},
        {"$set": {
            "is_active": False,
            "deleted_at": datetime.utcnow(),
            "deleted_by": current_user.get("user_id")
        }}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Gate not found")

    return {"success": True, "message": "Gate deactivated successfully"}


# ==================== Mine Visualization ====================

@router.get("/{mine_id}/visualization", response_model=MineVisualizationData)
async def get_mine_visualization(
    mine_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get visualization data for the mine map.
    Returns zones, gates, and current worker positions.
    """
    db = get_database()

    if not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    try:
        mine = await db.mines.find_one({"_id": ObjectId(mine_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid mine ID")

    if not mine:
        raise HTTPException(status_code=404, detail="Mine not found")

    # Get zones with worker counts
    zones_data = []
    zones_cursor = db.zones.find({"mine_id": ObjectId(mine_id)})
    async for zone in zones_cursor:
        worker_count = await db.workers.count_documents({
            "zone_id": zone["_id"],
            "is_active": True
        })

        # Get today's violations in this zone
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        violations_count = await db.gate_entries.count_documents({
            "mine_id": ObjectId(mine_id),
            "timestamp": {"$gte": today},
            "violations": {"$ne": []}
        })

        zones_data.append({
            "id": str(zone["_id"]),
            "name": zone["name"],
            "description": zone.get("description"),
            "risk_level": zone.get("risk_level", "normal"),
            "coordinates": zone.get("coordinates", {"x": 0, "y": 0, "width": 100, "height": 100}),
            "worker_count": worker_count,
            "violations_today": violations_count
        })

    # Get gates
    gates_data = []
    gates_cursor = db.gates.find({"mine_id": ObjectId(mine_id), "is_active": True})
    async for gate in gates_cursor:
        gates_data.append({
            "id": str(gate["_id"]),
            "name": gate["name"],
            "gate_type": gate.get("gate_type", "both"),
            "zone_id": str(gate["zone_id"]) if gate.get("zone_id") else None,
            "location": gate.get("location"),
            "has_camera": gate.get("has_camera", True),
            "position": gate.get("position", {"x": 50, "y": 0})
        })

    # Get current workers inside (those who entered but not exited today)
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # This is a simplified version - in production, you'd track real-time positions
    workers_inside = []
    workers_cursor = db.workers.find({
        "mine_id": ObjectId(mine_id),
        "is_active": True
    })

    async for worker in workers_cursor:
        # Check if worker has entered today
        last_entry = await db.gate_entries.find_one(
            {
                "worker_id": str(worker["_id"]),
                "timestamp": {"$gte": today}
            },
            sort=[("timestamp", -1)]
        )

        if last_entry and last_entry.get("entry_type") == "entry":
            zone_name = None
            if worker.get("zone_id"):
                zone = await db.zones.find_one({"_id": worker["zone_id"]})
                if zone:
                    zone_name = zone["name"]

            workers_inside.append({
                "id": str(worker["_id"]),
                "employee_id": worker["employee_id"],
                "name": worker["name"],
                "zone_id": str(worker["zone_id"]) if worker.get("zone_id") else None,
                "zone_name": zone_name,
                "last_entry_time": last_entry["timestamp"].isoformat(),
                "ppe_compliant": len(last_entry.get("violations", [])) == 0
            })

    # Get risk zones (zones with high violation rates)
    risk_zones = []
    for zone in zones_data:
        if zone["violations_today"] > 5 or zone["risk_level"] == "critical":
            risk_zones.append({
                "zone_id": zone["id"],
                "zone_name": zone["name"],
                "risk_level": "high" if zone["violations_today"] > 5 else zone["risk_level"],
                "violations_today": zone["violations_today"]
            })

    # Statistics
    total_workers = await db.workers.count_documents({
        "mine_id": ObjectId(mine_id),
        "is_active": True
    })

    total_violations_today = await db.gate_entries.count_documents({
        "mine_id": ObjectId(mine_id),
        "timestamp": {"$gte": today},
        "violations": {"$ne": []}
    })

    statistics = {
        "total_workers": total_workers,
        "workers_inside": len(workers_inside),
        "total_zones": len(zones_data),
        "total_gates": len(gates_data),
        "violations_today": total_violations_today,
        "compliance_rate": round(
            ((total_workers - total_violations_today) / total_workers * 100) if total_workers > 0 else 100,
            1
        )
    }

    return MineVisualizationData(
        mine_id=mine_id,
        mine_name=mine["name"],
        zones=zones_data,
        gates=gates_data,
        workers_positions=workers_inside,
        risk_zones=risk_zones,
        statistics=statistics
    )
