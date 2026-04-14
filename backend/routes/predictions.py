"""
Prediction API Routes.

Endpoints for worker risk prediction and analysis.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
from datetime import datetime, timedelta
from bson import ObjectId

from database import get_database
from auth import (
    get_current_user, get_shift_incharge_or_above, get_manager_or_above,
    check_mine_access
)
from schemas import (
    WorkerPredictionResponse, AtRiskWorkersSummary, AtRiskWorkerSummary,
    PredictionTrends, BatchPredictionResponse, BatchPredictionResult,
    RiskCategory, AttendancePattern, RiskFactor
)

router = APIRouter(prefix="/predictions", tags=["Predictive Analytics"])


@router.get("/worker/{worker_id}", response_model=WorkerPredictionResponse)
async def get_worker_prediction(
    worker_id: str,
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """
    Get prediction for a specific worker.
    Returns latest prediction or generates new if expired.
    """
    db = get_database()

    # Verify worker exists
    try:
        worker = await db.workers.find_one({"_id": ObjectId(worker_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid worker ID format")

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Check mine access
    mine_id = worker.get("mine_id")
    if mine_id and not check_mine_access(current_user, str(mine_id)):
        raise HTTPException(status_code=403, detail="No access to this worker's mine")

    # Find latest prediction
    prediction = await db.predictions.find_one(
        {"worker_id": worker_id},
        sort=[("prediction_date", -1)]
    )

    # If no prediction or expired, generate new one
    if not prediction or prediction.get("expires_at", datetime.min) < datetime.utcnow():
        prediction = await _generate_prediction(db, worker_id)

    # Format response
    return WorkerPredictionResponse(
        worker_id=worker_id,
        employee_id=worker.get("employee_id", ""),
        worker_name=worker.get("name"),
        prediction_date=prediction["prediction_date"],
        overall_risk_score=prediction["overall_risk_score"],
        risk_category=prediction["risk_category"],
        violation_risk_score=prediction["violation_risk_score"],
        attendance_risk_score=prediction["attendance_risk_score"],
        compliance_trend_score=prediction["compliance_trend_score"],
        predicted_violations_count=prediction["predicted_violations_count"],
        predicted_absent_days=prediction["predicted_absent_days"],
        high_risk_ppe_items=prediction["high_risk_ppe_items"],
        requires_intervention=prediction["requires_intervention"],
        attendance_pattern=prediction["attendance_pattern"],
        consecutive_absence_risk=prediction["consecutive_absence_risk"],
        attendance_rate_30d=prediction["attendance_rate_30d"],
        risk_factors=[RiskFactor(**rf) for rf in prediction.get("risk_factors", [])],
        confidence=prediction["confidence"],
        model_version=prediction["model_version"],
        created_at=prediction["created_at"],
        expires_at=prediction["expires_at"]
    )


@router.get("/at-risk-workers", response_model=AtRiskWorkersSummary)
async def get_at_risk_workers(
    mine_id: Optional[str] = None,
    risk_category: Optional[str] = Query(None, regex="^(medium|high|critical)$"),
    limit: int = Query(50, ge=1, le=100),
    current_user: dict = Depends(get_shift_incharge_or_above)
):
    """
    Get list of at-risk workers (medium, high, critical).
    Filtered by mine access.
    """
    db = get_database()

    # Build query
    query = {
        "risk_category": {"$in": ["medium", "high", "critical"]},
        "expires_at": {"$gt": datetime.utcnow()}  # Only valid predictions
    }

    if risk_category:
        query["risk_category"] = risk_category

    # Filter by mine access
    accessible_mine_ids = _get_accessible_mine_ids(current_user, mine_id)

    if accessible_mine_ids is not None:
        # Get workers from accessible mines
        accessible_workers = await db.workers.find(
            {"mine_id": {"$in": [ObjectId(mid) for mid in accessible_mine_ids]}}
        ).to_list(length=None)

        accessible_worker_ids = [str(w["_id"]) for w in accessible_workers]
        query["worker_id"] = {"$in": accessible_worker_ids}

    # Get predictions
    cursor = db.predictions.find(query).sort("overall_risk_score", -1).limit(limit)
    predictions = await cursor.to_list(length=limit)

    # Get worker details and format response
    workers_data = []
    for pred in predictions:
        worker = await db.workers.find_one({"_id": ObjectId(pred["worker_id"])})
        if not worker:
            continue

        # Get primary risk factor
        risk_factors = pred.get("risk_factors", [])
        main_issue = risk_factors[0]["description"] if risk_factors else "Multiple factors"

        workers_data.append(AtRiskWorkerSummary(
            worker_id=pred["worker_id"],
            employee_id=worker.get("employee_id", ""),
            worker_name=worker.get("name", ""),
            risk_score=pred["overall_risk_score"],
            risk_category=pred["risk_category"],
            main_issue=main_issue,
            requires_intervention=pred["requires_intervention"]
        ))

    # Calculate summary by category
    by_category = {}
    for cat in ["medium", "high", "critical"]:
        count_query = {**query, "risk_category": cat}
        by_category[cat] = await db.predictions.count_documents(count_query)

    return AtRiskWorkersSummary(
        total_at_risk=len(workers_data),
        by_category=by_category,
        workers=workers_data
    )


@router.post("/generate-all", response_model=BatchPredictionResponse)
async def generate_all_predictions(
    mine_id: Optional[str] = None,
    force_refresh: bool = False,
    current_user: dict = Depends(get_manager_or_above)
):
    """
    Generate predictions for all workers (or specific mine).
    Manager+ only.
    """
    db = get_database()

    # Build worker query
    query = {"is_active": True}

    # Check mine access
    if mine_id:
        if not check_mine_access(current_user, mine_id):
            raise HTTPException(status_code=403, detail="No access to this mine")
        query["mine_id"] = ObjectId(mine_id)
    else:
        # For non-super admins, filter by accessible mines
        accessible_mine_ids = _get_accessible_mine_ids(current_user, None)
        if accessible_mine_ids is not None:
            query["mine_id"] = {"$in": [ObjectId(mid) for mid in accessible_mine_ids]}

    workers = await db.workers.find(query).to_list(length=None)

    # Generate predictions
    results = []
    for worker in workers:
        worker_id = str(worker["_id"])
        try:
            # Check if recent prediction exists (unless force_refresh)
            if not force_refresh:
                recent_pred = await db.predictions.find_one({
                    "worker_id": worker_id,
                    "expires_at": {"$gt": datetime.utcnow()},
                    "created_at": {"$gt": datetime.utcnow() - timedelta(hours=12)}
                })
                if recent_pred:
                    results.append(BatchPredictionResult(
                        worker_id=worker_id,
                        status="success",
                        risk_category=recent_pred["risk_category"]
                    ))
                    continue

            # Generate new prediction
            prediction = await _generate_prediction(db, worker_id)

            results.append(BatchPredictionResult(
                worker_id=worker_id,
                status="success",
                risk_category=prediction["risk_category"]
            ))

        except Exception as e:
            results.append(BatchPredictionResult(
                worker_id=worker_id,
                status="error",
                error=str(e)
            ))

    successful = len([r for r in results if r.status == "success"])
    failed = len([r for r in results if r.status == "error"])

    return BatchPredictionResponse(
        total_workers=len(workers),
        successful=successful,
        failed=failed,
        results=results
    )


@router.get("/trends", response_model=PredictionTrends)
async def get_prediction_trends(
    mine_id: Optional[str] = None,
    days_back: int = Query(30, ge=7, le=90),
    current_user: dict = Depends(get_manager_or_above)
):
    """
    Get trends in risk scores over time.
    Shows how risk levels are changing.
    """
    db = get_database()

    # Check mine access
    if mine_id and not check_mine_access(current_user, mine_id):
        raise HTTPException(status_code=403, detail="No access to this mine")

    # Build match query
    match_query = {
        "prediction_date": {
            "$gte": datetime.utcnow() - timedelta(days=days_back)
        }
    }

    # Filter by accessible mines
    if mine_id:
        workers = await db.workers.find({"mine_id": ObjectId(mine_id)}).to_list(length=None)
        worker_ids = [str(w["_id"]) for w in workers]
        match_query["worker_id"] = {"$in": worker_ids}
    else:
        accessible_mine_ids = _get_accessible_mine_ids(current_user, None)
        if accessible_mine_ids is not None:
            workers = await db.workers.find({
                "mine_id": {"$in": [ObjectId(mid) for mid in accessible_mine_ids]}
            }).to_list(length=None)
            worker_ids = [str(w["_id"]) for w in workers]
            match_query["worker_id"] = {"$in": worker_ids}

    # Aggregation pipeline
    pipeline = [
        {"$match": match_query},
        {
            "$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$prediction_date"}},
                    "risk_category": "$risk_category"
                },
                "count": {"$sum": 1},
                "avg_risk_score": {"$avg": "$overall_risk_score"}
            }
        },
        {"$sort": {"_id.date": 1}}
    ]

    results = await db.predictions.aggregate(pipeline).to_list(length=None)

    # Format for frontend
    trend_data = {}
    for result in results:
        date = result["_id"]["date"]
        category = result["_id"]["risk_category"]

        if date not in trend_data:
            trend_data[date] = {}

        trend_data[date][category] = {
            "count": result["count"],
            "avg_risk_score": round(result["avg_risk_score"], 1)
        }

    return PredictionTrends(trends=trend_data)


# ==================== Helper Functions ====================

async def _generate_prediction(db, worker_id: str) -> dict:
    """Generate a new prediction for a worker"""
    from ml.prediction_service import PredictionService

    service = PredictionService(db)
    prediction = await service.predict_worker_risk(worker_id)

    # Save to database
    result = await db.predictions.insert_one(prediction)
    prediction["_id"] = result.inserted_id

    # Create alert if critical risk
    if prediction["risk_category"] == "critical" or prediction["requires_intervention"]:
        await _create_risk_alert(db, worker_id, prediction)

    return prediction


async def _create_risk_alert(db, worker_id: str, prediction: dict):
    """Create an alert for high-risk workers"""
    # Check if alert already exists today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    existing_alert = await db.alerts.find_one({
        "worker_id": worker_id,
        "alert_type": "worker_risk_prediction",
        "created_at": {"$gte": today_start}
    })

    if existing_alert:
        return  # Don't create duplicate

    worker = await db.workers.find_one({"_id": ObjectId(worker_id)})
    if not worker:
        return

    # Get primary risk factor
    risk_factors = prediction.get("risk_factors", [])
    primary_issue = risk_factors[0]["description"] if risk_factors else "Multiple risk indicators"

    risk_score = prediction["overall_risk_score"]
    risk_category = prediction["risk_category"]

    alert_doc = {
        "alert_type": "worker_risk_prediction",
        "severity": "critical" if risk_category == "critical" else "high",
        "status": "active",
        "message": f"Worker {worker['name']} ({worker['employee_id']}) flagged as {risk_category.upper()} risk (score: {risk_score:.0f}/100). {primary_issue}",
        "mine_id": worker.get("mine_id"),
        "zone_id": worker.get("zone_id"),
        "worker_id": worker_id,
        "metadata": {
            "risk_score": risk_score,
            "risk_category": risk_category,
            "predicted_violations": prediction["predicted_violations_count"],
            "attendance_rate": prediction["attendance_rate_30d"],
            "requires_intervention": prediction["requires_intervention"]
        },
        "created_at": datetime.utcnow(),
        "created_by": "system_ml_predictions"
    }

    await db.alerts.insert_one(alert_doc)


def _get_accessible_mine_ids(current_user: dict, requested_mine_id: Optional[str]) -> Optional[list]:
    """Get list of mine IDs accessible to the current user"""
    role = current_user.get("role")

    # Super admin can access all mines
    if role == "super_admin":
        return None  # None means all mines

    # If specific mine requested, verify access
    if requested_mine_id:
        if check_mine_access(current_user, requested_mine_id):
            return [requested_mine_id]
        else:
            raise HTTPException(status_code=403, detail="No access to this mine")

    # Area safety officer has multiple mines
    if role == "area_safety_officer":
        mine_ids = current_user.get("mine_ids", [])
        return [str(mid) for mid in mine_ids]

    # Other roles have single mine
    mine_id = current_user.get("mine_id")
    if mine_id:
        return [str(mine_id)]

    return []
