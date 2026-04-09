import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.business import build_dispatch_plan, build_dispatch_decision_for_point
from app.predictor import Predictor
from app.schemas import ForecastRequest, PointForecastRequest
from app.history_store import HistoryStore

app = FastAPI(title="Transport Auto Dispatch Service")

predictor = Predictor()
history_store = HistoryStore("data/train_team_track.parquet")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    history_df = pd.DataFrame([row.model_dump() for row in req.history])
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])

    pred_df = predictor.predict(history_df).copy()
    pred_df["timestamp"] = pred_df["timestamp"].astype(str)

    return {
        "horizon": predictor.horizon,
        "predictions": pred_df.to_dict(orient="records"),
    }


@app.post("/dispatch-plan")
def dispatch_plan(req: ForecastRequest):
    history_df = pd.DataFrame([row.model_dump() for row in req.history])
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])

    pred_df = predictor.predict(history_df)

    plan_df = build_dispatch_plan(
        pred_df=pred_df,
        route_to_office=predictor.route_to_office,
        vehicle_capacity=60.0,
        reserve_ratio=0.1,
    ).copy()

    plan_df["timestamp"] = plan_df["timestamp"].astype(str)

    return {
        "horizon": predictor.horizon,
        "plans": plan_df.to_dict(orient="records"),
    }


@app.post("/predict-point")
def predict_point(req: PointForecastRequest):
    ts = pd.to_datetime(req.timestamp)

    office_from_id = predictor.route_to_office.get(req.route_id)
    if office_from_id is None:
        raise HTTPException(status_code=404, detail="Unknown route_id")

    history_df = history_store.get_history()
    last_ts = history_store.get_last_timestamp()

    delta_minutes = int((ts - last_ts).total_seconds() // 60)
    if delta_minutes <= 0 or delta_minutes % 30 != 0:
        raise HTTPException(status_code=400, detail="timestamp must be a future 30-minute slot")

    horizon_step = delta_minutes // 30
    if horizon_step < 1 or horizon_step > predictor.horizon:
        raise HTTPException(
            status_code=400,
            detail=f"timestamp is outside supported horizon 1..{predictor.horizon} steps",
        )

    pred_row = predictor.predict_point(
        history_df=history_df,
        route_id=req.route_id,
        timestamp=ts,
    )

    result = build_dispatch_decision_for_point(
        predicted_target_2h=float(pred_row["prediction"]),
        fleet_10t=req.fleet_10t,
        fleet_20t_82=req.fleet_20t_82,
        fleet_20t_90=req.fleet_20t_90,
        fleet_20t_120=req.fleet_20t_120,
    )

    result["route_id"] = req.route_id
    result["office_from_id"] = int(office_from_id)

    return result