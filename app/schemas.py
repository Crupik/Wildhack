from pydantic import BaseModel
from typing import List


class HistoryRow(BaseModel):
    route_id: int
    timestamp: str
    target_2h: float


class ForecastRequest(BaseModel):
    history: List[HistoryRow]


class PointForecastRequest(BaseModel):
    route_id: int
    timestamp: str
    fleet_10t: int = 0
    fleet_20t_82: int = 0
    fleet_20t_90: int = 0
    fleet_20t_120: int = 0