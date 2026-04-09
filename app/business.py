import math
import pandas as pd


TRUCK_CAPACITIES = {
    "10t": 12,
    "20t_82": 33,
    "20t_90": 33,
    "20t_120": 36,
}

def calculate_required_capacity(predicted_target_2h: float, reserve_ratio: float = 0.1) -> int:
    return int(math.ceil(float(predicted_target_2h) * (1.0 + reserve_ratio)))

def allocate_vehicles(required_capacity: int, fleet: dict):
    capacities = {
        "20t_120": 36,
        "20t_90": 33,
        "20t_82": 33,
        "10t": 12,
    }

    allocation = {
        "10t": 0,
        "20t_82": 0,
        "20t_90": 0,
        "20t_120": 0,
    }

    remaining = required_capacity

    for truck_type, cap in capacities.items():
        available = int(fleet.get(truck_type, 0))
        if available <= 0:
            continue

        need = remaining // cap
        if remaining % cap != 0:
            need += 1

        use = min(need, available)
        allocation[truck_type] = use
        remaining -= use * cap

        if remaining <= 0:
            remaining = 0
            break

    return allocation, remaining

def calculate_additional_vehicles_to_call(remaining_capacity: int) -> int:
    if remaining_capacity <= 0:
        return 0
    return math.ceil(remaining_capacity / 36)

def calculate_required_vehicles(predicted_volume, vehicle_capacity, reserve_ratio=0.1):
    adjusted_volume = float(predicted_volume) * (1.0 + float(reserve_ratio))
    return int(math.ceil(adjusted_volume / float(vehicle_capacity)))


def build_dispatch_plan(
    pred_df: pd.DataFrame,
    route_to_office: dict,
    vehicle_capacity: float = 60.0,
    reserve_ratio: float = 0.1,
):
    work_df = pred_df.copy()
    work_df["office_from_id"] = work_df["route_id"].map(route_to_office)

    office_df = (
        work_df
        .groupby(["timestamp", "office_from_id"], as_index=False)["prediction"]
        .sum()
        .rename(columns={"prediction": "predicted_volume"})
    )

    office_df["required_vehicles"] = office_df["predicted_volume"].apply(
        lambda x: calculate_required_vehicles(
            predicted_volume=x,
            vehicle_capacity=vehicle_capacity,
            reserve_ratio=reserve_ratio,
        )
    )

    return office_df

def build_dispatch_decision_for_point(
    predicted_target_2h: float,
    fleet_10t: int = 0,
    fleet_20t_82: int = 0,
    fleet_20t_90: int = 0,
    fleet_20t_120: int = 0,
    reserve_ratio: float = 0.1,
):
    required_capacity = calculate_required_capacity(
        predicted_target_2h=predicted_target_2h,
        reserve_ratio=reserve_ratio,
    )

    fleet = {
        "10t": fleet_10t,
        "20t_82": fleet_20t_82,
        "20t_90": fleet_20t_90,
        "20t_120": fleet_20t_120,
    }

    allocation, remaining_capacity = allocate_vehicles(
        required_capacity=required_capacity,
        fleet=fleet,
    )

    additional_vehicles_to_call = calculate_additional_vehicles_to_call(remaining_capacity)

    return {
        "predicted_target_2h": float(predicted_target_2h),
        "required_capacity": int(required_capacity),
        "allocated_vehicles": allocation,
        "remaining_capacity": int(remaining_capacity),
        "additional_vehicles_to_call": int(additional_vehicles_to_call),
    }