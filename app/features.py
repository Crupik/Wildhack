import numpy as np
import pandas as pd
from app.types import CutoffCache
from app.config import SLOT_COUNT
from app.config import HORIZON


def build_route_frame(
    values: np.ndarray,
    slots: np.ndarray,
    total: np.ndarray,
    cut: int,
    h: int,
    cache: CutoffCache,
) -> pd.DataFrame:
    ti = cut + h
    s = int(slots[ti])
    prev_total = float(total[cut - 1])
    slot_total_mean = float(cache.total_slot_mean[s])
    route_sum = (
        0.35 * cache.lag1
        + 0.20 * cache.lag2
        + 0.15 * cache.lag3
        + 0.10 * cache.lag4
        + 0.10 * cache.mean4
        + 0.05 * cache.mean8
        + 0.05 * cache.slot_mean[s]
    )
    route_share_guess = route_sum / np.clip(route_sum.sum(), 1e-6, None)

    df = pd.DataFrame({
        'lag1': cache.lag1,
        'lag2': cache.lag2,
        'lag3': cache.lag3,
        'lag4': cache.lag4,
        'mean4': cache.mean4,
        'mean8': cache.mean8,
        'mean16': cache.mean16,
        'std8': cache.std8,
        'std16': cache.std16,
        'route_mean': cache.route_mean,
        'route_mean_share': cache.route_mean_share,
        'slot_mean': cache.slot_mean[s],
        'slot_mean_share': cache.slot_mean_share[s],
        'prev_total': prev_total,
        'slot_total_mean': slot_total_mean,
        'route_share_guess': route_share_guess,
        'h': h,
        'slot': s,
    })
    return df

def build_total_frame(
    values: np.ndarray,
    times: pd.DatetimeIndex,
    slots: np.ndarray,
    totals: np.ndarray,
    cache_map,
    cutoffs,
    h: int,
) -> pd.DataFrame:
    rows = []
    for cut in cutoffs:
        ti = cut + h
        ts = times[ti]
        s = int(slots[ti])
        cc = cache_map[cut]

        row = {
            "cut_idx": cut,
            "dow": int(ts.dayofweek),
            "month": int(ts.month),
            "day": int(ts.day),
            "h": h,
            "total_lag1": totals[cut - 1],
            "total_lag2": totals[cut - 2],
            "total_lag4_mean": totals[cut - 4:cut].mean(),
            "total_lag8_mean": totals[cut - 8:cut].mean(),
            "total_lag16_mean": totals[cut - 16:cut].mean(),
            "total_lag48": totals[ti - 48],
            "total_lag96": totals[ti - 96],
            "total_lag144": totals[ti - 144],
            "total_lag336": totals[ti - 336],
            "total_slot_mean": cc.total_slot_mean[s],
            "total_trend_1d": totals[ti - 48] - totals[ti - 96],
            "total_trend_1w": totals[ti - 48] - totals[ti - 336],
            "target_total": totals[ti],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def precompute_cutoff_cache(
    values: np.ndarray,
    slots: np.ndarray,
    totals: np.ndarray,
    cutoffs,
):
    cache_map = {}
    total_slot_mean_global = (
        pd.DataFrame({"slot": slots, "total": totals})
        .groupby("slot")["total"].mean()
        .reindex(range(SLOT_COUNT))
        .fillna(0.0)
        .to_numpy()
    )

    for cut in cutoffs:
        hist = values[:cut]
        hist_slots = slots[:cut]

        lag1 = hist[-1]
        lag2 = hist[-2]
        lag3 = hist[-3]
        lag4 = hist[-4]

        mean4 = hist[-4:].mean(axis=0)
        mean8 = hist[-8:].mean(axis=0)
        mean16 = hist[-16:].mean(axis=0)

        std8 = hist[-8:].std(axis=0)
        std16 = hist[-16:].std(axis=0)

        route_mean = hist.mean(axis=0)

        hist_total = hist.sum(axis=1, keepdims=True)
        hist_share = np.divide(hist, np.clip(hist_total, 1e-6, None))
        route_mean_share = hist_share.mean(axis=0)

        slot_mean = np.zeros((SLOT_COUNT, hist.shape[1]), dtype=float)
        slot_mean_share = np.zeros((SLOT_COUNT, hist.shape[1]), dtype=float)

        for s in range(SLOT_COUNT):
            mask = hist_slots == s
            if mask.any():
                slot_mean[s] = hist[mask].mean(axis=0)
                slot_mean_share[s] = hist_share[mask].mean(axis=0)
            else:
                slot_mean[s] = route_mean
                slot_mean_share[s] = route_mean_share

        cache_map[cut] = CutoffCache(
            lag1=lag1,
            lag2=lag2,
            lag3=lag3,
            lag4=lag4,
            mean4=mean4,
            mean8=mean8,
            mean16=mean16,
            std8=std8,
            std16=std16,
            route_mean=route_mean,
            route_mean_share=route_mean_share,
            slot_mean=slot_mean,
            slot_mean_share=slot_mean_share,
            total_slot_mean=total_slot_mean_global,
        )
    return cache_map


def collect_inference_dataset(
    values: np.ndarray,
    times: pd.DatetimeIndex,
    slots: np.ndarray,
    totals: np.ndarray,
    cutoffs,
    route_groups: np.ndarray,
):
    cache_map = precompute_cutoff_cache(values, slots, totals, cutoffs)

    by_h = {}
    for h in range(HORIZON):
        route_frames = []
        for cut in cutoffs:
            rf = build_route_frame(values, slots, totals, cut, h, cache_map[cut]).copy()
            rf["cut_idx"] = cut
            rf["route_group"] = route_groups
            rf["target"] = values[cut + h]
            rf["target_share"] = values[cut + h] / np.clip(totals[cut + h], 1e-6, None)
            route_frames.append(rf)
        route_df = pd.concat(route_frames, axis=0, ignore_index=True)

        total_df = build_total_frame(values, times, slots, totals, cache_map, cutoffs, h)
        by_h[h] = {"route": route_df, "total": total_df}

    return by_h

def choose_cutoffs(
    n_time: int,
    horizon: int,
    start: int = 24 * 14,
    end_margin: int = 1,
    step: int = 6,
):
    cutoffs = list(range(start, n_time - horizon - end_margin, step))
    return [c for c in cutoffs if c - 336 >= 0]