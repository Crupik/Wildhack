import numpy as np
import pandas as pd

from app.artefacts import load_all_artifacts
from app.types import StackCalibration


def apply_stack_calibration(y_pred: np.ndarray, h: int, calib: StackCalibration) -> np.ndarray:
    if calib.mode == "none":
        return np.clip(y_pred, 0, None)
    if calib.mode == "global_scale":
        return np.clip(y_pred * calib.global_scale, 0, None)
    if calib.mode == "horizon_scale":
        return np.clip(y_pred * float(calib.horizon_scale[h]), 0, None)
    if calib.mode == "horizon_affine":
        return np.clip(
            y_pred * float(calib.horizon_affine_a[h]) + float(calib.horizon_affine_b[h]),
            0,
            None,
        )
    raise ValueError(f"Unsupported calibration mode: {calib.mode}")


def combine_predictions(
    pred_by_model,
    weights_by_h,
    model_names,
    cut: int,
    h: int,
    route_groups: np.ndarray | None = None,
    weights_by_h_group=None,
    n_route_groups: int = 1,
    reconcile_total: float | None = None,
) -> np.ndarray:
    preds = [pred_by_model[name][(cut, h)] for name in model_names]
    X = np.vstack(preds).T

    if route_groups is None or n_route_groups <= 1 or not weights_by_h_group:
        y_hat = X @ weights_by_h[h]
    else:
        y_hat = np.zeros(X.shape[0], dtype=np.float64)
        for g in range(n_route_groups):
            idx = np.where(route_groups == g)[0]
            if len(idx) == 0:
                continue
            w = weights_by_h_group.get((h, g), weights_by_h[h])
            y_hat[idx] = X[idx] @ w

    y_hat = np.clip(y_hat, 0, None)
    if reconcile_total is not None:
        s = float(np.sum(y_hat))
        if s > 0:
            y_hat = y_hat * (float(reconcile_total) / s)
            y_hat = np.clip(y_hat, 0, None)

    return y_hat


class Predictor:
    def __init__(self):
        bundle = load_all_artifacts()

        self.meta = bundle["meta"]
        self.models = bundle["models"]

        self.model_abs_by_h = self.models["abs"]
        self.model_delta_by_h = self.models["delta"]
        self.model_ridge_by_h = self.models["ridge"]
        self.model_two_stage_total_by_h = self.models["two_stage_total"]
        self.model_two_stage_share_by_h = self.models["two_stage_share"]

        self.routes = self.meta["routes"]
        self.route_groups = self.meta["route_groups"]
        self.weights_by_h = self.meta["weights_by_h"]
        self.weights_by_h_group = self.meta["weights_by_h_group"]
        self.calib = self.meta["calib"]
        self.route_to_office = self.meta["route_to_office"]
        self.horizon = self.meta["horizon"]
        self.slot_count = self.meta["slot_count"]

        self.model_names = ["lag_formula", "cat_abs", "lgbm_delta", "two_stage", "ridge_direct"]

    def _prepare_history_arrays(self, history_df):
        panel = (
            history_df
            .pivot(index="timestamp", columns="route_id", values="target_2h")
            .sort_index()
            .astype(float)
        )

        panel = panel.reindex(columns=self.routes, fill_value=0.0)

        values = panel.to_numpy(dtype=float)
        times = panel.index
        slots = (times.hour * 2 + (times.minute // 30)).to_numpy(dtype=int)
        totals = values.sum(axis=1)

        return {
            "panel": panel,
            "values": values,
            "times": times,
            "slots": slots,
            "totals": totals,
        }

    def _prepare_single_cutoff(self, values):
        cut = len(values)

        if cut - 336 < 0:
            raise ValueError("Недостаточно истории для инференса: нужно минимум 336 временных точек.")

        return cut

    def _build_single_cache(self, values, slots, totals, cut):
        from app.features import precompute_cutoff_cache

        cache_map = precompute_cutoff_cache(
            values=values,
            slots=slots,
            totals=totals,
            cutoffs=[cut],
        )
        return cache_map[cut]

    def _build_feature_frames_for_h(self, values, times, totals, cut, cache, h):
        ts = times[-1] + pd.Timedelta(minutes=30 * (h + 1))
        s = int(ts.hour * 2 + (ts.minute // 30))
        ti = cut + h

        lag48 = values[ti - 48]
        lag96 = values[ti - 96]
        lag144 = values[ti - 144]
        lag336 = values[ti - 336]

        share = values / np.maximum(totals[:, None], 1e-9)
        lag48_share = share[ti - 48]
        lag336_share = share[ti - 336]

        route_df = pd.DataFrame(
            {
                "route_id": self.routes,
                "dow": int(ts.dayofweek),
                "month": int(ts.month),
                "day": int(ts.day),
                "h": h,
                "lag1": cache.lag1,
                "lag2": cache.lag2,
                "lag3": cache.lag3,
                "lag4": cache.lag4,
                "mean4": cache.mean4,
                "mean8": cache.mean8,
                "mean16": cache.mean16,
                "std8": cache.std8,
                "std16": cache.std16,
                "route_mean": cache.route_mean,
                "lag48": lag48,
                "lag96": lag96,
                "lag144": lag144,
                "lag336": lag336,
                "trend_1d": lag48 - lag96,
                "trend_1w": lag48 - lag336,
                "slot_mean": cache.slot_mean[s],
                "route_mean_share": cache.route_mean_share,
                "lag48_share": lag48_share,
                "lag336_share": lag336_share,
                "slot_mean_share": cache.slot_mean_share[s],
            }
        )

        total_df = pd.DataFrame(
            {
                "dow": [int(ts.dayofweek)],
                "month": [int(ts.month)],
                "day": [int(ts.day)],
                "h": [h],
                "total_lag1": [totals[cut - 1]],
                "total_lag2": [totals[cut - 2]],
                "total_lag4_mean": [totals[cut - 4:cut].mean()],
                "total_lag8_mean": [totals[cut - 8:cut].mean()],
                "total_lag16_mean": [totals[cut - 16:cut].mean()],
                "total_lag48": [totals[ti - 48]],
                "total_lag96": [totals[ti - 96]],
                "total_lag144": [totals[ti - 144]],
                "total_lag336": [totals[ti - 336]],
                "total_slot_mean": [cache.total_slot_mean[s]],
                "total_trend_1d": [totals[ti - 48] - totals[ti - 96]],
                "total_trend_1w": [totals[ti - 48] - totals[ti - 336]],
            }
        )

        return route_df, total_df

    def _predict_for_h(self, values, slots, cut, cache, route_df, total_df, h):
        feats_cat_abs = [
            "route_id", "dow", "month", "day", "h",
            "lag1", "lag2", "lag3", "lag4",
            "mean4", "mean8", "mean16",
            "std8", "std16",
            "route_mean",
            "lag48", "lag96", "lag144", "lag336",
            "trend_1d", "trend_1w",
            "slot_mean",
            "route_mean_share",
            "lag48_share", "lag336_share",
            "slot_mean_share",
        ]

        feats_lgbm_delta = [
            "dow", "month", "day", "h",
            "lag1", "lag2", "lag3", "lag4",
            "mean4", "mean8", "mean16",
            "std8", "std16",
            "route_mean",
            "lag48", "lag96", "lag144", "lag336",
            "trend_1d", "trend_1w",
            "slot_mean",
            "route_mean_share",
            "lag48_share", "lag336_share",
            "slot_mean_share",
            "route_id",
        ]

        feats_ridge = [
            "dow", "month", "day", "h",
            "lag1", "lag2", "lag3", "lag4",
            "mean4", "mean8", "mean16",
            "std8", "std16",
            "route_mean",
            "lag48", "lag96", "lag144", "lag336",
            "trend_1d", "trend_1w",
            "slot_mean",
            "route_mean_share",
            "lag48_share", "lag336_share",
            "slot_mean_share",
        ]

        feats_total = [
            "dow", "month", "day", "h",
            "total_lag1", "total_lag2",
            "total_lag4_mean", "total_lag8_mean", "total_lag16_mean",
            "total_lag48", "total_lag96", "total_lag144", "total_lag336",
            "total_slot_mean", "total_trend_1d", "total_trend_1w",
        ]

        pred_lag_formula = self._predict_lag_formula_for_cut(
            values=values,
            slots=slots,
            cut=cut,
            cache=cache,
            h=h,
        )

        pred_cat_abs = np.clip(
            self.model_abs_by_h[h].predict(route_df[feats_cat_abs]),
            0.0,
            None,
        )

        pred_lgbm_delta = np.clip(
            route_df["lag48"].to_numpy(dtype=float) +
            self.model_delta_by_h[h].predict(route_df[feats_lgbm_delta]),
            0.0,
            None,
        )

        pred_ridge = np.clip(
            self.model_ridge_by_h[h].predict(route_df[feats_ridge]),
            0.0,
            None,
        )

        total_pred = float(np.clip(
            self.model_two_stage_total_by_h[h].predict(total_df[feats_total])[0],
            0.0,
            None,
        ))

        share_raw = np.clip(
            self.model_two_stage_share_by_h[h].predict(route_df[feats_cat_abs]),
            0.0,
            None,
        )
        s = share_raw.sum()
        if s <= 0:
            norm_share = np.full(len(share_raw), 1.0 / len(share_raw), dtype=float)
        else:
            norm_share = share_raw / s
        pred_two_stage = np.clip(total_pred * norm_share, 0.0, None)

        pred_by_model = {
            "lag_formula": {(cut, h): pred_lag_formula},
            "cat_abs": {(cut, h): pred_cat_abs},
            "lgbm_delta": {(cut, h): pred_lgbm_delta},
            "two_stage": {(cut, h): pred_two_stage},
            "ridge_direct": {(cut, h): pred_ridge},
        }

        reconcile_total = total_pred

        pred = combine_predictions(
            pred_by_model=pred_by_model,
            weights_by_h=self.weights_by_h,
            model_names=self.model_names,
            cut=cut,
            h=h,
            route_groups=self.route_groups,
            weights_by_h_group=self.weights_by_h_group,
            n_route_groups=1 if len(np.unique(self.route_groups)) == 1 else len(np.unique(self.route_groups)),
            reconcile_total=reconcile_total,
        )

        pred = apply_stack_calibration(pred, h, self.calib)
        return pred

    @staticmethod
    def _predict_lag_formula_for_cut(
        values: np.ndarray,
        slots: np.ndarray,
        cut: int,
        cache,
        h: int,
        a: float = 0.6876323378629036,
        b: float = 0.15585209397332267,
        c: float = 0.0632273470621092,
        d: float = 0.12919920398783646,
    ) -> np.ndarray:
        last = cache.lag1.copy()
        for step in range(h + 1):
            ti = cut + step
            s = int(slots[ti]) if ti < len(slots) else int((s if step > 0 else 0))
            if ti >= len(values):
                p = a * last + b * values[ti - 48] + c * values[ti - 336] + d * cache.slot_mean[s]
            else:
                p = a * last + b * values[ti - 48] + c * values[ti - 336] + d * cache.slot_mean[s]
            p = np.clip(p, 0, None)
            last = p
        return last

    def predict_point(self, history_df, route_id: int, timestamp: pd.Timestamp):
        pred_df = self.predict(history_df)

        row = pred_df[
            (pred_df["route_id"] == route_id) &
            (pred_df["timestamp"] == timestamp)
            ]

        if row.empty:
            raise ValueError("Запрошенный timestamp не попадает в доступный горизонт прогноза.")

        return row.iloc[0]

    def predict(self, history_df):
        prepared = self._prepare_history_arrays(history_df)

        values = prepared["values"]
        times = prepared["times"]
        slots = prepared["slots"]
        totals = prepared["totals"]

        cut = self._prepare_single_cutoff(values)
        cache = self._build_single_cache(values, slots, totals, cut)

        last_ts = times[-1]
        result_rows = []

        for h in range(self.horizon):
            route_df, total_df = self._build_feature_frames_for_h(
                values=values,
                times=times,
                totals=totals,
                cut=cut,
                cache=cache,
                h=h,
            )

            pred = self._predict_for_h(
                values=values,
                slots=slots,
                cut=cut,
                cache=cache,
                route_df=route_df,
                total_df=total_df,
                h=h,
            )

            forecast_ts = last_ts + pd.Timedelta(minutes=30 * (h + 1))

            step_df = pd.DataFrame(
                {
                    "timestamp": forecast_ts,
                    "route_id": self.routes,
                    "prediction": pred,
                    "h": h,
                }
            )
            result_rows.append(step_df)

        result_df = pd.concat(result_rows, axis=0, ignore_index=True)
        return result_df