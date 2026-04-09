import os
import pickle
import joblib
from catboost import CatBoostRegressor

from app.config import ARTIFACTS_DIR, HORIZON
from app.types import StackCalibration


class ArtifactUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "StackCalibration":
            return StackCalibration
        return super().find_class(module, name)


def load_pickle(path):
    with open(path, "rb") as f:
        return ArtifactUnpickler(f).load()


def load_catboost_model(path):
    model = CatBoostRegressor()
    model.load_model(path)
    return model


def load_models(models_dir):
    model_abs_by_h = {}
    model_delta_by_h = {}
    model_ridge_by_h = {}
    model_two_stage_total_by_h = {}
    model_two_stage_share_by_h = {}

    for h in range(HORIZON):
        model_abs_by_h[h] = load_catboost_model(
            os.path.join(models_dir, f"catboost_abs_h{h}.cbm")
        )
        model_delta_by_h[h] = joblib.load(
            os.path.join(models_dir, f"lgbm_delta_h{h}.pkl")
        )
        model_ridge_by_h[h] = joblib.load(
            os.path.join(models_dir, f"ridge_direct_h{h}.pkl")
        )
        model_two_stage_total_by_h[h] = joblib.load(
            os.path.join(models_dir, f"two_stage_total_h{h}.pkl")
        )
        model_two_stage_share_by_h[h] = load_catboost_model(
            os.path.join(models_dir, f"two_stage_share_h{h}.cbm")
        )

    return {
        "abs": model_abs_by_h,
        "delta": model_delta_by_h,
        "ridge": model_ridge_by_h,
        "two_stage_total": model_two_stage_total_by_h,
        "two_stage_share": model_two_stage_share_by_h,
    }


def load_meta(meta_dir):
    return load_pickle(os.path.join(meta_dir, "meta_artifacts.pkl"))


def load_all_artifacts():
    models_dir = os.path.join(ARTIFACTS_DIR, "models")
    meta_dir = os.path.join(ARTIFACTS_DIR, "meta")

    meta = load_meta(meta_dir)
    models = load_models(models_dir)

    return {
        "meta": meta,
        "models": models,
    }