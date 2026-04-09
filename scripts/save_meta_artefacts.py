import os
import pickle


def save_meta_artifacts(
    routes,
    route_groups,
    weights_by_h,
    weights_by_h_group,
    calib,
    route_to_office,
    horizon,
    slot_count,
    output_dir="artifactes/meta",
):
    os.makedirs(output_dir, exist_ok=True)

    meta_artifacts = {
        "routes": routes,
        "route_groups": route_groups,
        "weights_by_h": weights_by_h,
        "weights_by_h_group": weights_by_h_group,
        "calib": calib,
        "route_to_office": route_to_office,
        "horizon": horizon,
        "slot_count": slot_count,
    }

    output_path = os.path.join(output_dir, "meta_artifacts.pkl")

    with open(output_path, "wb") as f:
        pickle.dump(meta_artifacts, f)

    print(f"Saved meta artifactes to: {output_path}")