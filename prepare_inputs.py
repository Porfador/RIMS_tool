"""
Prepares RIMS simulation inputs from a .xes event log:
  1. Discovers Petri net via inductive miner -> saves as .pnml
  2. Mines processing times, inter-arrival times, resources -> saves skeleton .json

Usage (from Anaconda Prompt, in the project root):
  python prepare_inputs.py --xes path/to/log.xes --out path/to/output_folder

The output folder will contain:
  - <xes_name>.pnml
  - <xes_name>_parameters.json   (fill in the parts marked TODO)
"""

import argparse
import json
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pm4py

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_distribution(series_seconds: pd.Series) -> dict:
    """Fits a simple distribution to a series of durations (in seconds).
    Returns a pm4py-compatible distribution dict."""
    s = series_seconds.dropna()
    s = s[s > 0]
    if len(s) < 2:
        return {"name": "uniform", "parameters": {"low": 0, "high": 3600}}

    mean = float(s.mean())
    std = float(s.std())

    # Exponential: std ≈ mean
    if std > 0 and abs(std - mean) / mean < 0.3:
        return {"name": "exponential", "parameters": {"scale": round(mean, 1)}}

    # Normal: symmetric, std < mean
    if std < mean:
        return {"name": "normal", "parameters": {"loc": round(mean, 1), "scale": round(std, 1)}}

    # Fallback: uniform between p10 and p90
    low = max(0.0, float(np.percentile(s, 10)))
    high = float(np.percentile(s, 90))
    return {"name": "uniform", "parameters": {"low": round(low, 1), "high": round(high, 1)}}


def mine_inter_arrival(log_df: pd.DataFrame) -> dict:
    """Estimates inter-arrival distribution from case start times (seconds)."""
    case_starts = (
        log_df.groupby("case:concept:name")["time:timestamp"]
        .min()
        .sort_values()
    )
    deltas = case_starts.diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if len(deltas) < 2:
        return {"type": "distribution", "name": "exponential", "parameters": {"scale": 86400}}
    dist = fit_distribution(deltas)
    dist["type"] = "distribution"
    return dist


def extract_resources(log_df: pd.DataFrame):
    """Returns (resource_col, role_col) column names found in the log, or None."""
    res_col = next(
        (c for c in log_df.columns if c in ("org:resource", "Resource", "resource")),
        None,
    )
    role_col = next(
        (c for c in log_df.columns if c in ("org:role", "Role", "role")),
        None,
    )
    return res_col, role_col


def build_resource_section(log_df: pd.DataFrame, res_col, role_col):
    """Builds resource and resource_table sections."""
    if res_col is None:
        # No resource info — create a placeholder
        activities = log_df["concept:name"].unique().tolist()
        resource_table = [{"role": "Role 1", "task": act} for act in activities]
        resource = {
            "Role 1": {
                "resources": ["Resource_1", "Resource_2"],
                "calendar": {"days": [0, 1, 2, 3, 4], "hour_min": 8, "hour_max": 17},
            }
        }
        return resource, resource_table

    if role_col is not None:
        # Log has explicit roles
        act_role = (
            log_df[["concept:name", role_col]]
            .dropna()
            .drop_duplicates()
            .set_index("concept:name")[role_col]
            .to_dict()
        )
        roles = log_df[[role_col, res_col]].dropna().drop_duplicates()
        resource = {}
        for role, grp in roles.groupby(role_col):
            resources = grp[res_col].unique().tolist()
            resource[str(role)] = {
                "resources": [str(r) for r in resources],
                "calendar": {
                    "days": [0, 1, 2, 3, 4],   # TODO: adjust
                    "hour_min": 8,              # TODO: adjust
                    "hour_max": 17,             # TODO: adjust
                },
            }
        resource_table = [
            {"role": str(v), "task": k} for k, v in act_role.items()
        ]
    else:
        # No role column — group by activity: one role per activity, all resources that did it
        act_res = (
            log_df[["concept:name", res_col]]
            .dropna()
            .drop_duplicates()
        )
        # Build activity -> role name, and collect all resources per role
        resource = {}
        resource_table = []
        for act, grp in act_res.groupby("concept:name"):
            role_name = "Role_" + act.replace(" ", "_").replace("&", "and").replace(".", "")
            resources_for_role = [str(r) for r in grp[res_col].unique()]
            resource[role_name] = {
                "resources": resources_for_role,
                "calendar": {
                    "days": [0, 1, 2, 3, 4],   # TODO: adjust
                    "hour_min": 8,              # TODO: adjust
                    "hour_max": 17,             # TODO: adjust
                },
            }
            resource_table.append({"role": role_name, "task": str(act)})

    return resource, resource_table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xes", required=True, help="Path to input .xes file")
    parser.add_argument("--out", required=True, help="Output folder path")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    xes_path = args.xes
    out_folder = args.out
    base_name = os.path.splitext(os.path.basename(xes_path))[0]

    print(f"[1/4] Reading log: {xes_path}")
    log = pm4py.read_xes(xes_path)
    log_df = pm4py.convert_to_dataframe(log)

    # Drop artificial Start/End activities if present
    log_df = log_df[~log_df["concept:name"].isin(["Start", "End"])]

    # Filter out start/complete lifecycle for Petri net discovery (use complete only)
    has_lifecycle = "lifecycle:transition" in log_df.columns
    if has_lifecycle:
        lifecycle_values = log_df["lifecycle:transition"].str.lower().unique()
        has_start_complete = {"start", "complete"}.issubset(set(lifecycle_values))
    else:
        has_start_complete = False

    if has_start_complete:
        log_complete = log_df[log_df["lifecycle:transition"].str.lower() == "complete"].copy()
        print(f"      Lifecycle: start+complete detected — using 'complete' events for Petri net")
    else:
        log_complete = log_df.copy()

    print(f"      {len(log_df)} events | "
          f"{log_df['case:concept:name'].nunique()} cases | "
          f"{log_df['concept:name'].nunique()} activities")

    # -- Petri net --
    print("[2/4] Discovering Petri net (inductive miner)...")
    log_for_pn = pm4py.convert_to_event_log(log_complete)
    net, im, fm = pm4py.discover_petri_net_inductive(log_for_pn)
    pnml_path = os.path.join(out_folder, f"{base_name}.pnml")
    pm4py.write_pnml(net, im, fm, pnml_path)
    print(f"      Saved: {pnml_path}")

    # -- Processing times --
    print("[3/4] Mining processing times per activity...")
    log_df = log_df.sort_values(["case:concept:name", "concept:name", "time:timestamp"])

    if has_start_complete:
        # Pair start/complete events to get real processing duration
        starts = log_df[log_df["lifecycle:transition"].str.lower() == "start"][
            ["case:concept:name", "concept:name", "time:timestamp", "org:resource"]
        ].copy()
        completes = log_df[log_df["lifecycle:transition"].str.lower() == "complete"][
            ["case:concept:name", "concept:name", "time:timestamp"]
        ].copy()
        starts = starts.rename(columns={"time:timestamp": "ts_start"})
        completes = completes.rename(columns={"time:timestamp": "ts_complete"})
        starts["_rank"] = starts.groupby(["case:concept:name", "concept:name"]).cumcount()
        completes["_rank"] = completes.groupby(["case:concept:name", "concept:name"]).cumcount()
        paired = pd.merge(starts, completes, on=["case:concept:name", "concept:name", "_rank"])
        paired["duration_s"] = (paired["ts_complete"] - paired["ts_start"]).dt.total_seconds()
        duration_df = paired[["concept:name", "duration_s", "org:resource"]]
        # For inter-arrival, use complete events only
        inter_df = log_complete
    else:
        log_df["duration_s"] = (
            log_df.groupby("case:concept:name")["time:timestamp"].diff().dt.total_seconds()
        )
        duration_df = log_df[["concept:name", "duration_s"]].copy()
        if "org:resource" in log_df.columns:
            duration_df["org:resource"] = log_df["org:resource"]
        inter_df = log_df

    processing_time = {}
    for act, grp in duration_df.groupby("concept:name"):
        processing_time[str(act)] = fit_distribution(grp["duration_s"])

    # -- Inter-arrival --
    print("[4/4] Estimating inter-arrival distribution...")
    inter_trigger = mine_inter_arrival(inter_df)

    # Calendar placeholder
    inter_trigger["calendar"] = {
        "days": [0, 1, 2, 3, 4],  # TODO: 0=Mon … 6=Sun
        "hour_min": 8,
        "hour_max": 17,
    }

    # -- Resources --
    res_col, role_col = extract_resources(duration_df if has_start_complete else log_df)
    resource, resource_table = build_resource_section(
        duration_df if has_start_complete else log_df, res_col, role_col
    )

    # -- Timestamps --
    start_ts = log_df["time:timestamp"].min()
    end_ts = log_df["time:timestamp"].max()
    duration_days = max(1, int((end_ts - start_ts).days))

    # -- Assemble JSON --
    params = {
        "start_timestamp": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_simulation": duration_days,
        "interTriggerTimer": inter_trigger,
        "processing_time": processing_time,
        "probability": {},    # Leave empty for AUTO, or add gateway names with "AUTO"/"CUSTOM"
        "resource": resource,
        "resource_table": resource_table,
    }

    json_path = os.path.join(out_folder, f"{base_name}_parameters.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4, ensure_ascii=False)
    print(f"      Saved: {json_path}")

    # -- Summary --
    print("\n=== DONE ===")
    print(f"  Petri net : {pnml_path}")
    print(f"  Parameters: {json_path}")
    print()
    print("TODO before running RIMS:")
    if res_col is None:
        print("  [!] Log has no resource column — placeholder roles created. Update 'resource' and 'resource_table' manually.")
    if role_col is None and res_col is not None:
        print("  [!] Log has no role column — each resource was mapped to its own role. Merge roles in 'resource' if needed.")
    print("  [ ] Check working calendars (days, hour_min, hour_max) in 'resource' and 'interTriggerTimer'.")
    print("  [ ] Add decision point names to 'probability' with value 'AUTO' (or a float) if your process has gateways.")
    print(f"  [ ] Adjust 'duration_simulation' (currently {duration_days} days) if you want a shorter/longer simulation.")


if __name__ == "__main__":
    main()
