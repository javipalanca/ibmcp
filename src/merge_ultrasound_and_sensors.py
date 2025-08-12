import os
from typing import Tuple

import numpy as np
import pandas as pd


ULTRA_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_task3", "plant_ultrasound_dataset.csv"))
SENSOR1_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sensordata1_all.csv"))
SENSOR2_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sensordata2_all.csv"))
OUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_task3", "plant_ultrasound_dataset_with_sensors.csv"))


def read_ultrasound(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "recording_time" not in df.columns:
        raise ValueError("recording_time column not found in ultrasound dataset")
    # Keep original order
    df["_row_idx"] = np.arange(len(df))
    # Parse recording_time as naive local datetime
    df["recording_time"] = pd.to_datetime(df["recording_time"], errors="coerce")
    missing = df["recording_time"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows in ultrasound dataset have invalid recording_time and will be kept with NaN sensors.")
    return df


def read_sensor(path: str) -> pd.DataFrame:
    # Some rows may be empty or partial; parse created_at and keep as naive local wall time
    df = pd.read_csv(path)
    if "created_at" not in df.columns:
        raise ValueError(f"created_at column not found in sensor file: {path}")
    # Parse as timezone-aware if tz info is present, then drop tz to compare as local wall time
    # Use utc=True to safely parse mixed offsets, then convert to naive by removing tz
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    # Convert to naive localized wall time by dropping tz (keeps UTC clock time). For nearest-match within 60min, this is acceptable.
    df["created_at"] = df["created_at"].dt.tz_localize(None)
    # Drop rows with no timestamp
    df = df.dropna(subset=["created_at"]).copy()
    # Sort by time for asof join
    df = df.sort_values("created_at").reset_index(drop=True)

    # Normalize column names to stable sensor_* names
    rename_map = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped.startswith("field1"):
            rename_map[col] = "sensor_temp_c"
        elif col_stripped.startswith("field2"):
            rename_map[col] = "sensor_humidity"
        elif col_stripped.startswith("field3"):
            rename_map[col] = "sensor_voltage"
        elif col_stripped.startswith("field4"):
            rename_map[col] = "sensor_wifi_rssi"
        elif col_stripped.startswith("field5"):
            rename_map[col] = "sensor_co2"
        elif col_stripped.startswith("field6"):
            rename_map[col] = "sensor_light"
        elif col_stripped.startswith("field7"):
            rename_map[col] = "sensor_soil_temp"
        elif col_stripped.startswith("field8"):
            rename_map[col] = "sensor_soil_moisture"
        elif col_stripped.startswith("field9"):
            rename_map[col] = "sensor_soil_ph"
        elif col_stripped.startswith("field10"):
            rename_map[col] = "sensor_pressure"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Keep only time and numeric columns to avoid weird strings
    keep_cols = ["created_at"] + [c for c in df.columns if c.startswith("sensor_")]
    df = df[keep_cols]
    return df


def asof_merge(ultra_subset: pd.DataFrame, sensor_df: pd.DataFrame, source_label: str, tolerance_minutes: int = 60) -> pd.DataFrame:
    # Ensure both sides have proper datetime and are sorted
    left = ultra_subset.copy()
    # Coerce to datetime to avoid object dtype issues
    left["recording_time"] = pd.to_datetime(left["recording_time"], errors="coerce")
    left = left.sort_values("recording_time").copy()
    right = sensor_df.sort_values("created_at").copy()

    merged = pd.merge_asof(
        left,
        right,
        left_on="recording_time",
        right_on="created_at",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )

    # Compute time delta in minutes (absolute)
    merged["sensor_time"] = merged["created_at"]
    merged.drop(columns=["created_at"], inplace=True)
    merged["sensor_delta_minutes"] = (
        (merged["recording_time"] - merged["sensor_time"]).abs().dt.total_seconds() / 60.0
    )
    merged["sensor_source"] = source_label

    # Restore original order
    merged = merged.sort_values("_row_idx").reset_index(drop=True)
    return merged


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    # If duplicate column names exist, coalesce values into the first occurrence
    cols = pd.Index(df.columns)
    dup_names = cols[cols.duplicated()].unique().tolist()
    if not dup_names:
        return df
    out = df
    for name in dup_names:
        # Positions of columns with the same name
        idxs = [i for i, c in enumerate(out.columns) if c == name]
        if len(idxs) <= 1:
            continue
        sub = out.iloc[:, idxs]
        # Prefer first non-null across columns, left to right
        coalesced = sub.bfill(axis=1).iloc[:, 0]
        # Assign to the first column position and drop the rest
        out.iloc[:, idxs[0]] = coalesced
        out = out.drop(columns=[out.columns[i] for i in idxs[1:]])
    return out


def main(
    ultra_csv: str = ULTRA_CSV,
    sensor1_csv: str = SENSOR1_CSV,
    sensor2_csv: str = SENSOR2_CSV,
    out_csv: str = OUT_CSV,
) -> Tuple[str, int]:
    print(f"Reading ultrasound dataset: {ultra_csv}")
    ultra = read_ultrasound(ultra_csv)

    print(f"Reading sensor1: {sensor1_csv}")
    s1 = read_sensor(sensor1_csv)
    print(f"Sensor1 rows: {len(s1)} time range: {s1['created_at'].min()} -> {s1['created_at'].max()}")

    print(f"Reading sensor2: {sensor2_csv}")
    s2 = read_sensor(sensor2_csv)
    print(f"Sensor2 rows: {len(s2)} time range: {s2['created_at'].min()} -> {s2['created_at'].max()}")

    # Split ultrasound by channel groups
    ch = ultra["channel"].astype(str).str.lower()
    mask_s1 = ch.isin(["ch1", "ch2"])  # sensor1 for ch1/ch2
    mask_s2 = ch.isin(["ch3", "ch4"])  # sensor2 for ch3/ch4

    ultra_s1 = ultra[mask_s1].copy()
    ultra_s2 = ultra[mask_s2].copy()
    ultra_other = ultra[~(mask_s1 | mask_s2)].copy()
    if len(ultra_other) > 0:
        print(f"Warning: {len(ultra_other)} rows with unknown channel; sensor data will be NaN for these rows.")

    merged_parts = []
    if len(ultra_s1) > 0:
        print(f"Merging {len(ultra_s1)} rows with sensor1 by nearest time (<=60 min)...")
        merged_parts.append(asof_merge(ultra_s1, s1, source_label="sensor1"))
    if len(ultra_s2) > 0:
        print(f"Merging {len(ultra_s2)} rows with sensor2 by nearest time (<=60 min)...")
        merged_parts.append(asof_merge(ultra_s2, s2, source_label="sensor2"))
    if len(ultra_other) > 0:
        # Add empty sensor columns for consistency
        empty_cols = [
            "sensor_temp_c",
            "sensor_humidity",
            "sensor_voltage",
            "sensor_wifi_rssi",
            "sensor_co2",
            "sensor_light",
            "sensor_soil_temp",
            "sensor_soil_moisture",
            "sensor_soil_ph",
            "sensor_pressure",
            "sensor_time",
            "sensor_delta_minutes",
            "sensor_source",
        ]
        for c in empty_cols:
            ultra_other[c] = np.nan
        merged_parts.append(ultra_other)

    merged_all = pd.concat(merged_parts, ignore_index=True).sort_values("_row_idx").reset_index(drop=True)
    # Deduplicate any duplicate columns (keep first non-null)
    merged_all = _coalesce_duplicate_columns(merged_all)
    merged_all.drop(columns=["_row_idx"], inplace=True)

    # Write output
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    merged_all.to_csv(out_csv, index=False)
    print(f"Wrote augmented dataset: {out_csv} ({len(merged_all)} rows)")
    return out_csv, len(merged_all)


if __name__ == "__main__":
    main()
