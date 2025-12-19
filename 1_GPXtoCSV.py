import os
import json
import numpy as np
import pandas as pd
import gpxpy
import folium

# ----------------------------
# Config
# ----------------------------
META_CSV = r"C:/ET_navigation/data/metadata.csv"
DATA_DIR = r"C:/ET_navigation/data"

# ----------------------------
# Load metadata (reldir + watch_lag)
# ----------------------------
meta_df = pd.read_csv(META_CSV, nrows=1)
reldirectory = str(meta_df.at[0, "reldir"])
watch_lag = float(meta_df.at[0, "watch_lag"])  # seconds (can be negative)

input_directory = os.path.join(DATA_DIR, reldirectory)
output_directory = os.path.join(input_directory, "output")
os.makedirs(output_directory, exist_ok=True)

gpx_file_path = os.path.join(input_directory, f"{reldirectory}.gpx")

# ----------------------------
# Helper: parse HR/CAD from GPX extensions (Garmin TrackPointExtension)
# ----------------------------
def extract_hr_cad_from_point(pt) -> tuple:
    """
    Returns (hr, cad) as floats (or np.nan if missing).
    gpxpy stores extensions as XML elements; we search by tag name suffix.
    """
    hr_val = np.nan
    cad_val = np.nan

    # pt.extensions is a list of XML elements
    # e.g. <gpxtpx:TrackPointExtension> ... <gpxtpx:hr>89</gpxtpx:hr> ...
    for ext in getattr(pt, "extensions", []) or []:
        # walk all descendants
        for child in list(ext.iter()):
            tag = child.tag.lower()
            text = child.text
            if text is None:
                continue
            if tag.endswith("}hr") or tag.endswith(":hr") or tag.endswith("hr"):
                try:
                    hr_val = float(text)
                except Exception:
                    pass
            elif tag.endswith("}cad") or tag.endswith(":cad") or tag.endswith("cad"):
                try:
                    cad_val = float(text)
                except Exception:
                    pass

    return hr_val, cad_val

# ----------------------------
# Parse GPX to raw rows
# ----------------------------
with open(gpx_file_path, "r", encoding="utf-8") as f:
    gpx = gpxpy.parse(f)

rows = []
for trk in gpx.tracks:
    for seg in trk.segments:
        for pt in seg.points:
            hr, cad = extract_hr_cad_from_point(pt)
            rows.append(
                {
                    "time": pt.time,  # datetime (timezone-aware in GPX)
                    "latitude": pt.latitude,
                    "longitude": pt.longitude,
                    "elevation": pt.elevation,
                    "hr": hr,
                    "cad": cad,
                }
            )

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError(f"No track points found in: {gpx_file_path}")

# Ensure datetime
df["time"] = pd.to_datetime(df["time"], utc=True)

# ----------------------------
# Make time unique to the second, then resample to 1 Hz
# ----------------------------
df["second"] = df["time"].dt.floor("S")
df = df.drop_duplicates(subset="second", keep="first").drop(columns="second")
df = df.sort_values("time").set_index("time")

# 1 Hz resample
df_1hz = df.resample("1S").asfreq()

# Interpolate continuous signals
df_1hz[["latitude", "longitude", "elevation"]] = df_1hz[["latitude", "longitude", "elevation"]].interpolate()

# HR/CAD: carry last observed value forward (and backward-fill initial gaps if any)
df_1hz[["hr", "cad"]] = df_1hz[["hr", "cad"]].ffill().bfill()

# Reset index
df_1hz = df_1hz.reset_index()

# ----------------------------
# Build lag-adjusted time axis starting at 0
# ----------------------------
# elapsed seconds from GPX start
elapsed_sec = (df_1hz["time"] - df_1hz["time"].iloc[0]).dt.total_seconds().astype(float)

# apply watch lag (seconds)
time_seconds = elapsed_sec + watch_lag

# shift so lag-adjusted time starts at 0
time_seconds = time_seconds - time_seconds.min()

df_1hz["time_seconds"] = time_seconds

# Keep only what you asked for
out_df = df_1hz[["time_seconds", "latitude", "longitude", "elevation", "hr", "cad"]].copy()

# ----------------------------
# Save CSV
# ----------------------------
csv_output = os.path.join(output_directory, "gpx_watch_1Hz.csv")
out_df.to_csv(csv_output, index=False)
print(f"Saved: {csv_output}")

# ----------------------------
# Create a simple course map (polyline)
# ----------------------------
m = folium.Map(
    location=[out_df.iloc[0]["latitude"], out_df.iloc[0]["longitude"]],
    tiles="CartoDB.VoyagerLabelsUnder",
    zoom_start=15,
)

coords = out_df[["latitude", "longitude"]].values.tolist()
folium.PolyLine(coords, weight=6).add_to(m)

# start/end markers (useful sanity check)
folium.Marker(coords[0], tooltip="Start").add_to(m)
folium.Marker(coords[-1], tooltip="End").add_to(m)

map_output = os.path.join(output_directory, "course_map.html")
m.save(map_output)
print(f"Saved: {map_output}")
