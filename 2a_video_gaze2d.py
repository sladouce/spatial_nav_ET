#!/usr/bin/env python3
"""
overlay_gaze_and_gps_minimap.py

ET_navigation version:
- Reads START/END/watch_lag/reldir from metadata.csv
- Loads gaze from gazedata.txt (JSON-lines) -> overlays gaze dot + trail on scenevideo.mp4
- Loads watch GPS from output/gpx_watch_1Hz.csv -> draws minimap (top-right) with moving location dot
- Outputs: output/scenevideo_with_gaze_and_minimap.mp4

Requirements:
    pip install pandas opencv-python numpy
"""

import os, sys, json
import numpy as np
import pandas as pd
import cv2

# ----------------------------
# Paths + metadata
# ----------------------------
META_CSV = r"C:/ET_navigation/data/metadata.csv"
DATA_DIR = r"C:/ET_navigation/data"

meta_df = pd.read_csv(META_CSV, nrows=1)

reldirectory = str(meta_df.at[0, "reldir"])
START_TIME   = float(meta_df.at[0, "START"])
END_TIME     = float(meta_df.at[0, "END"])
WATCH_LAG    = float(meta_df.at[0, "watch_lag"])  # used earlier when creating gpx_watch_1Hz.csv

input_dir  = os.path.join(DATA_DIR, reldirectory)
output_dir = os.path.join(input_dir, "output")
os.makedirs(output_dir, exist_ok=True)

VIDEO_IN  = os.path.join(input_dir, "scenevideo.mp4")
VIDEO_OUT = os.path.join(output_dir, "scenevideo_with_gaze_and_minimap.mp4")

GAZE_TXT  = os.path.join(input_dir, "gazedata.txt")
GPX_CSV   = os.path.join(output_dir, "gpx_watch_1Hz.csv")  # produced by your GPX script

# ----------------------------
# User-tunable parameters
# ----------------------------
TRAIL_SEC = 1.0

# If gaze timestamps are shifted relative to video time, fix here (seconds)
GAZE_TIME_OFFSET_SEC = 0.0

# Drawing params
GAZE_DOT_COLOR   = (0, 0, 255)      # red (BGR)
GAZE_TRAIL_COLOR = (255, 255, 255)  # white
GAZE_RADIUS      = 6
TRAIL_THICKNESS  = 2

# Minimap params (top-right)
ENABLE_MINIMAP     = True
MINIMAP_W          = 240
MINIMAP_H          = 240
MINIMAP_PADDING    = 12  # from frame border
MINIMAP_BG_COLOR   = (0, 0, 0)      # black
TRACK_COLOR        = (200, 200, 200) # light gray
TRACK_THICKNESS    = 2
POS_COLOR          = (0, 0, 255)     # red dot
POS_RADIUS         = 6

# Optional: show HR/CAD text near minimap
SHOW_HR_CAD = True
TEXT_COLOR  = (255, 255, 255)
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.6
FONT_THICK  = 1

# ----------------------------
# Helpers
# ----------------------------
def norm_to_px(xn, yn, w, h):
    if not (np.isfinite(xn) and np.isfinite(yn)):
        return None
    xn = 0.0 if xn < 0.0 else (1.0 if xn > 1.0 else xn)
    yn = 0.0 if yn < 0.0 else (1.0 if yn > 1.0 else yn)
    return int(round(xn * (w - 1))), int(round(yn * (h - 1)))

def find_nearest_idx(arr, t):
    if arr.size == 0:
        return None
    return int(np.abs(arr - t).argmin())

def load_gaze_jsonlines_nested(txt_path):
    """
    Expected JSON-lines:
      {"timestamp": 12.345, "data": {"gaze2d": [x, y]}}
    Returns df with columns: time_s, X, Y
    """
    times, xs, ys = [], [], []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            t = obj.get("timestamp", None)
            data = obj.get("data", None)
            g = None
            if isinstance(data, dict):
                g = data.get("gaze2d", None)

            if t is None:
                continue

            if isinstance(g, (list, tuple)) and len(g) >= 2:
                x, y = g[0], g[1]
            else:
                x, y = np.nan, np.nan

            times.append(t)
            xs.append(x)
            ys.append(y)

    df = pd.DataFrame({"time_s": times, "X": xs, "Y": ys})
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["X"]      = pd.to_numeric(df["X"], errors="coerce")
    df["Y"]      = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid gaze samples found in gazedata.txt")
    return df

def build_minimap_base(track_xy_px, w, h):
    """
    Creates a base minimap image with the track polyline already drawn.
    track_xy_px: Nx2 pixel coords in minimap space
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = MINIMAP_BG_COLOR
    if len(track_xy_px) >= 2:
        pts = np.array(track_xy_px, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=False, color=TRACK_COLOR,
                      thickness=TRACK_THICKNESS, lineType=cv2.LINE_AA)
    # optional border
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255), 1)
    return img

def lonlat_to_minimap_px(lon, lat, lon_min, lon_max, lat_min, lat_max, w, h, margin=10):
    """
    Map lon/lat into minimap pixel space using bounding box normalization.
    margin: inner padding inside minimap
    """
    if not (np.isfinite(lon) and np.isfinite(lat)):
        return None

    # avoid division by zero
    if abs(lon_max - lon_min) < 1e-12 or abs(lat_max - lat_min) < 1e-12:
        return None

    xn = (lon - lon_min) / (lon_max - lon_min)
    yn = (lat - lat_min) / (lat_max - lat_min)

    # clamp
    xn = 0.0 if xn < 0.0 else (1.0 if xn > 1.0 else xn)
    yn = 0.0 if yn < 0.0 else (1.0 if yn > 1.0 else yn)

    # pixel coords with padding; y inverted for image coords
    x = int(round(margin + xn * (w - 1 - 2 * margin)))
    y = int(round(margin + (1.0 - yn) * (h - 1 - 2 * margin)))
    return (x, y)

def overlay_minimap(frame, minimap_img, pos="top-right"):
    fh, fw = frame.shape[:2]
    mh, mw = minimap_img.shape[:2]

    if pos == "top-right":
        x0 = fw - mw - MINIMAP_PADDING
        y0 = MINIMAP_PADDING
    elif pos == "top-left":
        x0 = MINIMAP_PADDING
        y0 = MINIMAP_PADDING
    elif pos == "bottom-left":
        x0 = MINIMAP_PADDING
        y0 = fh - mh - MINIMAP_PADDING
    else:  # bottom-right
        x0 = fw - mw - MINIMAP_PADDING
        y0 = fh - mh - MINIMAP_PADDING

    # safety
    x0 = max(0, min(x0, fw - mw))
    y0 = max(0, min(y0, fh - mh))

    roi = frame[y0:y0+mh, x0:x0+mw]
    roi[:] = minimap_img
    return x0, y0, mw, mh

# ----------------------------
# Load gaze + GPX
# ----------------------------
if not os.path.isfile(VIDEO_IN):
    print(f"Error: video not found: {VIDEO_IN}")
    sys.exit(1)
if not os.path.isfile(GAZE_TXT):
    print(f"Error: gaze file not found: {GAZE_TXT}")
    sys.exit(1)
if ENABLE_MINIMAP and not os.path.isfile(GPX_CSV):
    print(f"Error: GPX CSV not found: {GPX_CSV}")
    sys.exit(1)

gaze = load_gaze_jsonlines_nested(GAZE_TXT)
gaze["time_s"] = gaze["time_s"] + float(GAZE_TIME_OFFSET_SEC)
times_gaze = gaze["time_s"].to_numpy(dtype=float)

gpx = None
if ENABLE_MINIMAP:
    gpx = pd.read_csv(GPX_CSV)

    required = {"time_seconds", "latitude", "longitude"}
    missing = required - set(gpx.columns)
    if missing:
        raise ValueError(f"Missing columns in {GPX_CSV}: {missing}")

    gpx = gpx.sort_values("time_seconds").reset_index(drop=True)

    # for interpolation
    t_gpx   = gpx["time_seconds"].to_numpy(dtype=float)
    lon_gpx = gpx["longitude"].to_numpy(dtype=float)
    lat_gpx = gpx["latitude"].to_numpy(dtype=float)

    # bounding box for minimap scaling (use whole track)
    lon_min, lon_max = np.nanmin(lon_gpx), np.nanmax(lon_gpx)
    lat_min, lat_max = np.nanmin(lat_gpx), np.nanmax(lat_gpx)

    # precompute track polyline in minimap pixel coords
    track_px = []
    for lo, la in zip(lon_gpx, lat_gpx):
        p = lonlat_to_minimap_px(lo, la, lon_min, lon_max, lat_min, lat_max, MINIMAP_W, MINIMAP_H, margin=10)
        if p is not None:
            track_px.append(p)

    minimap_base = build_minimap_base(track_px, MINIMAP_W, MINIMAP_H)

# ----------------------------
# Open video + writer
# ----------------------------
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    print(f"Error: cannot open video: {VIDEO_IN}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
if not np.isfinite(fps) or fps <= 1e-6:
    fps = 25.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Grab dimensions
ok, frame0 = cap.read()
if not ok:
    print("Error: cannot read first frame.")
    cap.release()
    sys.exit(1)
H, W = frame0.shape[:2]

# Seek back (we consumed one frame)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Clamp window
video_duration = total_frames / fps if total_frames > 0 else None
start_s = max(0.0, float(START_TIME))
end_s   = float(END_TIME)
if video_duration is not None:
    end_s = min(end_s, video_duration)

if end_s <= start_s:
    raise ValueError(f"END ({end_s}) must be > START ({start_s}).")

# Seek to start
cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

i_start = int(round(start_s * fps))
i_end   = int(round(end_s   * fps))
i_end   = min(i_end, total_frames)
n_frames = max(0, i_end - i_start)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (W, H))
if not out.isOpened():
    cap.release()
    raise RuntimeError(f"Cannot open output writer: {VIDEO_OUT}")

print(f"[INFO] FPS={fps:.3f}, frames={total_frames}, window={start_s:.2f}–{end_s:.2f}s, n_frames={n_frames}")

# ----------------------------
# Main loop
# ----------------------------
for k in range(n_frames):
    ok, frame = cap.read()
    if not ok:
        break

    t_frame = start_s + (k / fps)

    # ---- gaze trail ----
    if TRAIL_SEC > 0:
        mask = (times_gaze >= (t_frame - TRAIL_SEC)) & (times_gaze < t_frame)
        if mask.any():
            trail_xy = gaze.loc[mask, ["X", "Y"]].dropna(how="any").values
            pts = []
            for xn, yn in trail_xy:
                p = norm_to_px(xn, yn, W, H)
                if p is not None:
                    pts.append(p)
            if len(pts) >= 2:
                cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False,
                              GAZE_TRAIL_COLOR, TRAIL_THICKNESS, lineType=cv2.LINE_AA)

    # ---- current gaze dot ----
    gi = find_nearest_idx(times_gaze, t_frame)
    if gi is not None:
        xg, yg = gaze.at[gi, "X"], gaze.at[gi, "Y"]
        p = norm_to_px(xg, yg, W, H)
        if p is not None:
            cv2.circle(frame, p, GAZE_RADIUS, GAZE_DOT_COLOR, -1, lineType=cv2.LINE_AA)

    # ---- minimap: current GPS position ----
    if ENABLE_MINIMAP and gpx is not None:
        minimap_img = minimap_base.copy()

        # interpolate lon/lat at t_frame using gpx time_seconds
        # clamp to range
        if t_frame <= t_gpx[0]:
            lon_t, lat_t = lon_gpx[0], lat_gpx[0]
            row_idx = 0
        elif t_frame >= t_gpx[-1]:
            lon_t, lat_t = lon_gpx[-1], lat_gpx[-1]
            row_idx = len(t_gpx) - 1
        else:
            j = np.searchsorted(t_gpx, t_frame)
            t0, t1 = t_gpx[j - 1], t_gpx[j]
            a = 0.0 if t1 == t0 else (t_frame - t0) / (t1 - t0)
            lon_t = lon_gpx[j - 1] + a * (lon_gpx[j] - lon_gpx[j - 1])
            lat_t = lat_gpx[j - 1] + a * (lat_gpx[j] - lat_gpx[j - 1])
            row_idx = j

        pmini = lonlat_to_minimap_px(lon_t, lat_t, lon_min, lon_max, lat_min, lat_max, MINIMAP_W, MINIMAP_H, margin=10)
        if pmini is not None:
            cv2.circle(minimap_img, pmini, POS_RADIUS, POS_COLOR, -1, lineType=cv2.LINE_AA)

        # paste minimap onto frame (top-right)
        x0, y0, mw, mh = overlay_minimap(frame, minimap_img, pos="top-right")

        # optional HR/CAD readout (nearest row)
        if SHOW_HR_CAD:
            hr = None
            cad = None
            if "hr" in gpx.columns:
                hr = gpx.at[row_idx, "hr"]
            if "cad" in gpx.columns:
                cad = gpx.at[row_idx, "cad"]

            lines = []
            if hr is not None and np.isfinite(hr):
                lines.append(f"HR: {int(round(hr))}")
            if cad is not None and np.isfinite(cad):
                lines.append(f"CAD: {int(round(cad))}")

            # draw just under minimap
            if lines:
                y_text = min(H - 10, y0 + mh + 20)
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (x0, y_text + i * 18),
                                FONT, FONT_SCALE, TEXT_COLOR, FONT_THICK, cv2.LINE_AA)

    out.write(frame)

    # stop if capture time drifted beyond end (VFR safety)
    cur_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if cur_time >= end_s:
        break

cap.release()
out.release()

print(f"[DONE] Wrote: {VIDEO_OUT}")
print(f"[INFO] Used metadata START={START_TIME}, END={END_TIME}, watch_lag={WATCH_LAG} (already applied inside gpx_watch_1Hz.csv)")
