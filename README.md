# NaviGaze

A platform to **visualize and analyse eye-tracking + GPS location data** collected
with Tobii Pro Glasses 3 during a spatial navigation task (walking to ordered
checkpoints). It ingests a recording, aligns the GPS track to the eye-tracking
clock, detects checkpoint arrivals, segments the recording between checkpoints,
computes gaze and locomotion features (whole-course **and** per-segment), exports
them for group analysis, and builds an interactive video+map viewer.

## Data layout

```
checkpoints.json                 # GeoJSON: 8 ordered checkpoint circles
config/participants.yaml         # per-participant config (alignment, params)
Datasets/<pid>/                  # RAW inputs only (never written to)
  gazedata.gz  imudata.gz  eventdata.gz   # Tobii G3 raw streams
  scenevideo.mp4  recording.g3  snap0.jpg
  <pid>.gpx                      # GPS track (real or dummy)
derived/<pid>/                   # ← all generated outputs land here
output/group/                    # ← stacked group-level feature tables
app/                             # processing platform (Flask server + SPA)
```

## Setup (Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The repo's default `python` is 3.14 and lacks scientific wheels — always use the
`.venv` interpreter (`.venv\Scripts\python.exe`).

## Processing platform (web app)

Launch the interactive platform and open it in a browser:

```powershell
.\.venv\Scripts\python.exe app\server.py
# -> http://127.0.0.1:5000
```

Pick a dataset from the selector (datasets are auto-discovered from `Datasets/`).
Tabs:

0. **Process** — shows the selected dataset's status (raw data / GPX / CV model /
   processed) and a **Run processing** button that runs the full pipeline on
   demand. Options: **detect maps in scene video (CV)** — runs the trained YOLO
   detector as a step before feature extraction (with a sample-rate selector and
   a test-cap for trying a few frames) — and rendering the standalone folium maps.
   Results stream into a log.
0b. **Batch** — process **multiple datasets sequentially**: tick the datasets
   (auto-discovered from `Datasets/`, with select-all / only-unprocessed helpers),
   choose the shared CV/options, and run. Each dataset's status and timing stream
   into a log; datasets without a GPX are skipped with a note.
1. **Viewer** — scene video with the live gaze overlay beside a map showing the
   current position (pulsing marker) + path-so-far + checkpoints. The viewer has
   its **own play / pause / scrubber / speed** controls driving an independent
   playhead, so the map position and gaze advance from the GPX/gaze timeline even
   if the large video is slow to decode; the video is synced to the playhead
   best-effort. It is streamed with HTTP range support so it also seeks correctly.
2. **Map explorer** — one Leaflet map of the 5 m bins; pick a measure from the
   grouped side menu (Navigation / Gaze / Environment) and the bins recolour
   instantly with a legend.
3. **Segment bars** — horizontal bars, one per between-checkpoint segment,
   stacked top-to-bottom; pick a measure and the bars update to its mean value
   per segment.
4–5. **Group map** / **Group segments** — the same two views but averaged across
   **all processed participants** (`/api/group/...`): the bin map shows the
   per-bin mean (bins are in a shared frame) and the bars show the per-segment
   mean, with the participant count in the legend/title.

All generated outputs live in **`derived/<pid>/`** (outside `Datasets/`, which
holds only raw inputs), so adding more participants is just dropping their
folder under `Datasets/` and processing them.

## Quick start (participant AC_06-03)

```powershell
# 1. (Optional) generate a dummy GPX for pretesting — real GPX goes in the same place
.\.venv\Scripts\python.exe tools\make_dummy_gpx.py AC_06-03

# 2. Run the full pipeline
.\.venv\Scripts\python.exe scripts\run_participant.py AC_06-03

# 3. Stack per-participant features into group tables
.\.venv\Scripts\python.exe scripts\collect_group.py
```

Then open `Datasets/AC_06-03/derived/viewer.html` in a browser.

`run_participant.py` also accepts `--all` (every participant in
`participants.yaml`) and `--no-maps` (skip folium maps for speed).

## Manual GPS ↔ eye-tracking alignment

The GPS and glasses were **not** hardware-synced. GPX points carry absolute
datetimes; gaze/video timestamps are seconds from the recording start (and share
one clock, so gaze time == video time). We map GPX → eye-tracking seconds with a
single offset — the ET second at which the GPX track begins:

```
et_s = gpx_align_et_seconds + (gpx_point_time − gpx_first_point_time)
```

Set `gpx_align_et_seconds` per participant in `config/participants.yaml`. If left
`null`, the pipeline auto-suggests it from `recording.g3`'s `created` time vs the
GPX first timestamp (only valid if both device clocks agreed). **Refine it by eye:**
open `viewer.html` and nudge the value until the map marker enters checkpoint 1
in sync with the video.

## Outputs (`Datasets/<pid>/derived/`)

| File | Contents |
|------|----------|
| `gpx_aligned.csv` | GPX track with `et_s` (eye-tracking seconds) |
| `track_grid.csv` | track resampled to a uniform 1 Hz ET grid |
| `arrivals.csv` | per-checkpoint first-entry time (`arrival_et_s`) |
| `segments.csv` | between-checkpoint segments in ET seconds |
| `fixations.csv` | I-DT fixations (onset, duration, centroid, dispersion) |
| `features_segment.csv` | one row per segment — all features |
| `features_course.csv` | one row — whole-course features |
| `features_binned.csv` | one row per 5 m spatial bin — instantaneous metrics + isovist |
| `osm_buildings.geojson` | cached OSM building footprints (for isovist) |
| `viewer.html` | interactive scene-video + gaze overlay + live map |
| `binmap_<feature>.html` | course in 5 m bins, coloured by an instantaneous metric |
| `covmap_<X>__x__<Y>.html` | per-bin covariance contribution of X × Y |

### Features

**Gaze:** `n_fixations`, `fixation_rate_hz`, `mean_fixation_dur_s`,
`gaze_dispersion_rms_deg`, `bcea_deg2` (bivariate contour ellipse area),
`spatial_entropy_norm` (grid-based Shannon entropy, 0–1), `direction_changes` /
`direction_change_rate_hz`, `n_saccades`.

**Locomotion:** `mean_speed_ms`, `max_speed_ms`, `time_to_checkpoint_s`,
`path_length_m`, `straightline_m` (ideal = straight legs between checkpoints),
`path_efficiency`, `path_deviation_mean_m`, `path_deviation_max_m`.

All are computed identically for the whole course and each segment, so they are
directly comparable and stack cleanly across participants.

#### Measure glossary & grouping

The Map-explorer and Segment-bars tabs share **one grouped, normalized measure
registry** (so the two are consistent):

- **Navigation** — `time_to_checkpoint_s`, `walking_speed_kmh` (mean speed while
  *moving*, i.e. excluding pauses, in km/h), `n_pauses` (contiguous stretches
  below `walk_threshold_ms`=0.3 m/s lasting ≥ `min_pause_s`=2 s), `prop_pausing`,
  `path_deviation_mean_m` / `path_deviation_max_m`, `path_efficiency`.
- **Gaze** — `fixation_rate_hz` (per second, not raw count), `mean_fixation_dur_s`,
  `gaze_dispersion_rms_deg`, `bcea_deg2` (**B**ivariate **C**ontour **E**llipse
  **A**rea — the area of the ellipse enclosing 68 % of gaze points; bigger = more
  scattered), `spatial_entropy_norm`, `direction_change_rate_hz`,
  `map_fixation_proportion` (CV).
- **Environment** (map bins) — `isovist_openness`, `isovist_area_m2`, and
  `time_in_bin_s` (**occupancy** time the participant spent in that 5 m cell — this
  is *not* fixation duration; that's `mean_fixation_dur_s`).

Counts are presented as rates so segments of different length/duration compare
fairly. Spatial bins are anchored to the **checkpoints centroid** (a fixed frame
shared by all participants), so a given `bin_id` is the same place for everyone
and bins can be averaged across people.

### Spatial bins, isovist & covariance maps

The course is partitioned into fixed **5 m square cells** (`bin_size_m`). Each
occupied bin (`features_binned.csv`) carries the *instantaneous* metrics of what
happened while the participant was there — `mean_speed_ms`, `dwell_time_s`, the
gaze features above — plus **isovist openness**:

- `isovist_openness` (mean sightline length / max radius, 0–1), `isovist_mean_radius_m`,
  `isovist_min_radius_m`, `isovist_area_m2`. Computed by casting `isovist_n_rays`
  rays (default 72) from each bin centre and measuring the free distance to the
  nearest OpenStreetMap **building footprint** (fetched once via Overpass for the
  course bbox, cached to `osm_buildings.geojson`). If the network/Overpass is
  unavailable, openness is skipped and its covariance maps are omitted.

`binmap_<feature>.html` colours every 5 m bin by one metric. `covmap_<X>__x__<Y>.html`
colours each bin by its **signed contribution to the covariance** between two
variables — red = they co-vary together there, blue = they move oppositely — and
the legend reports the global covariance and Pearson *r* across bins. Pairs are
generated for `mean_speed_ms` and `isovist_openness` against each gaze metric
(e.g. dispersion, entropy, direction-change rate, fixation rate, and — once the
CV map detector exists — `map_fixation_proportion`).

### Viewer notes

`viewer.html` **bundles Leaflet inline** (no CDN), so the gaze overlay and map
work even on locked-down networks; only the map *tiles* need internet. The gaze
overlay runs independently of the map, so a tile/map hiccup can't blank it. The
dummy GPX spans ~90% of the recording, so scrubbing shows the position moving
across the course.

## Architecture

```
navet/
  config.py          paths + ParticipantConfig (reads participants.yaml)
  geo.py             haversine, local ENU projection, point→segment distance
  io_tobii.py        parse gazedata/imudata/eventdata.gz + recording.g3 meta
  io_gpx.py          parse GPX → track (distance, speed)
  io_checkpoints.py  parse checkpoints.json → ordered (center, radius_m)
  align.py           manual GPX→ET alignment + 1 Hz resampling
  detect.py          checkpoint-entry detection + segmentation
  fixations.py       I-DT fixation detection (numpy, no external toolbox)
  binning.py         5 m spatial bins + per-bin instantaneous metrics
  isovist.py         OSM buildings (Overpass) + ray-cast isovist openness
  features/          gaze.py, locomotion.py, aggregate.py (build + export)
  viewer/            build_viewer.py (interactive HTML), build_feature_map.py
                     (binned feature maps + covariance maps), assets/leaflet.*
  pipeline.py        orchestrates one participant end-to-end
app/
  server.py          Flask: dataset discovery, JSON APIs, range video, process-on-demand
  static/            SPA (index.html, app.js, app.css) + vendored leaflet/plotly
tools/make_dummy_gpx.py
scripts/run_participant.py, scripts/collect_group.py
```

## Notes & deferred work

- `viewer.html` bundles Leaflet locally; only OpenStreetMap *tiles* need internet.
  Gaze/path/checkpoint data are embedded, so the gaze overlay works fully offline.
- Isovist openness needs internet **once** to fetch OSM buildings (then cached).
### Map detection (computer vision)

A trained YOLO model (`training_computervision/runs/detect/train/weights/best.pt`,
single class `map`) detects the held map in the scene video. Inference runs in
the **training virtualenv** (which has ultralytics/torch/cv2) via subprocess
(`navet/cv/detect_maps.py`), so the main venv stays torch-free. Boxes are written
**per frame** to `derived/<pid>/map_detections.csv` (`frame, time_s, x1, y1, x2,
y2, conf, cls`, normalised 0–1; frames with no detection still get a row), either
sampled at a chosen fps or for **every frame** (`--all-frames` / the Process-tab
checkbox). `navet/mapdetect.py` caches results (parameter-aware), flags each
fixation `on_map` when its centroid falls in a map box near its time, and the
gaze features then fill `map_fixation_count` (number of map fixations) and
`map_fixation_proportion` (fraction of window time fixating the map) — for the
whole course, per segment, and per 5 m bin. The **Viewer** tab has a *map
detections* toggle that overlays the boxes on the scene video.

> **Inference env matters.** The detector itself is good (the fine-tuned model
> reaches mAP50 ≈ 0.995). A subtle gotcha cost real time: with
> `ultralytics 8.2.81 + torch 2.4.0` the CV venv silently corrupted inference —
> *every* model (including stock COCO `yolov8s.pt`) returned 300 boxes/frame at
> confidence 1.0 with nonsense classes. **Upgrading the CV venv fixed it**
> (`training_computervision/venv` → ultralytics 8.4.57 + torch 2.12.0). Rule of
> thumb: if even a stock model floods conf-1.0 boxes, suspect the torch/ultralytics
> versions, not the weights. `trainYOLO.py` fine-tunes from `yolov8s.pt` with a
> real `images/val` split (created by `make_val_split.py`); trained weights land
> at `runs/detect/finetune/weights/best.pt`, which the platform prefers
> automatically.

- **Map-fixation features** (`map_fixation_count`, `map_fixation_proportion`) are
  present as columns but left `NaN`. They will be filled once the computer-vision
  map detector is trained: per-frame map bounding boxes (à la the reference
  `code_walkingstudy/5_gazesegmentation_CV.py`) intersected with fixation points.
- A custom ideal trajectory can later replace the straight-line default by
  dropping an `ideal_trajectory.json` alongside the data and wiring it into
  `features/locomotion.py`.
- IMU-derived head/gait features are available to add (`io_tobii.parse_imu`).
