# Data Setup

## DeepSense 6G — Scenario 33 (verified locally)

Layout expected under `data/` (git-ignored):

```
data/scenario33/
  scenario33_dev.csv          # index CSV, 3981 rows
  unit1/
    camera_data/  image_<i>.jpg
    lidar_data/   lidar_data_<i>.ply
    mmWave_data/  mmWave_power_<i>.txt   # 64 lines, one float per line (linear power)
    radar_data/   radar_data_<i>.npy
    GPS_data/     gps_location.txt      # static base-station position
  unit2/
    GPS_data/     GPS_location_<i>.txt  # moving user position per sample
```

The loader resolves `data/scenarioN/` or `data/N/` (any case / separator), picks the shortest
`.csv` whose name contains `dev` (else the shortest `.csv`), and resolves columns by regex.

### `scenario33_dev.csv` columns (actual header)

| Column | Meaning | Resolved as |
|---|---|---|
| `index` | DeepSense sample index (1-based, not contiguous: 1 … 4685) | sample index |
| `unit1_rgb` | path to camera frame | `image` |
| `unit1_pwr_60ghz` | path to the 64-line beam-power file | `power` |
| `unit1_lidar` | path to `.ply` point cloud | `lidar` |
| `unit1_radar` | path to radar `.npy` | `radar` |
| `unit1_loc` | base-station GPS file (same file for every row) | `gps` |
| `unit2_loc` | user GPS file per sample | `gps` |
| `unit1_beam` | released optimal-beam label, **1-based** | `beam_label` |
| `unit1_max_pwr` | released peak power (NaN-aware max) | `max_power` |
| `time_stamp` | `HH:MM:SS-microseconds` | `time` |
| `seq_index` | drive / pass-by segment id (18 segments, 119–399 samples each) | `sequence` |
| `unit2_spd_over_grnd_kmph`, `unit2_num_sats`, `unit2_altitude`, `unit2_geo_sep`, `unit2_mode_fix_type`, `unit2_pdop`, `unit2_hdop`, `unit2_vdop`, `unit2_interpolated_position` | GPS quality / speed fields | numeric side info |

Median sampling interval: 91.8 ms. Power values are linear, range 0.125–0.794.

### Beam label offset (empirically detected)

`unit1_beam` is 1-based. `load_scenario` compares the released label with `argmax(power)`
under offsets 0 and 1 and keeps the better one, storing it in `DeepSenseScenario.beam_label_offset`
and reporting agreement both overall and on rows without NaN beams. Scenario 33: offset 1,
100 % agreement on clean rows. Do not assume the offset for other scenarios; read it from the
loaded object.

### NaN beams

38 rows contain 1–9 NaN entries in the 64-vector. The released `unit1_beam` at those rows
points at the NaN beam (wrong), while `unit1_max_pwr` was computed NaN-aware. The loader's
`optimal_beam` uses `nanargmax`, and `load_scenario(..., nan_policy=...)` accepts:

| policy | behaviour |
|---|---|
| `interpolate` (default) | linear interpolation across adjacent beams inside the vector; an all-NaN row falls back to the previous finite row |
| `keep` | leave NaNs in place (argmax stays safe, metrics will not be) |
| `raise` | raise `ValueError` if any NaN is present |

Validate any scenario with:

```bash
python -m channeldreamer.scripts.prepare_data --data-root data --scenario 33 --check-files
```

### GPS files

`unit2_loc` points to a per-sample text file with two lines (latitude, longitude in degrees);
`unit1_loc` is the static base-station position (one file for every row).
`channeldreamer.data.modalities.load_gps` reads both and projects the user position to local
east/north metres relative to the base station (Scenario 33: the user stays within about
−9 … 15 m east and −2 … 52 m north). `trajectory_windows` builds the `(N, 16, 4)` context the
trajectory encoder consumes (position and per-step displacement, clamped at segment starts).

### LiDAR files

`unit1_lidar` points to ASCII PLY files (`x y z` as double, `intensity` as ushort, ~18 k points,
range up to ~200 m). `read_ply_points` handles ASCII and binary PLY.

### Offline caches for Phase 3 (`data/scenario33/cache/`)

```bash
python -m channeldreamer.scripts.prepare_data --data-root data --scenario 33 --pretokenize-lidar --precompute-camera
```

| cache | content | produced by |
|---|---|---|
| `cache/lidar/lidar_tokens_<index>.npz` | `tokens (16, 64)`, `centroids (16, 3)`, `groups (16, 32, 4)` per sample; FPS + kNN grouping within 60 m, seeded frozen mini-PointNet | `LidarTokenizer` (CPU, ~60 clouds/s) |
| `cache/camera_features.npz` | `features (N, 512)` pooled ResNet-18 ImageNet features + `sample_index` | `CameraEncoder.encode_paths` (frozen backbone, GPU) |

Training reads these small arrays through `make_sequence_batch`; raw point clouds and JPEGs
are never touched by the world model.

## WiWorld-RealData (Phase 2, not downloaded yet)

Dual-band (3.7 GHz / 6.775 GHz) complex channel impulse responses along a single public route,
with a per-sample quality flag. The loader in `channeldreamer.data.wiworld` is written against
a synthetic CIR generator and takes every manifest column name as a parameter, because the
real manifest has not been inspected. Once the dataset is downloaded, place it under
`data/wiworld/`, run the loader with `--dry-run` style column resolution, and update this
section with the actual header.
