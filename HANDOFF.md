# Handoff

## TL;DR

- This repo has two main branches:
  - `Online Adaptive CBF`: `GAT` and `MLP`
  - `BarrierNet`
- The common rollout / evaluation engine for the adaptive branch is `adaptation_experiment.py`.
- Most final adaptive results live under `epoch_exp/`.
- Most final BarrierNet results live under `barriernet_results/`.
- `best_results_summary_*.txt` and `best_mlp_results_summary_*.txt` should be treated as curated representative result files.
  - The raw auto-generated summaries are usually `results_summary_epoch_*.txt` and `results_summary_epoch_*_mlp_threshold_*.txt`.
- `run_training_tmux.sh` is **not** a universal launcher.
  - It only wraps `run_post_training_pipeline.py` inside `tmux`.
  - It does **not** generate data, train the main model by itself, or run the full MLP / BarrierNet pipelines.
- `run_barriernet_all_models.sh` is a BarrierNet-only all-in-one wrapper.
  - In the current checked-in version, it processes `DynamicUnicycle2D`, `Quad2D`, and `KinematicBicycle2D_DPCBF`.
  - `Quad3D` is currently excluded from the default all-model BarrierNet shell pipeline.
  - Its experiment phase still depends on baseline CSVs already existing from the adaptive branch.

## Latest Results

The latest adaptive results live in `epoch_exp/`, and the latest BarrierNet results live in `barriernet_results/`.

Important interpretation note:

- In this section, "latest results" means the **latest final / representative result files currently being used for handoff**, not necessarily the most recently modified file in the filesystem.
- For the adaptive branch, these are the representative `best_results_summary_*.txt` and `best_mlp_results_summary_*.txt` files inside the latest final folders.
- For BarrierNet, these are the handoff-selected `.summary.txt` files in `barriernet_results/<Dynamics>/`, chosen by the experiment timestamp embedded in the filename rather than filesystem mtime.

For each dynamics below:

- the `GAT / baseline / optimal-decay` rows come from the `best_results_summary_*.txt` file
- the `Online Adaptive ... MLP` row comes from the `best_mlp_results_summary_*.txt` file
- the `BarrierNet` row comes from the selected BarrierNet `.summary.txt` handoff file

### DynamicUnicycle2D

Source files used:

- Folder: `epoch_exp/du2d_1113/`
- GAT / baseline / optimal-decay source:
  - `epoch_exp/du2d_1113/best_results_summary_epoch_300_2_threshold_0_987007.txt`
- MLP source:
  - `epoch_exp/du2d_1113/best_mlp_results_summary_epoch_400_mlp_threshold_0_130267 copy.txt`
- BarrierNet source:
  - `barriernet_results/DynamicUnicycle2D/sim_results_in_obs_sweep_BarrierNet_DynamicUnicycle2D_0126_1108.summary.txt`

| Controller | Collision Rate | Reach Rate | Avg Time (Reached) | Avg Time (All) |
| --- | --- | --- | --- | --- |
| MPC-CBF high fixed param | 39.0% (39/100) | 61.0% (61/100) | 10.270 sec | 7.635 sec |
| MPC-CBF low fixed param | 0.0% (0/100) | 100.0% (100/100) | 31.052 sec | 31.052 sec |
| Online Adaptive MPC-CBF GAT | 0.0% (0/100) | 100.0% (100/100) | 10.832 sec | 10.832 sec |
| Online Adaptive MPC-CBF MLP | 11.0% (11/100) | 89.0% (89/100) | 23.876 sec | 22.343 sec |
| Optimal Decay MPC-CBF | 0.0% (0/100) | 100.0% (100/100) | 29.213 sec | 29.213 sec |
| Optimal Decay CBF-QP | 39.0% (39/100) | 36.0% (36/100) | 12.956 sec | 32.250 sec |
| BarrierNet | 61.0% (61/100) | 39.0% (39/100) | 7.972 sec | 5.005 sec |

### Quad2D

Source files used:

- Folder: `epoch_exp/quad2d_1111/`
- GAT / baseline / optimal-decay source:
  - `epoch_exp/quad2d_1111/best_results_summary_epoch_200_threshold_0_022786.txt`
- MLP source:
  - `epoch_exp/quad2d_1111/best_mlp_results_summary_epoch_400_mlp_threshold_0_897910 copy.txt`
- BarrierNet source:
  - `barriernet_results/Quad2D/sim_results_in_obs_sweep_BarrierNet_Quad2D_0126_1146.summary.txt`

| Controller | Collision Rate | Reach Rate | Avg Time (Reached) | Avg Time (All) |
| --- | --- | --- | --- | --- |
| MPC-CBF high fixed param | 4.0% (4/100) | 85.0% (85/100) | 11.645 sec | 21.009 sec |
| MPC-CBF low fixed param | 1.0% (1/100) | 99.0% (99/100) | 29.186 sec | 29.130 sec |
| Online Adaptive MPC-CBF GAT | 0.0% (0/100) | 100.0% (100/100) | 16.753 sec | 16.753 sec |
| Online Adaptive MPC-CBF MLP | 11.0% (11/100) | 89.0% (89/100) | 16.229 sec | 15.448 sec |
| Optimal Decay MPC-CBF | 4.0% (4/100) | 96.0% (96/100) | 28.590 sec | 28.204 sec |
| Optimal Decay CBF-QP | 9.0% (9/100) | 0.0% (0/100) | 0.000 sec | 91.170 sec |
| BarrierNet | 51.0% (51/100) | 0.0% (0/100) | 0.000 sec | 49.384 sec |

### Quad3D

Source files used:

- Folder: `epoch_exp/quad3d_1207/`
- GAT / baseline / optimal-decay source:
  - `epoch_exp/quad3d_1207/best_results_summary_epoch_600_2_threshold_0_877601.txt`
- MLP source:
  - `epoch_exp/quad3d_1207/best_mlpresults_summary_epoch_800_mlp_threshold_13_608210 copy.txt`
- BarrierNet source:
  - `barriernet_results/Quad3D/sim_results_in_obs_sweep_BarrierNet_Quad3D_0110_0036.summary.txt`

| Controller | Collision Rate | Reach Rate | Avg Time (Reached) | Avg Time (All) |
| --- | --- | --- | --- | --- |
| MPC-CBF high fixed param | 21.0% (21/100) | 79.0% (79/100) | 12.484 sec | 10.627 sec |
| MPC-CBF low fixed param | 0.0% (0/100) | 100.0% (100/100) | 24.603 sec | 24.603 sec |
| Online Adaptive MPC-CBF GAT | 0.0% (0/100) | 100.0% (100/100) | 15.051 sec | 15.051 sec |
| Online Adaptive MPC-CBF MLP | 2.0% (2/100) | 95.0% (95/100) | 16.965 sec | 19.237 sec |
| Optimal Decay MPC-CBF | 0.0% (0/100) | 100.0% (100/100) | 22.727 sec | 22.727 sec |
| BarrierNet | 66.0% (66/100) | 14.0% (14/100) | 15.582 sec | 23.930 sec |

### KinematicBicycle2D_DPCBF

Source files used:

- Folder: `epoch_exp/kinematicbicycle2D_1210/`
- GAT / baseline / optimal-decay source:
  - `epoch_exp/kinematicbicycle2D_1210/best_results_summary_epoch_200.txt`
- MLP source:
  - `epoch_exp/kinematicbicycle2D_1210/best_mlp_results_summary_epoch_200_mlp_threshold_0_024335 copy.txt`
- BarrierNet source:
  - `barriernet_results/KinematicBicycle2D_DPCBF/sim_results_in_obs_sweep_BarrierNet_KinematicBicycle2D_DPCBF_0126_1117.summary.txt`

| Controller | Collision Rate | Reach Rate | Avg Time (Reached) | Avg Time (All) |
| --- | --- | --- | --- | --- |
| CBF-QP high fixed param | 84.0% (84/100) | 16.0% (16/100) | 4.228 sec | 1.340 sec |
| CBF-QP low fixed param | 0.0% (0/100) | 85.0% (85/100) | 19.900 sec | 30.284 sec |
| Online Adaptive CBF-QP GAT | 1.0% (1/100) | 99.0% (99/100) | 9.753 sec | 9.694 sec |
| Online Adaptive CBF-QP MLP | 2.0% (2/100) | 98.0% (98/100) | 8.556 sec | 8.478 sec |
| Optimal Decay CBF-QP | 21.0% (21/100) | 79.0% (79/100) | 5.164 sec | 4.816 sec |
| BarrierNet | 73.0% (73/100) | 27.0% (27/100) | 4.948 sec | 3.639 sec |

## Which Script Runs What

### Adaptive GAT

1. Generate graph data with `gat_data_generation.py`
2. Train GAT-PENN checkpoints with `nn_model/train_data_with_checkpoints.py`
3. Run epoch-wise calibration + epoch-wise experiment with `run_post_training_pipeline.py`
4. Archive those epoch outputs into `epoch_exp/<folder>/`
5. Run final best-window comparison with `run_best_window_analysis_and_experiments.py`

### Adaptive MLP

1. Generate graph data with `gat_data_generation.py`
2. Convert graph dataset to flat CSV with `convert_pkl_csv.py`
3. Copy the `.pkl` and `.csv` into `nn_model/data/`
4. Train MLP-PENN checkpoints with `nn_model/train_data_with_checkpoints.py`
5. Run final evaluation with `run_mlp_experiments_all_models.py`

### BarrierNet

1. Generate BarrierNet dataset with `safe_control/position_control/BarrierNet/generate_dataset.py`
2. Train BarrierNet with `safe_control/position_control/BarrierNet/train.py`
3. Evaluate best-window on baseline-matched `obs_id`s with `run_barriernet_experiments_all_models.py`
4. Re-run `summarize_barriernet_results.py` only if you want standalone or bulk summary regeneration
5. Re-run `analyze_barriernet_p_values.py` only if you want standalone p/alpha diagnostics

## About the Shell Scripts

- `run_training_tmux.sh`
  - Not a universal pipeline.
  - It only launches `python run_post_training_pipeline.py ...` inside `tmux`.
  - Useful only as a convenience wrapper for GAT post-processing.
- `run_barriernet_all_models.sh`
  - BarrierNet all-in-one wrapper for data generation -> training -> experiment -> summary.
  - Useful for BarrierNet only.
  - It still expects baseline CSVs to already exist for the fairness-matched experiment phase.
- Therefore, **two shell scripts are not enough to cover all GAT/MLP/BarrierNet workflows**.
  - GAT and MLP are mainly driven by Python entry scripts.
  - BarrierNet is the part that has a practical shell wrapper.

## Script-by-Script Configuration Guide

This is the most important practical section.

Legend:

- `Run as-is`
  - usually safe to execute directly
- `CLI/env-driven`
  - no source edit needed; control behavior by arguments or environment variables
- `Edit source first`
  - you must change variables inside the script before running

### Adaptive / Root Scripts

| Script | Main purpose | Can run as-is? | How it is controlled | What you must change to do something specific |
| --- | --- | --- | --- | --- |
| `gat_data_generation.py` | Generate graph dataset for GAT/MLP | `Edit source first` | bottom-of-file constants | Change `controller_name`, `robot_model`, `num_samples`, `num_processes`, `output_prefix`; only touch `ROBOT_SPECS` if you are changing the actual experiment design |
| `convert_pkl_csv.py` | Convert graph `.pkl` to flat `.csv` for MLP | `Run as-is` or `CLI` | optional positional arg | If no arg: converts the four standard datasets; if one arg: `python convert_pkl_csv.py <dataname>` |
| `nn_model/train_data_with_checkpoints.py` | Train MLP-PENN or GAT-PENN | `Edit source first` | top-of-file constants | Change `DATANAME`, `MODELNAME_SAVE`, `SCALERNAME_SAVE`, `robot_model`, `USE_GAT_EMBED`; run from inside `nn_model/` |
| `run_post_training_pipeline.py` | GAT-only epoch calibration + epoch-wise raw experiments | `CLI-driven, but partly hard-coded` | command-line args plus hard-coded dataset / experiment settings | Pass `--robot`, `--controller`; optionally `--pattern`, `--max-checkpoints`; if you trained on non-canonical dataset filenames, update the hard-coded `data_pkl` mapping inside `run_calibration()`; if you want different raw rollout settings, also check the fixed `MAX_T` / `OBS_SET_COUNT` values inside `run_experiment()` |
| `run_best_window_analysis_and_experiments.py` | Final GAT comparison on archived `epoch_exp/<folder>` | `CLI-driven` | folder argument | Run `python run_best_window_analysis_and_experiments.py <folder>`; no source edit normally needed if folder already contains epoch CSVs + config/calibration files |
| `run_mlp_experiments_all_models.py` | Final MLP comparison and summary generation | `Edit source first` | hard-coded maps near top | Check/update `CONTROLLER_MAP`, `DYNAMICS_FOLDER_MAP`, `HIGH_FIXED_PARAM_CONTROLLER`, `CSV_DATA_MAP`, `GAMMA_DIM_MAP`; current checked-in file has several dynamics commented out |
| `adaptation_experiment.py` | Core rollout engine for all adaptive methods and BarrierNet evaluation | `CLI/env-driven` | environment variables | Usually do not edit source; set `SELECTED_ROBOT`, `SELECTED_CONTROLLER`, `DISTR_MODE`, `SELECTED_MODE`, `MAX_T`, `OBS_SET_COUNT`, `OBS_IDS_FILTER`, `RAW_EPISTEMIC_THRESHOLD`, `CHECKPOINT_FILE`, `SCALER_FILE`, `OUTPUT_DIR`, `EPOCH_NUMBER` |
| `online_adaptive_cbf.py` | Adaptive model / adapter library used by `adaptation_experiment.py` | `Usually not run directly` | functions plus defaults in `online_cbf_config.py` | Treat this mainly as library code; edit `online_cbf_config.py` if changing default adaptive model paths or thresholds |
| `online_cbf_config.py` | Central adaptive model/config registry | `Do not run directly` | source file | Update model paths, thresholds, and defaults when switching the active model or checkpoint |
| `run_training_tmux.sh` | Launch `run_post_training_pipeline.py` in tmux | `CLI-driven` | shell args | Pass a session name and then the same args you would give `run_post_training_pipeline.py`; this is only a tmux wrapper, not a general pipeline runner |

### BarrierNet Scripts

| Script | Main purpose | Can run as-is? | How it is controlled | What you must change to do something specific |
| --- | --- | --- | --- | --- |
| `safe_control/position_control/BarrierNet/generate_dataset.py` | Generate BarrierNet `.mat` dataset | `CLI/env-driven` | environment variables | Set `BN_ROBOT_MODEL`, `BN_TARGET_ROWS`, `BN_MAX_SIMS`, `BN_N_PROCESSES`, `BN_N_SIMS`, `BN_DT`, `BN_T_MAX`; no source edit normally needed |
| `safe_control/position_control/BarrierNet/train.py` | Train BarrierNet model | `CLI/env-driven` | environment variables | Set `BN_ROBOT_MODEL`, `BN_EPOCHS`, `BN_BATCH_SIZE`, `BN_LR`, `BN_DEVICE`, `BN_PATIENCE`, `BN_P_REG_WEIGHT`; no source edit normally needed |
| `run_barriernet_experiments_all_models.py` | Best-window BarrierNet evaluation plus per-run summary / p-analysis | `CLI/env-driven` | environment variables | Optionally set `BN_ROBOTS`, `BN_WINDOW_SIZE`, `BN_LIMIT_OBS_IDS`, `BN_DRY_RUN`; note that the current checked-in default excludes `Quad3D` unless you modify the script |
| `summarize_barriernet_results.py` | Standalone or bulk `.summary.txt` regeneration for BarrierNet CSVs | `CLI/env-driven` | environment variables | Optionally set `BN_RESULTS_DIR` and `BN_SUMMARY_OUT`; useful when you want to regenerate summaries without rerunning experiments |
| `analyze_barriernet_p_values.py` | Standalone optional diagnostic for learned `p1/p2` or `alpha` | `CLI-driven` | command-line args | Pass `--robot_model`, `--checkpoint`, optional `--data`; useful when you want p/alpha analysis without rerunning experiments |
| `run_barriernet_all_models.sh` | All-in-one BarrierNet wrapper | `Run as-is` or `env-driven` | env overrides before command | Optionally override `BN_TARGET_ROWS`, `BN_MAX_SIMS`, `BN_EPOCHS`, `BN_BATCH_SIZE`, `BN_LR`, `BN_DEVICE`, `BN_N_PROCESSES`, `BN_WINDOW_SIZE`, `BN_LIMIT_OBS_IDS`; it also requires the `cbf` conda env and existing baseline CSVs for the experiment phase; edit source only if changing the hard-coded `MODELS` list |
| `run_barriernet_dynamic_unicycle.sh` | DU2D-only BarrierNet wrapper | `Run as-is` or `env-driven` | env overrides before command | Same override pattern as above; no source edit normally needed |
| `run_barriernet_quad2d_full.sh` | Quad2D-only BarrierNet wrapper | `Run as-is` or `env-driven` | env overrides before command | Same override pattern as above; no source edit normally needed |
| `run_barriernet_remaining_models.sh` | Quad2D + Kinematic DPCBF BarrierNet wrapper | `Run as-is` or `env-driven` | env overrides before command | Same override pattern as above; edit source only if changing `MODELS` |

### Plot / Analysis Scripts

| Script | Main purpose | Can run as-is? | How it is controlled | What you must change to do something specific |
| --- | --- | --- | --- | --- |
| `plot_safety_loss_function_grid.py` | Paper-style safety-loss surface plots | `Edit source first` | plotting constants in file | Change `lambda_1_values`, `delta_theta_values`, plot ranges, obstacle setup, and example robot/controller block |
| `plot_uncertainties_grid.py` | Paper-style PENN uncertainty / prediction grid plots | `Edit source first` | hard-coded model/scaler and grid values | Change `model_path`, `scaler_path`, `gamma_pairs`, and feature ranges before running |
| `test_plot.py` | Interactive real-time GMM plot during rollout | `Edit source first` | example function setup | Change checkpoint/scaler paths and scenario setup in `test_plot_example()` |
| `check_data_histogram.py` | Quick dataset histogram inspection | `Edit source first` | `DATANAME` constant | Change `DATANAME` to the dataset you want to inspect |
| `visualize_single_simulation.py` | Visualize one archived failing case | `CLI + partly hard-coded` | `obs_id` arg plus top config | It is currently wired mainly for `KinematicBicycle2D_DPCBF`; to use another robot, source edits are needed |
| `analyze_and_visualize_failing_cases.py` | Find best-window failures and visualize them | `CLI-driven` with top constants | folder args + top constants | Usually pass folders on CLI; edit top constants only if changing distribution assumptions |
| `monitor_dataset_generation.sh` | Watch dataset-generation log | `Edit source if log path differs` | hard-coded `LOG_FILE` | Change `LOG_FILE` if your generator is writing elsewhere |

### Older / Archive Candidates

| Script | Status | Why |
| --- | --- | --- |
| `data_generation.py` | archive candidate | older predecessor to `gat_data_generation.py` |
| `run_full_pipeline.sh` | archive candidate | old DU2D-specific wrapper |
| `run_final_barriernet_pipeline.sh` | archive candidate | older BarrierNet wrapper |
| `run_best_window_experiments.py` | archive candidate | narrow older helper superseded by `run_best_window_analysis_and_experiments.py` |
| `calculate_metrics.py` | archive candidate | hard-coded old file paths and old experiment set |
| `list_valid_obs_ids.py` | archive candidate | one-off helper for a specific failure list |

### Minimal "Do Not Guess" Rules

- If a script contains hard-coded `DATANAME`, `MODELNAME_SAVE`, `SCALERNAME_SAVE`, `robot_model`, or large mapping dictionaries near the top, assume **you must read and update those first**.
- If a script takes `argparse` args or uses `BN_*` / `SELECTED_*` env vars, prefer **args/env overrides instead of source edits**.
- For `nn_model/train_data_with_checkpoints.py`, always check:
  - `DATANAME`
  - `robot_model`
  - `USE_GAT_EMBED`
  - current working directory is `nn_model/`
- For `run_post_training_pipeline.py`, remember:
  - it is **GAT-only**
  - it is **not** the final summary stage
  - it uses a hard-coded robot -> dataset filename map during calibration
  - it also fixes raw experiment settings inside the script
- For `run_mlp_experiments_all_models.py`, remember:
  - the file itself is part of the configuration
  - it is not fully generic in its current checked-in state

---

## Details

## 1. Adaptive Pipeline

### 1.1 Common Base: `gat_data_generation.py`

This is the common data generator for both GAT and MLP.

- It simulates `safe_control` rollouts.
- It samples robot state / obstacle configuration / CBF parameters.
- It computes labels:
  - `Safety Loss`
  - `Deadlock Time`
- It saves a graph dataset:
  - output format: `gat_datagen_<tag>_<num_samples>_<robot>_<controller>.pkl`

Important notes:

- The script is **not CLI-driven**.
- It is configured by editing values near the bottom of the file:
  - `controller_name`
  - `robot_model`
  - `num_samples`
  - `output_prefix`
- Current default bottom block is set up like:
  - `controller_name = "mpc_cbf"`
  - `robot_model = "DynamicUnicycle2D"`
  - `output_prefix = "gat_datagen_1112"`
  - `num_samples = 200000`

Typical usage:

```bash
python gat_data_generation.py
```

Outputs:

- root-level `gat_datagen_*.pkl`

Historical/canonical dataset names already present:

- `gat_datagen_1112_200000_DynamicUnicycle2D_mpc_cbf`
- `gat_datagen_1106_200000_Quad2D_mpc_cbf`
- `gat_datagen_1103_200000_Quad3D_mpc_cbf`
- `gat_datagen_1101_200000_KinematicBicycle2D_DPCBF_cbf_qp`

### 1.2 MLP-only Extra Step: `convert_pkl_csv.py`

MLP does not train on the raw graph directly.

This script:

- loads the `.pkl` graph dataset
- extracts flat features from the closest relevant obstacle
- writes a `.csv` version used by the MLP branch

Typical usage:

```bash
python convert_pkl_csv.py gat_datagen_1106_200000_Quad2D_mpc_cbf
```

Or run with no argument to convert the four standard datasets.

Outputs:

- root-level `gat_datagen_*.csv`

### 1.3 Copy Data into `nn_model/data/`

`nn_model/train_data_with_checkpoints.py` reads from `nn_model/data/`, not from repo root.

So for actual training, keep the training inputs in:

- `nn_model/data/gat_datagen_*.pkl`
- `nn_model/data/gat_datagen_*.csv`

Right now both root copies and `nn_model/data/` copies exist.

Recommended convention:

- treat `nn_model/data/` as the training source of truth
- treat root-level copies as archive or temporary staging copies

### 1.4 Training: `nn_model/train_data_with_checkpoints.py`

This single script handles both GAT and MLP.

It is controlled by editing the header constants:

- `DATANAME`
- `MODELNAME_SAVE`
- `SCALERNAME_SAVE`
- `robot_model`
- `USE_GAT_EMBED`

Behavior:

- `USE_GAT_EMBED = True`
  - trains the GAT-PENN branch using the `.pkl`
- `USE_GAT_EMBED = False`
  - trains the MLP-PENN branch using the `.csv`

It saves:

- best model: `nn_model/checkpoint/<MODELNAME_SAVE>.pth`
- 100-epoch checkpoints:
  - `nn_model/checkpoint/<MODELNAME_SAVE>_epoch_100.pth`
  - ...
  - `nn_model/checkpoint/<MODELNAME_SAVE>_epoch_1000.pth`
- for MLP, matching scaler `.save` files

Important:

- run this script from inside `nn_model/`
- because it uses relative paths like `data/...` and `checkpoint/...`

Typical usage:

```bash
cd nn_model
python train_data_with_checkpoints.py
```

### 1.5 GAT Post-Training: `run_post_training_pipeline.py`

This script is **GAT-only**.

It explicitly rejects non-GAT controllers.

What it does:

1. finds all epoch checkpoints for one robot
2. runs CCCP calibration for each epoch
3. writes:
   - `calibration_epoch_<epoch>.json`
   - `config_epoch_<epoch>.py`
4. runs `adaptation_experiment.py` once per epoch checkpoint

What it does **not** do:

- it does not create the final paper-style best-window comparison summary
- it does not automatically archive everything into `epoch_exp/<folder>/`

So this is best understood as:

- **epoch-wise calibration + epoch-wise raw experiment generation**

Typical usage:

```bash
python run_post_training_pipeline.py --robot Quad2D --controller "Online Adaptive MPC-CBF GAT"
```

Convenience tmux wrapper:

```bash
./run_training_tmux.sh my_session --robot Quad2D --controller "Online Adaptive MPC-CBF GAT"
```

Again, `run_training_tmux.sh` only wraps this exact command in tmux.

Important:

- `run_post_training_pipeline.py` does **not** infer the calibration dataset from the checkpoint metadata.
- Inside `run_calibration()`, it uses a hard-coded `robot_model -> data_pkl filename` mapping.
- If you train a new GAT model on a differently named `.pkl`, you should update that mapping before trusting the calibration results.
- Inside `run_experiment()`, it also fixes raw rollout settings such as `MAX_T='30.0'` and `OBS_SET_COUNT='400'`.

### 1.6 GAT Final Comparison: `run_best_window_analysis_and_experiments.py`

This is the final comparison script for the archived GAT experiment folders in `epoch_exp/`.

Input expectation:

- a folder like `epoch_exp/quad2d_1111/`
- containing:
  - epoch-wise GAT CSVs
  - `config_epoch_*.py`
  - `calibration_epoch_*.json`

What it does:

1. finds the best 100-row window for each epoch CSV
2. loads the corresponding `config_epoch_<epoch>.py`
3. re-runs baseline controllers on exactly those `obs_id`s
4. writes `results_summary_epoch_*_threshold_*.txt`

Typical usage:

```bash
python run_best_window_analysis_and_experiments.py quad2d_1111
```

Important:

- this script assumes the per-epoch GAT outputs are already archived into `epoch_exp/<folder>/`
- in the current repo state, that archival has already happened
- epoch-to-config pairing is inferred by sorting CSV filenames by timestamp
- because of that, each `epoch_exp/<folder>/` should contain one clean time-ordered run family, not a mix of unrelated CSV dumps

### 1.7 MLP Final Comparison: `run_mlp_experiments_all_models.py`

This is the main final-evaluation script for the MLP branch.

What it does:

1. finds MLP checkpoints and scalers
2. loads best-window `obs_id`s from `epoch_exp/<folder>/`
3. runs CCCP calibration for each MLP checkpoint
4. runs `adaptation_experiment.py`
5. moves results into `epoch_exp/<folder>/`
6. writes `results_summary_epoch_*_mlp_threshold_*.txt`

Typical usage:

```bash
python run_mlp_experiments_all_models.py
```

Important:

- this script is currently partly hard-coded by maps near the top:
  - `CONTROLLER_MAP`
  - `DYNAMICS_FOLDER_MAP`
  - `HIGH_FIXED_PARAM_CONTROLLER`
  - `CSV_DATA_MAP`
  - `GAMMA_DIM_MAP`
- in the current checked-in version, several dynamics are commented out
- so before rerunning all dynamics, those maps should be re-enabled / checked

### 1.8 Where Adaptive Results Live

Final adaptive results are in `epoch_exp/`.

What to check first:

- per-dynamics latest folders:
  - `du2d_1113`
  - `quad2d_1111`
  - `quad3d_1207`
  - `kinematicbicycle2D_1210`
- automatic summaries:
  - `results_summary_epoch_*.txt`
  - `results_summary_epoch_*_mlp_threshold_*.txt`
- curated representative summaries:
  - `best_results_summary_*.txt`
  - `best_mlp_results_summary_*.txt`

Recommended interpretation:

- `results_summary_...` = script-generated source result
- `best_...summary...` = hand-picked representative final result

## 2. BarrierNet Pipeline

### 2.1 Full BarrierNet Flow

BarrierNet is a separate branch from the adaptive PENN/GAT/MLP pipeline.

Core files:

- `safe_control/position_control/BarrierNet/generate_dataset.py`
- `safe_control/position_control/BarrierNet/train.py`
- `run_barriernet_experiments_all_models.py`
- `summarize_barriernet_results.py`
- `analyze_barriernet_p_values.py` (optional diagnostics)

Important current-status note:

- The current checked-in default BarrierNet evaluation path is focused on:
  - `DynamicUnicycle2D`
  - `Quad2D`
  - `KinematicBicycle2D_DPCBF`
- `Quad3D` has historical BarrierNet results in `barriernet_results/`, but it is not included in the current default `run_barriernet_all_models.sh` / `run_barriernet_experiments_all_models.py` flow.

### 2.2 Dataset Generation

`generate_dataset.py`:

- uses baseline `cbf_qp` rollouts
- collects `[z, ctx, u_ref, u*]`
- writes:
  - `safe_control/position_control/BarrierNet/data/<ROBOT>_data_train.mat`
  - `..._valid.mat`
  - `..._test.mat`

It supports environment-variable control:

- `BN_ROBOT_MODEL`
- `BN_TARGET_ROWS`
- `BN_MAX_SIMS`
- `BN_N_PROCESSES`
- `BN_N_SIMS`
- `BN_DT`
- `BN_T_MAX`

Typical usage:

```bash
BN_ROBOT_MODEL=Quad2D BN_TARGET_ROWS=200000 BN_MAX_SIMS=10000 python safe_control/position_control/BarrierNet/generate_dataset.py
```

### 2.3 Training

`train.py`:

- loads the generated `.mat` splits
- trains BarrierNet
- saves:
  - `safe_control/position_control/BarrierNet/checkpoints/<ROBOT>_barriernet.pth`
  - matching `_meta.json`

Typical usage:

```bash
BN_ROBOT_MODEL=Quad2D BN_EPOCHS=100 BN_BATCH_SIZE=64 python safe_control/position_control/BarrierNet/train.py
```

### 2.4 Experiment Stage

`run_barriernet_experiments_all_models.py`:

- finds the most recent baseline CSV from `epoch_exp/` or `sim_results/`
- picks the best 100-row window from that baseline
- runs BarrierNet on the same `obs_id`s
- writes result CSVs under `barriernet_results/<Dynamics>/`
- after each successful run, it also calls `summarize_barriernet_results.py` and `analyze_barriernet_p_values.py`

This is the key fairness step for BarrierNet comparison.

### 2.5 Summary and Optional P-value Analysis

`summarize_barriernet_results.py`:

- writes `.summary.txt` next to each BarrierNet CSV
- this is the part that turns raw BarrierNet CSVs into the text summaries used in handoff / comparison tables

`analyze_barriernet_p_values.py`:

- loads a trained BarrierNet checkpoint
- reports the learned `p1/p2` or `alpha` statistics on test data
- this is optional analysis, not part of the minimum "produce final result table" path

### 2.6 Shell Wrappers

Useful BarrierNet wrappers:

- `run_barriernet_all_models.sh`
  - all-in-one wrapper
- `run_barriernet_dynamic_unicycle.sh`
  - DU2D-only
- `run_barriernet_quad2d_full.sh`
  - Quad2D-only
- `run_barriernet_remaining_models.sh`
  - Quad2D + Kinematic DPCBF

Current recommendation:

- keep `run_barriernet_all_models.sh`
- keep the model-specific wrappers if you still use them
- archive `run_final_barriernet_pipeline.sh`
- remember that the current all-model wrapper does **not** include `Quad3D`
- remember that its experiment phase still needs baseline adaptive CSVs to exist first

### 2.7 Where BarrierNet Results Live

BarrierNet final outputs are in `barriernet_results/`.

Per-dynamics:

- `barriernet_results/DynamicUnicycle2D/`
- `barriernet_results/Quad2D/`
- `barriernet_results/Quad3D/`
- `barriernet_results/KinematicBicycle2D_DPCBF/`

What to check first:

- latest `sim_results_*.csv`
- same-name `.summary.txt`

## 3. Plot Scripts

### `plot_safety_loss_function_grid.py`

Purpose:

- paper/analysis plot
- plots the safety-loss surface over a 2D spatial grid
- sweeps several `lambda_1` and `delta_theta` combinations
- shows how the analytical safety-loss function changes around an obstacle

Use this when:

- you want a figure explaining the safety-loss definition itself
- you want paper-quality static surfaces

### `plot_uncertainties_grid.py`

Purpose:

- paper/analysis plot
- loads a trained PENN model and scaler
- evaluates the model over a 3D feature grid:
  - distance
  - velocity
  - relative angle
- repeats that over a `3 x 3` grid of gamma pairs

Use this when:

- you want to visualize how predicted safety loss changes across state / gamma settings
- you want a model-behavior figure, not a rollout figure

### `test_plot.py`

Purpose:

- real-time visualization during simulation
- shows predicted GMM distributions for a few selected gamma pairs

Use this when:

- you want an interactive qualitative sanity check during rollout
- you want to see the PENN/GMM outputs online

## 4. Root-Level Script Inventory

### Keep: Core Runtime / Pipelines

- `adaptation_experiment.py`
- `online_adaptive_cbf.py`
- `online_cbf_config.py`
- `gat_data_generation.py`
- `convert_pkl_csv.py`
- `run_post_training_pipeline.py`
- `run_best_window_analysis_and_experiments.py`
- `run_mlp_experiments_all_models.py`
- `run_barriernet_all_models.sh`
- `run_barriernet_dynamic_unicycle.sh`
- `run_barriernet_quad2d_full.sh`
- `run_barriernet_remaining_models.sh`
- `run_barriernet_experiments_all_models.py`
- `summarize_barriernet_results.py`
- `analyze_barriernet_p_values.py`
- `safety_loss_function.py`
- `setup.py`

### Keep: Plot / Analysis Utilities

- `plot_safety_loss_function_grid.py`
- `plot_uncertainties_grid.py`
- `test_plot.py`
- `visualize_single_simulation.py`
- `analyze_and_visualize_failing_cases.py`
- `check_data_histogram.py`
- `monitor_dataset_generation.sh`

### Archive Candidates

- `data_generation.py`
  - older predecessor to `gat_data_generation.py`
- `run_full_pipeline.sh`
  - old DU2D-specific wrapper
- `run_final_barriernet_pipeline.sh`
  - older BarrierNet wrapper
- `run_best_window_experiments.py`
  - older narrow helper, superseded by `run_best_window_analysis_and_experiments.py`
- `calculate_metrics.py`
  - hard-coded old analysis script
- `list_valid_obs_ids.py`
  - one-off helper
- `results_summary.txt`
  - old loose summary file

### Disposable Artifacts

- `*.log`
  - keep only if you need provenance for a past run
- root-level `gat_datagen_*.pkl`
- root-level `gat_datagen_*.csv`
  - if `nn_model/data/` is treated as the canonical training input location

## 5. Practical Reproduction Notes

### Reproducing a GAT run

1. edit `gat_data_generation.py`
2. run `python gat_data_generation.py`
3. copy `.pkl` to `nn_model/data/`
4. edit `nn_model/train_data_with_checkpoints.py`
5. run from `nn_model/` with `USE_GAT_EMBED=True`
6. run `python run_post_training_pipeline.py --robot <Robot> --controller "<GAT controller>"`
   - if you trained on a new dataset filename, update the calibration dataset mapping inside `run_post_training_pipeline.py` first
7. archive the produced per-epoch files into `epoch_exp/<folder>/`
8. run `python run_best_window_analysis_and_experiments.py <folder>`

### Reproducing an MLP run

1. edit `gat_data_generation.py`
2. run `python gat_data_generation.py`
3. run `python convert_pkl_csv.py <dataname>`
4. copy `.pkl` and `.csv` to `nn_model/data/`
5. edit `nn_model/train_data_with_checkpoints.py`
6. run from `nn_model/` with `USE_GAT_EMBED=False`
7. configure maps in `run_mlp_experiments_all_models.py`
8. run `python run_mlp_experiments_all_models.py`

### Reproducing a BarrierNet run

Fastest practical route:

```bash
./run_barriernet_all_models.sh
```

Manual route:

1. run `generate_dataset.py`
2. run `train.py`
3. run `run_barriernet_experiments_all_models.py`
4. optionally rerun `summarize_barriernet_results.py`
5. optionally rerun `analyze_barriernet_p_values.py`

## 6. Known Caveats

- `run_post_training_pipeline.py` is GAT-only.
- `run_post_training_pipeline.py` also hard-codes the calibration dataset filename per robot.
- `run_post_training_pipeline.py` also hard-codes raw experiment settings such as `MAX_T=30.0` and `OBS_SET_COUNT=400`.
- `run_training_tmux.sh` is not a universal train/eval script.
- `run_barriernet_all_models.sh` does not currently include `Quad3D`.
- `run_barriernet_all_models.sh` also assumes fairness baseline CSVs already exist for its experiment phase.
- `run_barriernet_experiments_all_models.py` also excludes `Quad3D` by default in its current checked-in configuration.
- `run_best_window_analysis_and_experiments.py` infers epoch ordering from CSV timestamps, so mixed or manually copied folder contents can confuse it.
- `run_mlp_experiments_all_models.py` is partially hard-coded and should be checked before rerunning all dynamics.
- `best_*summary*.txt` files appear to be curated copies, not purely automatic outputs.
