# Network Traffic Classification with Few-Shot Learning

<div align="center">
<pre>
 ______   __     __   __     ______     __         ______    
/\  ___\ /\ \   /\ "-.\ \   /\  __ \   /\ \       /\  ___\   
\ \  __\ \ \ \  \ \ \-.  \  \ \  __ \  \ \ \____  \ \  __\   
 \ \_\    \ \_\  \ \_\\"\_\  \ \_\ \_\  \ \_____\  \ \_____\ 
  \/_/     \/_/   \/_/ \/_/   \/_/\/_/   \/_____/   \/_____/ 
</pre>
</div>

## Project Overview

This repository implements methods for network traffic classification with support for standard machine-learning approaches and few-shot learning (FSL) algorithms. It provides data handling, model implementations (ML and DL), training and evaluation logic, and utilities to run combinatorial experiments. Typical use cases are benchmarking classifiers and experimenting with transfer- and meta-learning approaches on labeled network traffic datasets.

Key components:
- `src/main.py`: entry point to run single experiments.
- `src/run_experiments.sh`: helper script to launch multiple experiments combinatorially.
- `src/approach/`: implementations of approaches (DL and ML wrappers).
- `src/data/`: data loading, batching, and few-shot episode sampling.
- `src/trainer/`: training orchestration and experiment lifecycle.
- `config.yaml`: default configuration used to populate CLI defaults.

## Installation

Prerequisites:
- Python 3.8+ (verify with `python --version`).
- GNU `parallel` is optional but recommended for `run_experiments.sh`.

1. Place the dataset according to the path defined in `config.yaml` under `base_data_path`. Each dataset must then be placed in its own folder as defined in `dataset_config.py`.

2. Create and activate a virtual environment (example using `venv`):

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Note: If you need CUDA-enabled `torch`, install the appropriate `torch` wheel per your CUDA version (the `requirements.txt` contains a generic `torch` entry).

## Project structure

Top-level layout (important files and folders):

```
.
├── config.yaml              # Main configuration file
├── requirements.txt         # Project dependencies
├── README.md                # Documentation
├── src/                     # Source code
│   ├── main.py              # Application entry point
│   ├── run_experiments.sh   # Experiment launcher script
│   ├── approach/            # ML/DL methods: RelationNet, ProtoNet, MAML, etc.
│   ├── data/                # Data loading, samplers, and DataModule
│   ├── network/             # Neural network architectures
│   ├── trainer/             # Training logic and trainer factories
│   ├── callback/            # Checkpointing, logging, early stopping
│   └── util/                # Config, logging, args, and directory helpers
├── results/                 # Experiment outputs, as configured in config.yaml
└── log_dump/                # Batch/job logs
```

See the `src/approach` folder for concrete implementations of algorithms and the mapping between approach names and classes.

## Configuration

The repository uses `config.yaml` to supply default values for many CLI options. Edit `config.yaml` or override values with command-line arguments when running experiments.

Important config keys include `log_dir` (default output directory), dataset-related keys (e.g. `base_data_path`, `datasets`), training hyperparameters (learning rate, epochs), and approach-specific defaults.

## Command-line arguments and parameters

The project exposes arguments via `src/util/args_parser.py`. Below are the main groups of CLI arguments (names match the flags accepted by `main.py`).

- **Global / experiment control (defined in `src/util/args_parser.py`):**
  - `--seed` (int) — RNG seed (default from `config.yaml`).
  - `--gpu` (flag) — enable GPU if available.
  - `--n-thr` (int) — number of threads.
  - `--log-dir` (str) — output directory for logs and results.
  - `--approach` (str) — approach name to run (ML or DL approach identifier).
  - `--network` (str) — neural network backbone to use for DL approaches.
  - `--ckpt-path` (str) — path to a checkpoint `.pt` file to resume or evaluate.

- **Data-related (defined in `src/data/data_module.py` and `src/util/args_parser.py`):**
  - `--datasets` (one or more strings) — dataset names/paths to use.
  - `--is-flat` (flag) — treat PSQ input as flat structure.
  - `--return-quintuple` (flag) — return quintuple along with data and labels.
  - `--num-pkts` (int) — number of packets per biflow.
  - `--fields` (one or more from `PL, IAT, DIR, WIN, FLG, TTL`) — fields to include.
  - `--batch-size`, `--adapt-batch-size` (int) — batch sizes.
  - `--num-workers` (int) — dataloader workers.
  - `--pin-memory` (flag) — pin memory for DataLoader.
  - `--num-episodes`, `--k`, `--train-k`, `--train-q`, `--num-ways` — few-shot episode settings.

- **Deep-learning common args (defined in `src/approach/dl_module.py`):**
  - `--lr`, `--lr-strat`, `--sch-monitor` — learning rate and scheduler options.
  - `--max-epochs`, `--min-epochs` — training runtime.
  - `--optimizer` — optimizer choice (`adam`, `adamw`, `sgd`).

- **Approach-specific arguments (examples; each approach defines its own flags in `src/approach/*`):**
  - RelationNet: `--rn-hidden-dim` (int)
  - ProtoNet: `--pn-distance` (choices `euclidean`, `cosine`)
  - MAML: `--maml-adapt-steps`, `--maml-adapt-lr`, `--maml-first-order`
  - MetaOptNet: `--mon-svm-c-reg`, `--mon-svm-max-iters`, `--mon-normalize`
  - RFS: `--alpha`, `--gamma`, `--is-distill`, `--kd-t`, `--teacher-path`
  - NegativeMargin: `--nm-margin`, `--nm-inner-margin`, `--nm-temp`, `--nm-inner-temp`
  - KNN: `--knn-n-neighbors`, `--knn-weights`, `--knn-p`, `--knn-metric`
  - RandomForest: `--rf-criterion`, `--rf-n-estimators`, `--rf-max-depth`
  - XGB: `--xgb-n-estimators`, `--xgb-max-depth`, `--xgb-eval-metric`

If you need a complete list of available flags, inspect `src/util/args_parser.py` and the `add_appr_specific_args` implementations in `src/approach/`.

## Running a single experiment (using `main.py`)

You can manually run experiments by navigating to the `src` directory and executing:
```bash
# Example META-LEARNING: run MatchingNet, meta-training on <source_dataset> and meta-testining on <target_dataset>
python main.py --datasets <source_dataset> <target_dataset> --approach matching_net --max-epochs 20 --train-k 5 --train-q 5 --k 5 --num-ways 5 --seed 0 --log-dir ./results/example_run

# Example TRANSFER LEARNING: run Finetuning, pre-training on <source_dataset> and adaptation on <target_dataset>
python main.py --datasets <source_dataset> <target_dataset> --approach baseline --max-epochs 50 --adapt-epochs 20 --adapt-strat finetuning --k 5 --seed 0

# Example 1-TASK MACHINE LEARNING: run 3 Random Forest classifier, one for each dataset
python src/main.py --datasets <dataset_1> <dataset_2> <dataset_3> --approach random_forest --rf-n-estimators 1000 --is-flat --log-dir ./results/knn_run

# Example 1-TASK DEEP LEARNING: run a Scratch baseline
python src/main.py --datasets <dataset_1> --approach scratch --max-epochs 20 --log-dir ./results_scratch --gpu
```

Notes:
- Few-shot approaches require exactly two datasets: a non-few (source) and a few (target) dataset. The code enforces this when `is_fsl` is detected.
- Override any default by passing CLI flags; defaults are loaded from `config.yaml`.

## Running multiple experiments with `run_experiments.sh`

Purpose: `src/run_experiments.sh` generates and runs a set of experiment commands combinatorially (approaches × seeds × datasets). It writes per-experiment logs and (optionally) uses GNU `parallel` for concurrency.

Make the script executable and run it:

```bash
chmod +x src/run_experiments.sh
cd src
./run_experiments.sh --datasets "iot23 cic2018" --seed 0-2 --approach relation_net,proto_net --cpu 4 --log-keyword myrun
```

Script arguments (summary):
- `--datasets` (quoted space-separated list) — datasets used for the experiment set.
- `--seed` (comma-separated or range `X-Y`) — seeds to run.
- `--approach` (comma-separated) — approaches to run.
- `--is-flat` (flag) — passed to each `main.py` invocation when set.
- `--cpu` (int) — number of parallel jobs / CPU cores to use.
- `--extra-args` (string) — raw extra CLI arguments forwarded to `python main.py`.
- `--log-keyword` — additional tag appended to output directory names.

What the script produces:
- Per-experiment directories: by default created as `../results_<dataset_tag>_<approach>[_<keyword>]_6f_20p` (relative to `src/`). These contain `output.log` and `errors.log` and any artifacts saved by callbacks (checkpoints, metrics CSVs, JSON results).
- A batch joblog at `../log_dump/joblog.txt` when GNU `parallel` is used.

## Output locations, logs and checkpoints

- Default log directory: `log_dir` from `config.yaml` (default `"../results"`).
- `run_experiments.sh` builds per-run directories under `../results_*` and stores `output.log` and `errors.log` there.
- Trainer callbacks (in `src/callback/`) control saving of checkpoints and per-epoch metrics; check `src/callback/model_checkpoint.py` and `src/callback/save_train_log.py` for details.

## Acknowledgement

We thank the following open-source implementations that were used in this work:

- [LibFewShot](https://github.com/RL-VIG/LibFewShot)
- [learn2learn](https://github.com/learnables/learn2learn/)
- [GNU parallel](https://www.gnu.org/software/parallel/)
</div>
