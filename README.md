# ML-Based VM Placement for Energy-Efficient Cloud Scheduling

A machine-learning-driven virtual machine (VM) placement and consolidation simulator for cloud data centers. It classifies workload behaviour with **K-Means → Autoencoder → SVM**, places workloads onto physical machines (PMs) with a **least-loaded-first** scheduler, then **consolidates** them onto fewer PMs to reduce estimated energy use. A local web dashboard runs every script live and shows the results.

This project reproduces and adapts the approach in:

> Alelyani, A., Datta, A., Hassan, G. M. *A machine learning technique for optimizing virtual machine placement in data-centres.* Journal of Cloud Computing 15, 37 (2026). https://doi.org/10.1186/s13677-026-00851-3

The original paper uses the Google Cluster Trace with per-VM identifiers. That dataset was not practical to obtain, so this project uses a public pre-processed trace and adapts the method. **Read [Differences from the original paper](#differences-from-the-original-paper) before judging the numbers.**

---

## Table of contents

1. [Quick start](#quick-start)
2. [What you need](#what-you-need)
3. [Installation](#installation)
4. [Getting the dataset](#getting-the-dataset)
5. [Project structure](#project-structure)
6. [How to run](#how-to-run)
7. [How it works](#how-it-works)
8. [Results](#results)
9. [Differences from the original paper](#differences-from-the-original-paper)
10. [Limitations](#limitations)
11. [Troubleshooting](#troubleshooting)
12. [References and credits](#references-and-credits)

---

## Quick start

If the trained models and processed data are already in the project folder (`models/` and `data/processed/` are filled), you only need three steps:

```powershell
conda activate vm-scheduler
python app.py
```

Then open **http://127.0.0.1:5000** in your browser and click the buttons.

If you are setting up from scratch, follow [Installation](#installation) first, then [How to run](#how-to-run).

---

## What you need

| Requirement | Version | Why |
|---|---|---|
| Operating system | Windows 10/11 (tested). macOS and Linux should also work | Commands below use PowerShell |
| Python | **3.10** (3.9 to 3.11 work) | TensorFlow 2.15 does not support Python 3.12 or newer |
| Conda | Miniconda or Anaconda | Keeps this project's packages isolated |
| Git | Any recent version | Optional, only for cloning or version control |
| RAM | 8 GB minimum, 16 GB comfortable | Tested on 16 GB |
| GPU | **Not needed** | The models are small and train on CPU |
| Disk | About 3 GB free | TensorFlow is large (about 300 MB download) |

Python packages (installed in the next section):

| Package | Used for |
|---|---|
| numpy 1.26.4 | Numerical arrays (TensorFlow 2.15 requires numpy below 2.0) |
| pandas | Loading and processing CSV data |
| scikit-learn | K-Means, SVM, GridSearchCV, metrics, scaling |
| tensorflow 2.15.0 | The autoencoder |
| matplotlib, seaborn | All charts and plots |
| flask | The local web dashboard |
| joblib | Saving and loading the SVM (installed with scikit-learn) |

---

## Installation

### 1. Install Python tooling

- Miniconda: https://www.anaconda.com/download/success
- Git (optional): https://git-scm.com/downloads

Check they work:

```powershell
conda --version
git --version
```

### 2. Accept the conda terms of service (first-time conda only)

Newer conda versions refuse to create environments until you accept the channel terms. If `conda create` shows a `CondaToSNonInteractiveError`, run:

```powershell
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

### 3. Create the environment

```powershell
conda create -n vm-scheduler python=3.10 -y
conda activate vm-scheduler
python --version
```

The last command **must print `Python 3.10.x`**. If it prints anything else, the environment is not active. See [Troubleshooting](#troubleshooting).

If `conda activate` does nothing in PowerShell, run `conda init powershell`, close the window, and open a new one.

### 4. Install the packages

Run these inside the activated environment:

```powershell
pip install numpy pandas scikit-learn matplotlib seaborn
pip install tensorflow==2.15.0
pip install flask
```

Optional packages (installed during development, not used by the final scripts): `jupyterlab`, `simpy`, `mlflow`.

### 5. Verify the installation

```powershell
python -c "import numpy, pandas, sklearn, tensorflow, flask; print('All good!')"
```

You should see `All good!`. You may also see some TensorFlow messages above it. See the note under [Troubleshooting](#troubleshooting) about the harmless `MessageFactory` warnings.

---

## Getting the dataset

The pipeline uses one public file from Zenodo (no login needed):

- Page: https://zenodo.org/records/14564935
- **File you need:** `instance_usage_grouped_300_seconds_month.csv` (about 600 kB)
- Put it in: `data/raw/`

The same Zenodo record has three other small files. You can download them, but **the pipeline does not use them**.

Quick check that the file loaded correctly:

```powershell
python -c "import pandas as pd; df = pd.read_csv('data/raw/instance_usage_grouped_300_seconds_month.csv'); print(df.shape); print(df.columns.tolist())"
```

Expected: `(8064, 4)` and the columns `avg_cpu, avg_mem, avg_assigned_mem, avg_cycles_per_instruction`.

---

## Project structure

```
vm-scheduler-project/
├── app.py                         # Flask server for the web dashboard
├── templates/
│   └── dashboard.html             # Dashboard page (HTML, CSS, JS)
├── data/
│   ├── raw/                       # Downloaded dataset (input)
│   └── processed/                 # Files written by the pipeline
│       ├── features.csv           #   10 scaled features per sample
│       ├── labeled_features.csv   #   features + K-Means cluster label
│       └── latent_features.csv    #   5 autoencoder features + label
├── models/
│   ├── encoder_model.keras        # Trained encoder (autoencoder first half)
│   └── svm_classifier.pkl         # Trained SVM classifier
├── results/                       # All charts (PNG) and tables (CSV)
└── src/
    ├── feature_extraction.py      # Step 1: build and scale features
    ├── clustering.py              # Step 2: K-Means (k=3)
    ├── autoencoder.py             # Step 3: train autoencoder
    ├── svm_classifier.py          # Step 4: train SVM
    ├── entities.py                # VM, PM and PMPool classes
    ├── preliminary_scheduling.py  # Step 5: initial placement
    ├── consolidation.py           # Step 6: churn + consolidation + energy
    ├── evaluation.py              # Energy model and reports
    ├── baselines.py               # Round Robin, First Fit, Best Fit
    └── experiments/
        ├── run_full_comparison.py          # Experiments 1 and 2
        ├── run_multiseed_experiment.py     # 5-seed stability test
        ├── run_sensitivity_analysis.py     # 36-combination threshold study
        └── run_hyperparameter_comparison.py# k, latent size, SVM kernel
```

Always run commands **from the project root** (the folder that contains `app.py`). Running from another folder breaks the relative paths.

---

## How to run

### Option A: the web dashboard (recommended for demos)

```powershell
conda activate vm-scheduler
cd path\to\vm-scheduler-project
python app.py
```

Open **http://127.0.0.1:5000**. Stop the server with `Ctrl + C`.

What the page shows:

| Section | What it does |
|---|---|
| Buttons | Run Clustering, Preliminary Scheduling, Consolidation, Full Comparison, Multi-Seed Validation. Each runs the real Python script |
| Terminal Stream | The script's printed output, streamed live line by line |
| PM Pool Status | A grid of 700 squares. Teal = active PM, grey = idle PM. It updates as the script reports active and idle counts |
| Algorithm Comparison | After Full Comparison finishes: PM pools for our algorithm, Round Robin, First Fit and Best Fit, side by side |
| Results Gallery | All PNG charts from `results/`. Click one to open it in an in-page viewer, then use Previous, Next, or the arrow keys |

Notes for presenting:

- Clustering and Consolidation finish in seconds. Full Comparison and Multi-Seed take longer because they repeat the whole pipeline. Start them early.
- The gallery shows whatever PNG files are in `results/`, so run the experiments at least once before presenting.

### Option B: the command line

Run these in order. **Order matters**, because each step writes files the next step reads.

| # | Command | What it does | Time | Writes |
|---|---|---|---|---|
| 1 | `python src/feature_extraction.py` | Builds 10 features per sample and scales them to 0 to 1 | seconds | `data/processed/features.csv` |
| 2 | `python src/clustering.py` | K-Means with k=3 and cluster labels | seconds | `labeled_features.csv`, `cluster_profile.png`, `cluster_sizes.png` |
| 3 | `python src/autoencoder.py` | Trains the autoencoder (500 epochs) and exports 5 latent features | a few minutes | `latent_features.csv`, `encoder_model.keras`, `autoencoder_training.png` |
| 4 | `python src/svm_classifier.py` | Trains and tunes the SVM, evaluates on the test set | about a minute | `svm_classifier.pkl`, `confusion_matrix.png` |
| 5 | `python src/preliminary_scheduling.py --num_pms 700` | Places all VMs onto 700 PMs | under a minute | printed summary |
| 6 | `python src/consolidation.py` | Placement, then churn, then consolidation, with energy at each stage | under a minute | printed summary |
| 7 | `python src/experiments/run_full_comparison.py` | Experiment 1 and 2 against the baselines | a minute or two | `experiment1_*`, `experiment2_*` CSVs and PNGs |
| 8 | `python src/experiments/run_multiseed_experiment.py` | Repeats consolidation with 5 random seeds | a few minutes | `multiseed_*` CSV and PNG |
| 9 | `python src/experiments/run_sensitivity_analysis.py` | 36 threshold combinations | a few minutes | `sensitivity_*` CSV and heatmap |
| 10 | `python src/experiments/run_hyperparameter_comparison.py` | Compares k, latent size and SVM kernels | a few minutes | `kmeans_*.png`, `hyperparameter_comparison_summary.csv` |

If `models/` and `data/processed/` are already filled, you can skip steps 1 to 4 and start at step 5.

Try a different pool size to see capacity effects:

```powershell
python src/preliminary_scheduling.py --num_pms 500   # too few PMs, many VMs skipped
python src/preliminary_scheduling.py --num_pms 900   # more room, lower utilization
```

---

## How it works

### The big picture

```
Raw trace (8,064 five-minute windows)
        |
Feature extraction  -> 10 scaled features per sample
        |
K-Means (k=3)       -> labels each sample: CPU-intensive / memory-intensive / light
        |
Autoencoder         -> compresses 10 features to 5 latent features
        |
SVM (RBF)           -> predicts the type from the 5 latent features
        |
Preliminary scheduling -> least-loaded-first placement on 700 PMs
        |
Churn simulation       -> 30% of VMs leave (random, seeded)
        |
Consolidation          -> migrate VMs off underloaded PMs, idle the empty ones
        |
Energy evaluation      -> estimated watts from a SPECpower table
```

### The idea in plain words

A data center keeps many physical machines (PMs) switched on, and each runs several virtual machines (VMs). Machines that are only lightly used still draw a lot of power. If VMs can be packed onto fewer machines, the empty ones can sleep and save energy. Packing too tightly causes slowdowns, especially when many CPU-heavy VMs share one machine. So the scheduler needs to know **what kind of workload each VM is** before deciding where it goes.

### Stage 1: Data and feature extraction (`feature_extraction.py`)

The dataset has one row per five-minute window with `avg_cpu`, `avg_mem`, `avg_assigned_mem` and `avg_cycles_per_instruction`. It does **not** contain VM identifiers, so each window is treated as one independent **pseudo-VM sample**.

For each row the script adds rolling statistics over the previous 5 rows (max, min and standard deviation of CPU and of memory). This imitates the short-term variability the paper measures inside a VM's monitoring window.

The result is **10 features**: the 4 original columns plus `cpu_rolling_max/min/std` and `mem_rolling_max/min/std`. The first 5 rows are dropped (no full window), leaving **8,059 samples**. All features are MinMax-scaled to the range 0 to 1.

### Stage 2: K-Means clustering (`clustering.py`)

K-Means with **k = 3** (`random_state=42`) groups the samples into three behaviour classes. The cluster with the highest mean CPU is called CPU-intensive, the one with the highest remaining mean memory is memory-intensive, and the last is light. Cluster sizes are about 3,934, 2,454 and 1,671. These cluster ids are the **labels** used to train the classifier.

### Stage 3: Autoencoder (`autoencoder.py`)

A neural network that squeezes the 10 features into 5 numbers and rebuilds the original from them. The 5-number middle layer (the latent features) keeps the most important information.

| Setting | Value |
|---|---|
| Architecture | 10 → 8 → 5 → 8 → 10 |
| Activations | LeakyReLU (alpha 0.1) in hidden layers, sigmoid output |
| Initialisation | He normal |
| Optimizer | Adam, learning rate 0.0005 |
| Loss | Mean squared error |
| Training | 500 epochs, batch size 64, 80/20 train/validation split |
| Final loss | about 0.0012 |

LeakyReLU replaced plain ReLU after a "dying ReLU" bug: with plain ReLU, 2 to 3 of the 5 latent neurons got stuck at exactly 0 for every sample. The script now prints the standard deviation of each latent dimension so you can confirm all five are active.

The encoder half is saved to `models/encoder_model.keras`. Do not read the "accuracy" line in the training plot: it is a Keras side effect that means nothing for an autoencoder. Only the loss matters.

### Stage 4: SVM classifier (`svm_classifier.py`)

An SVM with an RBF kernel learns to predict the K-Means label from the 5 latent features.

- Split: 70% train, 10% validation, 20% test (stratified, seed 42)
- Tuning: GridSearchCV with 5-fold cross-validation over gamma {0.001, 0.01, 0.1, 1, 10} and C {0.1, 1, 10, 100}
- Best: gamma = 0.1, C = 100
- Test accuracy: **98.88%**, with precision, recall and F1 all about 0.99

**Read this number carefully.** The labels come from K-Means, and the SVM sees a compressed version of the same data K-Means used. So the SVM is largely **recovering K-Means' own decision boundary**. The high accuracy shows the classes are well separated in latent space. It does not show that the model can predict the future behaviour of unseen VMs, which is what the paper's harder per-VM setting tests (76.9%).

### Stage 5: Entities (`entities.py`)

- `VM`: id, `cpu_request`, `memory_request`, arrival time, predicted type, current PM.
- `PM`: capacity 1.0 for CPU and for memory. Can host a VM only if both resources would stay at or below the **75% upper threshold**. Also reports its CPU-intensive VM ratio.
- `PMPool`: the list of PMs. Sorts them least-loaded first, and finds overloaded, underloaded, active and idle PMs.

Each sample's `avg_cpu` and `avg_mem` are multiplied by **0.1** to become a VM's resource request, so many VMs fit on one PM.

### Stage 6: Preliminary scheduling (`preliminary_scheduling.py`)

1. Load the data, encoder and SVM, and classify every VM (encoder, then SVM).
2. Create 700 PMs. About 520 to 600 are needed for 8,059 VMs at the 75% cap. With 50 PMs most VMs are skipped.
3. For each VM in arrival order, sort PMs by load and place the VM on the first one that can host it.

This balances load across PMs. It does **not** by itself minimise the number of active PMs. That is the next stage's job.

### Stage 7: Churn and consolidation (`consolidation.py`)

**Churn.** With a single static placement every PM ends up about equally loaded, so none would qualify as underloaded. To imitate workloads finishing, `simulate_vm_churn` removes **30% of VMs at random** (2,417 VMs, seed 42).

**Consolidation.** A PM is *underloaded* if both its CPU and memory use are below the **35% lower threshold**. For each underloaded PM:

1. Try to move **every** VM on it to other active PMs, choosing the least-loaded valid target.
2. A target is valid if it can host the VM within the 75% cap **and** the move keeps its share of CPU-intensive VMs at or below **25%**.
3. **All-or-nothing:** if even one VM cannot find a target, the whole PM is left untouched and any partial moves are rolled back.
4. A PM left with no VMs becomes idle.

### Stage 8: Energy model (`evaluation.py`)

Energy is **estimated**, not measured. Each active PM's power is looked up from a SPECpower table for HP G5 servers using its CPU load, with linear interpolation between points. Idle PMs count as 10 W (sleep mode).

| CPU load | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Watts | 10 | 93.7 | 97 | 101 | 105 | 110 | 116 | 121 | 125 | 133 | 135 |

### Stage 9: Baselines (`baselines.py`)

- **Round Robin:** PMs in rotation, skipping any that cannot host the VM.
- **First Fit:** the first PM in creation order that can host the VM.
- **Best Fit:** the PM that would be fullest after the VM is added, without passing 75%.

All use the same VMs, the same 700 PMs and the same 75% cap.

---

## Results

All energy values are **estimates** from the SPECpower model.

### Reading the percentages

Two different reductions appear in this project, and they are not the same thing:

- **Consolidation-only reduction** compares energy *after churn, before consolidation* with energy *after consolidation*. Both states contain the same VMs, so this isolates what consolidation does. For seed 42 it is **10.03%** (75,496.99 W to 67,920.59 W).
- **Total reduction vs. initial placement** compares energy *before churn* (all 8,059 VMs) with energy *after consolidation* (5,642 VMs). This **includes the energy saved simply because 30% of VMs left**, so it overstates consolidation. For seed 42 it is 17.95% (82,776.91 W to 67,920.59 W). The final line printed by `consolidation.py` and the "Energy Reduction %" column of the multi-seed table use this total measure.

When describing the benefit of consolidation, use the **10.03%** figure.

### Classification

| Metric | Value |
|---|---|
| SVM test accuracy | 98.88% |
| Macro precision / recall / F1 | 0.989 / 0.988 / 0.988 |
| Autoencoder final loss | about 0.0012 |

### Experiment 1: placement strategy only (8,059 VMs, 700 PMs, no churn, no consolidation)

| Algorithm | Active PMs | Energy (W) |
|---|---|---|
| Our placement (least-loaded-first) | 700 | 82,776.91 |
| Round Robin | 635 | 77,616.36 |
| First Fit | 635 | 77,616.36 |
| Best Fit | 635 | 77,611.71 |

On placement alone, our scheduler uses **more** PMs than the three baselines. Load balancing and PM-count minimisation are different goals. Reducing PM count is the consolidation stage's role.

### Experiment 2: effect of consolidation (seed 42)

| Stage | Active PMs | Idle PMs | Energy (W) |
|---|---|---|---|
| After churn, before consolidation | 700 | 0 | 75,496.99 |
| After consolidation | 601 | 99 | 67,920.59 |

941 VMs were migrated. Average utilization rose from 0.45 / 0.42 to 0.52 / 0.49 (CPU / memory).

### Stability across 5 seeds (seeds 42, 7, 123, 2024, 99)

| Metric | Mean ± std |
|---|---|
| Active PMs after consolidation | 610.6 ± 7.1 |
| Idle PMs | 89.4 ± 7.1 |
| VMs migrated | 819 ± 98 |
| Total reduction vs. initial placement | 16.99% ± 0.67 |

The specific PMs freed vary by seed, but the outcome is tight. The 16.99% figure is the **total** measure described above. A per-seed consolidation-only figure is not yet reported, and from the numbers above it would be roughly 8 to 10%.

### Threshold sensitivity (36 combinations, seed 42)

Reductions below are consolidation-only (vs. the pre-consolidation state).

| Lower threshold | Reduction | Note |
|---|---|---|
| 0.25 | about 1.5% | Almost no PM counts as underloaded |
| 0.35 (default) | about 10.0% | Default setting |
| 0.45 | 11% to 34% | Depends heavily on the CPU-intensive cap |

- The **upper threshold had almost no effect** (0.65 to 0.80): after churn no PM gets near it.
- The best setting (lower 0.45, cap 0.35) reaches 34.3% but pushes average utilization to about 83%, close to the level where service-level violations become likely. The defaults (upper 0.75, lower 0.35, cap 0.25) are deliberately conservative.

### Hyperparameter checks

| Choice | Finding |
|---|---|
| K-Means k | Silhouette: k=2 is 0.310, k=3 is 0.256, k=4 is 0.265. k=2 scores higher. k=3 was kept to match the paper's three-type design |
| Latent size | Loss for 3 / 5 / 8 dimensions: 0.0038 / 0.0012 / 0.00001. Lower loss at 8 is expected (little compression). 5 was kept as a real 50% reduction |
| SVM kernel | Accuracy: linear 98.64%, RBF 98.39%, polynomial 97.46%. Linear nearly matches RBF, so the classes are close to linearly separable |

---

## Differences from the original paper

| Aspect | Original paper | This project |
|---|---|---|
| Dataset | Google Cluster Trace with per-VM IDs | Pre-processed public trace (Zenodo 14564935), no VM IDs |
| Unit classified | Individual VMs | Five-minute windows treated as **pseudo-VM samples** |
| Features | 16 | 10 (4 original columns + 6 rolling statistics) |
| Samples | 2,293 VMs for ML, 7,644 for scheduling | 8,059 pseudo-VM samples for both |
| Autoencoder | 16-12-5-12-16, ReLU | 10-8-5-8-10, LeakyReLU |
| Classifier accuracy | 76.9% | 98.88% (see the explanation above) |
| Energy saving reported | 51% vs. Google's scheduler | See [Results](#results). Not comparable, since the baselines differ |
| VM departures | Occur naturally in the trace | Simulated with random 30% churn |
| SLA metrics (SLATAH, PMD, SLAv) | Reported | Not implemented |

Describe the work as **classifying workload behaviour represented by pseudo-VM samples**, not as classifying VMs.

---

## Limitations

- **Pseudo-VMs.** A window is not a VM, so the classification task is easier than the paper's.
- **Labels come from K-Means.** The SVM accuracy shows separability, not independent predictive power.
- **Random churn.** The dataset has no VM lifecycles, so departures are simulated. Results depend on this assumption.
- **Estimated energy.** The model uses a SPECpower lookup for one server class. It does not measure real power or include cooling.
- **No SLA evaluation.** Violations and migration cost are not measured. The 75% cap is a proxy.
- **No ablation.** There is no run of consolidation without the ML classification.
- **Scale and data.** One dataset, 700 simulated PMs, 5 seeds.

---

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `python --version` shows 3.11 or 3.14 inside the project | The conda environment is not active. Run `conda activate vm-scheduler` and check again. A `(vm-scheduler)` prefix should appear in the prompt |
| `pip install tensorflow` says no matching distribution | You are on Python 3.12 or newer. Activate the Python 3.10 environment first |
| `CondaToSNonInteractiveError` | Run the three `conda tos accept` commands in [Installation](#installation) |
| `conda activate` does nothing in PowerShell | Run `conda init powershell`, then close and reopen the terminal |
| `ModuleNotFoundError` or `No such file or directory` for a script or data file | You are not in the project root. `cd` to the folder that contains `app.py` |
| `>>>` prompt and `python -c ...` gives `SyntaxError` | You are inside the Python shell. Type `exit()`, then run the command at the `PS C:\...>` prompt |
| `<<'PY'` gives a parser error | That heredoc syntax is for bash, not PowerShell. Use `python -c "..."` or a `.py` file |
| Many `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` lines | A harmless protobuf version warning. The scripts still run. Ignore it |
| TensorFlow messages about `oneDNN` or deprecated `tf.losses` | Harmless information and warnings |
| Autoencoder loss is huge (in the thousands) and flat | Input is not scaled to 0 to 1. Re-run `feature_extraction.py`, then `clustering.py`, then `autoencoder.py` |
| Some latent dimensions have a standard deviation of exactly 0 | Dead neurons. Use the LeakyReLU version of `autoencoder.py` and re-train |
| Many "VM could not be placed" warnings | Too few PMs. Use `--num_pms 700` |
| Dashboard shows no gallery images | `results/` has no PNGs yet. Run the scripts first |
| Comparison panels do not appear on the dashboard | Run **Full Comparison** and wait for "Run complete" first |
| `Address already in use` when starting `app.py` | Another program uses port 5000. Close it or stop the old server with `Ctrl + C` |
| Results differ slightly between runs | Seeds are fixed for K-Means, SVM splits and churn, but TensorFlow training is not fully deterministic. Small differences in autoencoder loss are normal |

---

## References and credits

- Alelyani, A., Datta, A., Hassan, G. M. (2026). *A machine learning technique for optimizing virtual machine placement in data-centres.* Journal of Cloud Computing 15:37. https://doi.org/10.1186/s13677-026-00851-3 (open access, CC BY 4.0)
- Fernández-Montes, A., Fernández Cerero, D. *DataCenter-Traces-Datasets.* Zenodo. https://doi.org/10.5281/zenodo.14564935 (processed from the Alibaba cluster-trace-v2018)
- Power model: SPECpower benchmark values for HP G5 servers, as used in the original paper (SPEC: https://www.spec.org/power/)
- Libraries: TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, seaborn, Flask, and plain HTML/CSS/JavaScript for the dashboard
