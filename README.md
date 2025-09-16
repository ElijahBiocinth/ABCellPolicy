# A/B Cells Policy

---

![Downloads](https://img.shields.io/pypi/dm/ABCellPolicy.svg)  <!-- dm = downloads/month -->
![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=yourgithubusername.ABCellPolicy.readme&title=visitors)
![DOI](https://doi.org/10.5281/zenodo.16480754)

Although bright-field microscopy is one of the simplest and least invasive methods of live-cell imaging, it is often overlooked in favour of fluorescence-based techniques. Using only transmitted light, bright-field illumination captures cell morphology, boundary contours, and gross refractive index variations. This eliminates the need for exogenous labels or dyes. This label-free approach enables the continuous, long-term observation of living cells with minimal phototoxicity or disturbance to normal physiology. In time-lapse experiments, bright-field images provide a robust basis for automated segmentation and centroid tracking, yielding reliable measurements of area, shape, and displacement that can be compared directly across experimental conditions. In practice, the combination of high temporal resolution, gentle illumination and rich morphological content makes bright-field microscopy an ideal tool for the comprehensive quantitative profiling of cell migration dynamics in cell cultures, particularly when paired with advanced tracking pipelines, such as comparing **baseline (A)** vs **perturbation (B)**.

---

## The following stages are to be completed over a period of 2025 year:

<h3>
1. Deriving the A/B Cells Policy as a Robust Multi-Object Cell Pipeline for Time-lapse Microscopy
</h3>
</h2>
Larin, I., Panferov, E., Dodina, M., Shaykhutdinova, D., Larina, S., Minskaia, E., & Karabelsky, A. (2025). Deriving the A/B Cells Policy as a Robust Multi-Object Cell Pipeline for Time-Lapse Microscopy. International Journal of Molecular Sciences, 26(17), 8455.
</h2>

<table align="center">
  <tr>
    <td><img src="./assets/01PNG.gif" alt="mmSCs" width="380" /></td>
    <td><img src="./assets/01MSC.gif" alt="bmMSCs" width="366" /></td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <em>mmSCs on Millicell plate (left); bmMSCs on 48 well plate (right)</em>
    </td>
  </tr>
</table>

<h3>
  2. Deriving the A/B Cells Policy as a Reinforcement Learning Successor Feature.
  <img src="./assets/spinner (1).gif" 
       alt="spinner gif" 
       width="32" 
       align="top" 
       style="margin-left: 8px;"/>
</h3>

We introduce a diffusion transformer-like world model that maps high-dimensional per-frame and trajectory features into a latent dynamical representation. This model supports a successor feature formulation, whereby each cell state is encoded by the expected, discounted future occupancy of descriptive features. Forward-backward representations (where forward is predictive and backward is retrospective occupancy) are exploited to separate environment dynamics (where low-level transition regularities are largely preserved across A/B) from condition-specific feature weighting (where reward salience is considered). This allows hidden dynamic states to be inferred (trajectory segmentation beyond simple HMM phase labels), intervention effects to be quantitatively attributed as reweighting of successor measures rather than wholesale alteration of primitive dynamics, knowledge to be transferred between conditions (policy adaptation under altered feature importance) and active experimental design to be performed by maximising expected information gain on successor structure (e.g. frame sampling frequency and cell subset prioritisation). Using contrastive, metric or joint-embedding predictive learning to learn fixed-dimensional latent embeddings applied to successor-aware embeddings yields more robust similarity and dissimilarity scores between A and B than conventional feature averaging. This framework links descriptive statistics to a transferable, value-oriented policy representation. It opens up avenues for regenerative medicine, pharmacology, and early translational pipelines by providing an interpretable bridge, anchored in measurements, from in vitro imaging to in silico planning of intervention strategies.

<h3>
  3. Deriving the A/B Cells Policy as a Сell-Сhip Implementation.
  <img src="./assets/spinner (1).gif" 
       alt="spinner gif" 
       width="32" 
       align="top" 
       style="margin-left: 8px;"/>
</h3>

Based on electrically programmable cell chips, we present a framework for adaptive control that combines saccadic vision-inspired sensing with a Proximal Policy Optimisation (PPO) agent to steer cell fate decisions dynamically. A high-speed imaging module emulates biological saccades by executing rapid, discrete glances across the microelectrode array and extracting real-time features of nascent cytoskeletal alignment and marker expression. These sparse yet information-rich observations are fed into a PPO policy network, which updates its action distribution continuously to select voltage pulses that are spatially and temporally resolved and which bias differentiation trajectories and anisotropic growth. Through iterative trial-and-error exploration, tempered by a clipped surrogate objective, the system converges on stimulation protocols that maximise the desired lineage commitment and orientation fidelity, while minimising phototoxic exposure and computational overhead. This offers a scalable route to autonomous, high-throughput biofabrication.


---

## Project Layout

<details open>
  <summary>celltracker/</summary>

  - `__init__.py`
  - `config.py`
  - `progress.py`
  - `color_utils.py`
  - `geometry.py`
  - `features.py`
  - `tiling.py`
  - `assignment.py`
  - `reid.py`
  - `track.py`
  - `db.py`
  - `pipeline.py`
  - `visualization.py`
  <details>
    <summary>detectors/</summary>

    - `__init__.py`
    - `base.py`
    - `yolo_detector.py`
    - `cellpose_detector.py`
  </details>
</details>


<details>
  <summary>cellanalyser/</summary>

  - `__init__.py`
  - `config.py`
  - `aggregator.py`
  - `database.py`
  - `features.py`
  - `stats.py`
  - `visualization.py`
  - `pipeline.py`
</details>

<details>
  <summary>scripts/</summary>

  - `gif_cli.py`
  - `analyser_cli.py`
  - `track_cli.py`
</details>

- `requirements.txt`
- `README.md`

</details>

---


<h2>
  <img
    src="./assets/romb.gif"
    alt="romb gif"
    width="32"
    style="vertical-align: middle; margin-left: 8px;"
  />
  Installation
</h2>

Create a virtual environment (recommended) and install:

```bash
git clone git@github.com:ElijahBiocinth/ABCellPolicy.git
cd ABCellPolicy

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

```

If you need a specific CUDA build of PyTorch, install it first following the instructions on pytorch.org, then run `pip install -r requirements.txt`.

---
<h2>
  <img src="./assets/DNA.gif" 
       alt="DNA gif" 
       width="32" 
       align="top" 
       style="margin-left: 8px;"/>
  Data
</h2>


Our `data/` archive contains **eight separate YOLO datasets**, each in its own folder with its own `dataset.yaml`, `images/`, and `labels/` subdirectories.

## Fetch data

```bash
# Replace 1234567 with your actual Zenodo deposition number
DOI=1234567
URL="https://zenodo.org/record/${DOI}/files/data.zip"

curl -L -o data.zip "$URL"
mkdir -p data
unzip -o data.zip -d data/
rm data.zip
ls data/
```

| No. | Cell Type                                          | Culture Conditions                                                                                                      | Passages     |
|-----|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|--------------|
| 1   | Mononuclear cells (donor peripheral blood PBMCs)   | DMEM High Glucose + 10% FBS (20% for P1), 1% penicillin/streptomycin, 1% GlutaMAX™;<br>37°C, 5% CO2; medium change every 3-4 days              | up to P3     |
| 2   | Primary bronchial epithelial cells                 | Bronchial Epithelial Growth Medium (EGF, insulin, hydrocortisone, bovine pituitary extract);<br>37 °C, 5% CO₂; change every 48 h | up to P7     |
| 3   | C2C12 myoblasts                                    | High‑glucose DMEM + 10% FBS, 1% penicillin/streptomycin;<br>37 °C, 5% CO₂;<br>to differentiate: switch to 2% horse serum | P3–P16 (gm),<br>P18 (WT/CRISPR) |
| 4   | Human dental pulp stem cells (DPSCs)               | α‑MEM + 15% FBS, 1% penicillin/streptomycin, 50 µg/mL ascorbic acid;<br>37 °C, 5% CO₂; media change every 3 days           | P3–P6       |
| 5   | C2C12 Δdystrophin (CRISPR/Cas9 exon 23 deletion)    | Same as WT C2C12 (DMEM + 10% FBS);<br>genotype and dystrophin checked by PCR/Western blot every 3 passages              | up to P18    |
| 6   | Adipose‑derived mesenchymal stem cells (AD‑MSCs)   | DMEM/F12 + 20% FBS, 1% penicillin/streptomycin, 5 ng/mL EGF, 10 ng/mL bFGF;<br>37 °C, 5% CO₂; media change every 3 days   | up to P5     |
| 7   | MSCs under oxidative stress (5% H₂O₂)              | α‑MEM/DMEM + 10% FBS;<br>expose to 5% H₂O₂ in PBS for 2 h;<br>wash ×2 PBS, return to DMEM + 10% FBS; 20% O₂/5% CO₂         | analyzed at P8 |
| 8   | Mouse bone marrow mesenchymal stem cells (mBMMSCs)           | DMEM-F12 + 10% FBS, 1% penicillin/streptomycin, 1% GlutaMAX™;<br>37°C, 5% CO2; medium change every 2 days, passage every 4 days                   | up to P10        |


<h2>
  <img
    src="./assets/dog.gif"
    alt="dog gif"
    width="32"
    style="vertical-align: middle; margin-left: 8px;"
  />
  Quick Start
</h2>

### Quick Start A/B CellTracker

```bash
cd ABCellPolicy
export PYTHONPATH="$PWD:$PYTHONPATH"
python scripts/track_cli.py \
  --backend cellpose \
  --src '~/Scene_test' \
  --db '~/test.sqlite' \
  --out '~/outputs' \
  --first-n 5 \
  --device cuda:0 \
  --jit
```

#### CLI Arguments (core)
| Flag         | Description                                     | Default (in `celltracker/config.py`)              |
|--------------|-------------------------------------------------|---------------------------------------------------|
| `--backend`  |  `YOLO` or `cellpose`                           | `yolo`                                            |
| `--db`       | Path to SQLite database                         | `DEFAULT_DB_PATH` (`~/test.sqlite`)               |
| `--src`      | Folder with source images                       | `DEFAULT_SRC_FOLDER` (`~/Scene_test`)             |
| `--model`    | YOLO segmentation weights file                  | `DEFAULT_MODEL_PATH_ONLY_YOLO` (`~/weights_yolo.pt`)        |
| `--first-n`  | Verbose debug output on first N frames          | `DEBUG_FIRST_N_FRAMES` (`5`)                      |
| `--no-vis`   | Disable saving annotated frames                 | visuals are saved by default                      |
| `--out`      | Output folder for annotated frames              | `<src>/annotated`                                 |
| `--device`   | Compute device (`cpu` or `cuda:<index>`)        | `0` (interpreted as `cuda:0` in code)             |
| `--jit`      | Enable numba JIT optimizations                  | disabled unless set                               |

You can extend the parser to expose more hyperparameters (e.g. `--min-area`, `--max-missed`, etc.).

Organize all your `.sqlite` label files into a single folder and run the cleaning pipeline in one go.

If used Cellpose see: https://github.com/MouseLand/cellpose
```
pip install --upgrade pip
pip install 'cellpose[gui]'
```

---

### Quick Start A/B CellAnalyser

```bash
cd ABCellPolicy
export PYTHONPATH=$PWD
python3 scripts/analyse_cli.py \
  --db_paths '{"1":"~/test1234.sqlite","2":"~/test234.sqlite","3":"~/test34.sqlite"}' \
  --out_dir results_labels_test \
  --px_to_um 1 \
  --frame_dt 1 \  #seconds
  --smooth_window 15 \
  --alpha_test 0.005 \
  --perform_stats true \
  --n_threads 4

```
#### CLI Arguments
| Flag               | Description                                                      | Default (in `analyser_cli.py`) |
| ------------------ | ---------------------------------------------------------------- | ------------------------------------------ |
| `<DB_FILE>…`       | One or more SQLite databases to analyse (supports glob/symlinks) | —                                          |
| `-o`, `--out-dir`  | Output directory for CSV and PNG                                 | `DEFAULT_OUTPUT_DIR` (`"./output"`)        |
| `--static`         | Compute static metrics (area, eccentricity, etc.)                | disabled                                   |
| `--dynamic`        | Compute dynamic metrics (MSD, directional persistence, etc.)     | disabled                                   |
| `--stats`          | Run statistical tests (Friedman, Wilcoxon, Tukey HSD)            | disabled                                   |
| `--plot`           | Generate and save visualizations                                 | disabled                                   |
| `--px-to-um FLOAT` | Pixel-to-micron conversion factor                                | `PX_TO_UM` (`1`)                           |
| `--frame-dt FLOAT` | Time between frames in seconds                                   | `FRAME_DT` (`1.0`)                         |

---

### Quick Start gif CLI

```bash
cd ABCellPolicy
export PYTHONPATH=$(pwd)
python scripts/gif_cli.py \
  --scene-name myscene \
  --db-path /path/to/scene.sqlite \
  --image-dir /path/to/images \
  --out-dir ./output \
  --fps 30 \
  --alpha 0.5 \
  --contour-thick 3 \
  --px-to-um 1.0 \
  --make-gif
```

<h4>
  All of console logs for progress and debug info
  <img
    src="./assets/romb.gif"
    alt="romb gif"
    width="16"
    style="vertical-align: middle; margin-left: 8px;"
  />
</h4>

---

  
## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

This project is licensed under the MIT License. See the full text in the `LICENSE` file.

In short: permission is granted, free of charge, to any person obtaining a copy of this software and associated documentation files to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to including the copyright notice and this permission notice. 
The software is provided "as is", without warranty of any kind.
