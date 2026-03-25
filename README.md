# AI Energy & CO2 Monitoring Tool
*Estimating energy consumption and carbon footprint of LLM inference*

This repository contains the software and methodology developed to assess the environmental impact of the DGFiP’s Generative AI platform. This project is a collaboration between the DGFiP and students from the master *Data and Economics for public policy (DEPP)* from École Polytechnique, ENSAE and Télécom.

## Repository Structure

- **`Ex Post Model Estimation/`** — Code to estimate model energy consumption with available usage and energy data. The following README is tailored to this folder, as it was the main deliverable of the project.
  - **`monitoring_tool.py`** — Automates the loading of iDRAC energy logs and LiteLLM usage data.

- **`Data Discovery/`** — Code to obtain a high-level understanding of usage and energy data.
  - **`Data_description.ipynb`** — Descriptive statistics and visualizations of the raw data.

- **`Ex Ante Model Estimation/`** — Code to estimate model energy consumption based on model characteristics, typically pulled from HuggingFace API.
  - **`Per-Model Energy & CO₂ Estimation/`** — Applies weighted attribution to disaggregate energy use by specific LLMs.
  - **`Hugging_face_model_cards/`** — Queries the HuggingFace API to retrieve model metadata (parameters, quantization) required for energy weight corrections.

- **`Documentation/`** — All written documentation of this project, including limitations, assumptions, and recommendations.
  - **`Presentation_environmental_impact_assessment.pdf`** — Final summary slides detailing the Life-Cycle Assessment (LCA) results and strategic recommendations.
  - **`Report_Environmental_Impact_DGFiP.pdf`** — Detailed description of all LCA phases, including methodology, tool results, and full bibliography.

> **Scope:** This README covers the **ex post** estimation pipeline implemented in
> [`ex_post_model_estimation/Monitoring_tool.py`](ex_post_model_estimation/Monitoring_tool.py).
> The ex-ante estimation is not covered here, but the code is included in the repository for future reference.

[TODO](Report%20and%20Presentation/TODO.md) &nbsp;|&nbsp; [Technical Documentation](Report%20and%20Presentation/DOCUMENTATION.md)

---

## Overview

This tool estimates the energy consumption and CO2 emissions of large language model (LLM) inference at DGFiP. It does this by combining three data sources:

- Model usage logs (tokens, requests, spend per team and model)
- Data center energy readings (Wh, from physical sensors per server "Voie")
- Real-time grid CO2 intensity (gCO2/kWh, from RTE éCO2mix)

From these inputs, the tool produces per-model estimates of energy use (Wh per 1,000 tokens) and carbon footprint (gCO2e per 1,000 tokens), along with peak energy analysis and model parameter lookups from HuggingFace.

---

## Step-by-Step Setup

### Step 1 — Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/rcfpowers/DGFiP_Environmental_AI.git
cd DGFiP_Environmental_AI
```


---

### Step 2 — Make sure Python is installed

This tool requires Python 3.9 or higher. Check your version with:

```bash
python --version
```

If Python is not installed, download it from: https://www.python.org/downloads/

---

### Step 3 — Install the dependencies

All required packages are installed automatically when the script runs.
They are: `pandas`, `numpy`, `matplotlib`, `huggingface_hub`, `boto3`, `xlrd`.

If you prefer to install them manually beforehand, run:

```bash
pip install pandas numpy matplotlib huggingface_hub boto3 xlrd
```

---

### Step 4 — Set your data file paths

Open **`monitoring_tool.py`** and update the following variables near the top of the file:

| Variable | Description |
|---|---|
| `USAGE_FILES` | List of usage CSV filenames, e.g. `["usage_file_MM_1.csv", "usage_file_MM_2.csv"]` |
| `ENERGY_FILES` | List of usage CSV filenames, e.g. `["energy_file_MM_1.csv", "energy_file_MM_2.csv"]` |
| `OUTPUT_CSV` | *(Optional)* Path to save merged results as CSV. Set to `None` to skip. |

---

### Step 5 — Check the energy file naming convention

Energy files are automatically discovered in local directory. They must follow this exact naming pattern:

```
[3 digits]_[Letter][Digit]_Voie[Digit]_[YYYYMMDD].csv
```

Example: `101_J37_Voie1_20260201.csv`

Files that do not match this pattern will be ignored.

---

### Step 6 — Run the tool

From the repository folder, run:

```bash
python monitoring_tool.py
```

The script will:
1. Load usage and energy data from local directory
2. Fetch real-time CO2 factors from RTE éCO2mix (falls back to 17 gCO2/kWh if the download fails)
3. Compute daily energy totals and merge with usage data
4. Estimate Wh per 1,000 tokens for the anchor model window
5. Run peak energy analysis (monthly peaks + top 5 quarterly peaks)
6. Look up model parameter counts from HuggingFace

Results are printed to the terminal. An energy plot is displayed and the energy data is saved to `energy.csv`.

---

## Data Sources

- **RTE éCO2mix (real-time):** https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-TR.zip
- **RTE éCO2mix (historical):** https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-Consolide.zip
- **HuggingFace model metadata:** https://huggingface.co


## Contact 

For questions about the methodology, data pipeline, or energy estimation approach, contact robert.powers@polytechnique.edu, anne.thebaud@polytechnique.edu, margot.martin@polytechnique.edu, or letizia.gaggiotti@polytechnique.edu.