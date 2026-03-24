# AI Energy & CO2 Monitoring Tool
*Estimating energy consumption and carbon footprint of LLM inference*

> **Scope:** This README covers the **ex post** estimation pipeline implemented in
> [`DGFiP_Environmental_AI/ex_post_model_estimation/Monitoring_tool.py`](DGFiP_Environmental_AI/ex_post_model_estimation/Monitoring_tool.py).
> The ex-ante estimation is not covered here, but the code is included in the repository for future reference.

📋 [TODO](DGFiP_Environmental_AI/Report and Presentation/TODO.md) &nbsp;|&nbsp; 📖 [Technical Documentation](DGFiP_Environmental_AI/Report and Presentation/DOCUMENTATION.md)

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
git clone <your-repository-url>
cd <repository-folder>
```

Replace `<your-repository-url>` with the actual URL of the repository.

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
They are: `pandas`, `numpy`, `matplotlib`, `huggingface_hub`, `boto3`, `s3fs`, `xlrd`.

If you prefer to install them manually beforehand, run:

```bash
pip install pandas numpy matplotlib huggingface_hub boto3 s3fs xlrd
```

---

### Step 4 — Configure your AWS / S3 credentials

The tool reads usage and energy data from an S3 bucket. You need valid AWS credentials configured on your machine. The recommended way is:

```bash
aws configure
```

You will be prompted for:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g. `eu-west-1`)

Alternatively, set environment variables directly in your terminal:

```bash
export AWS_ACCESS_KEY_ID=your_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_here
```

---

### Step 5 — Set your data file paths

Open the main script and update the following variables near the top of the file:

| Variable | Description |
|---|---|
| `USAGE_FILES` | List of usage CSV filenames, e.g. `["usage_file_MM_1.csv", "usage_file_MM_2.csv"]` |
| `S3_BUCKET` | Name of your S3 bucket, e.g. `"your-bucket-name"` |
| `OUTPUT_CSV` | *(Optional)* Path to save merged results as CSV. Set to `None` to skip. |

---

### Step 6 — Check the energy file naming convention

Energy files are automatically discovered in the S3 bucket. They must follow this exact naming pattern:

```
[3 digits]_[Letter][Digit]_Voie[Digit]_[YYYYMMDD].csv
```

Example: `101_J37_Voie1_20260201.csv`

Files that do not match this pattern will be ignored.

---

### Step 7 — Run the tool

From the repository folder, run:

```bash
python main.py
```

The script will:
1. Load usage and energy data from S3
2. Fetch real-time CO2 factors from RTE éCO2mix (falls back to 17 gCO2/kWh if the download fails)
3. Compute daily energy totals and merge with usage data
4. Estimate Wh per 1,000 tokens for the anchor model window
5. Run peak energy analysis (monthly peaks + top 5 quarterly peaks)
6. Look up model parameter counts from HuggingFace

Results are printed to the terminal. An energy plot is displayed and the energy data is saved to `/home/onyxia/work/energy.csv`.

---

## Project Structure

| File | Description |
|---|---|
| `main.py` | Main script (entry point) |
| `README.md` | This file |
| `TECHNICAL_DOC.txt` | Function reference, assumptions, and limitations |
| `energy.csv` | Output energy data (generated at runtime) |

---

## Data Sources

- **RTE éCO2mix (real-time):** https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-TR.zip
- **RTE éCO2mix (historical):** https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-Consolide.zip
- **HuggingFace model metadata:** https://huggingface.co


## Contact 

For questions about the methodology, data pipeline, or energy estimation approach, contact robert.powers@polytechnique.edu, anne.thebaud@polytechnique.edu, margot.martin@polytechnique.edu, or letizia.gaggiotti@polytechnique.edu.