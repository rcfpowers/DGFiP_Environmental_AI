# DGFiP Model Energy Monitoring Tool

---

## TODO: Priority Improvements

The following items must be addressed to move from a proof-of-concept to a production-grade energy monitoring system.

### 1. Data Infrastructure

1. **Configure data source paths for DGFiP's environment**
   - Replace the hardcoded `USAGE_FILES` constants with environment variables or a config file
   - Update `load_usage()` and `load_energy()` to point to the correct internal data paths

2. **Provide model-level token and request data disaggregated by model and time period**
   - Currently the usage CSV aggregates all models to a daily total, making per-model energy estimation impossible except on single-model days
   - The usage source must expose: total tokens per model per hour (or finer), request counts per model, and ideally input vs. output token split

3. **Provide model run timestamps at 5 minute-level granularity**
   - Required to identify server idle periods and subtract idle energy from active energy
   - Without this, the current estimate is an upper bound that overstates energy by up to ~10%
   - Timestamps should record when each model starts and stops serving requests

4. **Supply GPU-to-model assignment data**
   - Current energy data comes from the PDU at the voie level, not per GPU
   - If multiple models share a rack, their energy cannot be separated without knowing which GPU hosts which model
   - Implement `nvidia-smi` polling or an orchestrator log that maps model name → GPU ID → voie

### 2. GPU-Level Monitoring

5. **Implement `nvidia-smi` integration**
   - Replace PDU-level energy data with per-GPU power draw via `nvidia-smi --query-gpu=power.draw --format=csv,noheader`
   - Log at 10-minute intervals minimum; 5-min preferred
   - Capture both power draw (W) and memory utilisation (MiB) per GPU

### 3. Idle Energy Estimation

6. **Enable the `estimate_idle_energy()` function once run timestamps are available**
   - The function body already exists but returns `None` with a warning
   - Once model run start/stop times are available, uncomment the quantile-based idle baseline logic
   - Validate by checking that idle power aligns with known server specs (e.g. H100 idle ~70 W)

### 4. CO₂ and Geographic Scope

7. **Switch to historical CO₂ data for reporting periods older than 1.5 months**
   - The real-time RTE zip only covers the past ~6 weeks
   - For analysis covering January 2026 or earlier, use the consolidated historical archive: `https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-Consolide.zip`
   - Add a date-range check at the start of `fetch_rte_co2_factor()` to automatically choose the correct source

### 5. Model Coverage

8. **Add January 2026 usage data when it becomes available**
   - No January data was included in this iteration; the `USAGE_FILES` list must be updated

9. **Expand the energy estimate to all deployed models, not just `gte-Qwen2-1-5B-instruct`**
    - `estimate_wh_per_1000_tokens()` currently hard-codes two specific dates and one model
    - Once per-model hourly data is available, replace the `clean_dates` filter with a generalised loop over all models

10. **Maintain and extend `MODEL_HF_REPO` and `MODEL_PARAMS_B_KNOWN` mappings**
    - Any newly deployed model should be added to `MODEL_HF_REPO` if a HuggingFace equivalent exists
    - If HuggingFace does not publish parameter counts (e.g. internal fine-tunes), add the count manually to `MODEL_PARAMS_B_KNOWN`

### 6. Input/Output Token Split

11. **Distinguish between input (prompt) and output (completion) tokens in usage data**
    - Output tokens are significantly more energy-intensive to generate than input tokens
    - The current estimate treats all tokens equally, which understates energy for generation-heavy workloads
    - Request that the usage export includes `input_tokens` and `output_tokens` as separate columns

### 7. Automation and Integration

12. **Schedule the pipeline to run automatically**
    - Write results to a persistent output location

13. **Surface estimates in the model selection decision process**
    - Feed Wh/1000 tokens and gCO₂e/1000 tokens into the model catalogue or deployment dashboard
    - Consider a pre-deployment estimation step using HuggingFace metadata and benchmark data

---

## 1. Overview

This tool automates the collection, aggregation, and analysis of energy consumption and carbon emissions associated with large language models (LLMs) deployed by DGFiP on its on-premise GPU infrastructure. It was developed as part of a collaboration between Polytechnique Masters of DEPP and the DGFiP.

The primary outputs are:

- **Wh / 1,000 tokens** — energy consumed per unit of LLM output
- **gCO₂e / 1,000 tokens** — carbon emissions per unit of LLM output
- Monthly and quarterly peak energy per voie
- Model parameter counts (from HuggingFace API or hardcoded fallback)
- Daily energy and CO₂ time series for each voie

---

## 2. Architecture and Data Flow

The pipeline is a single Python script that executes the following steps in sequence:

1. **Load usage data** — reads daily model usage CSVs from S3. Each row corresponds to one model on one day, providing total tokens, request counts, and spend.
2. **Load energy data** — scans the S3 bucket for energy files matching the naming convention `NNN_XN_VoieN_YYYYMMDD.csv` and computes per-interval energy consumption (Wh) from cumulative Wh readings.
3. **Build daily energy series** — filters to the relevant voie (currently hardcoded to `101_J37_Voie1`), aggregates to daily totals, subtracts idle energy if available, and attaches RTE CO₂ factors.
4. **Merge usage and energy** — joins the daily energy series to the usage data on date, computing Wh per 1,000 tokens and gCO₂e per 1,000 tokens for each day.
5. **Estimate anchor model efficiency** — isolates the two days in February 2026 where only a single model (`gte-Qwen2-1-5B-instruct`) was running, producing a clean estimate.
6. **Peak analysis** — computes monthly maximum energy and the five highest energy intervals across the observation window.
7. **Model parameter lookup** — queries the HuggingFace API for parameter counts; falls back to a hardcoded dictionary for models not listed there.

---

## 3. Configuration and Setup

### 3.1 Dependencies

```bash
pip install pandas numpy matplotlib huggingface_hub boto3 s3fs xlrd
```

---

### Step 4 — Set your data file paths

Open the main script and update the following variables near the top of the file:

| Variable | Description |
|---|---|
| `USAGE_FILES` | List of usage CSV filenames, e.g. `["usage_file_MM_1.csv", "usage_file_MM_2.csv"]` |
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
| `merged_df` | Full DataFrame joining usage and energy data, with `wh_per_1000_tokens` and `co2_g_per_1000_tokens` columns |
| `wh_per_1000` | Float, the anchor model Wh per 1,000 tokens estimate |
| `energy_daily` | Daily energy DataFrame with CO₂ factors attached |
| `monthly_peaks` | Monthly maximum energy per voie |
| `top5_peaks` | Five highest energy intervals across the observation window |
| `params_map` | Dictionary of model name → parameter count in billions |

The script also writes `energy.csv` to `/home/onyxia/work/` and produces a matplotlib time-series plot of energy by voie during `load_energy()`.

---

## 8. Contact and Ownership This tool was developed by the ENSAE/DEPP team in collaboration with the DGFiP Datalab. For questions about the methodology, data pipeline, or energy estimation approach, contact the robert.powers@polytechnique.edu, anne.thebaud@polytechnique.edu, margot.martin@polytechnique.edu, or letizia.gaggiotti@polytechnique.edu.