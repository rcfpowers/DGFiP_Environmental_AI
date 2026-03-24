# DGFiP Model Energy Monitoring Tool

---
This repository contains the software and methodology developed to assess the environmental impact of the DGFiP’s Generative AI platform. This project is a collaboration between the DGFiP and students from the master *Data and Economics for public policy (DEPP)* from École Polytechnique, ENSAE and Télécom.

## Repository structure
- **`data_exploration.py`**: This script automates the loading of iDRAC energy logs and LiteLLM usage data.
- **`Per-Model Energy & CO₂ Estimation`**: A module (or logic within the main script) that applies weighted attribution to disaggregate energy use by specific LLMs. 
- **`Data_description.ipynb`**: Descriptive statistics and visualizations of the raw data.
- **`Hugging_face_model_cards`**: Logic used to query the Hugging Face API to retrieve model metadata (parameters, quantization) required for energy weight corrections.
- **Presentation_environmental_impact_assessment.pdf**: The final summary slides detailing the Life-Cycle Assessment (LCA) results and strategic recommendations.
- **Report_Environmental_Impact_DGFiP.pdf**: A detailed description of all the phases of the LCA (Life-Cycle Assessment), including methodology, tool results, and the full bibliography.

## TO DO: Priority Improvements

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
pip install pandas numpy matplotlib huggingface_hub boto3 s3fs xlrd requests
```

### 3.2 Key Constants

The following constants at the top of the script must be reviewed and updated for each deployment context:

| Constant | Description |
|---|---|
| `USAGE_FILES` | List of usage CSV filenames to load |
| `MODEL_HF_REPO` | Mapping from DGFiP internal model names to HuggingFace repository IDs |
| `MODEL_PARAMS_B_KNOWN` | Fallback parameter counts (billions) for models not on HuggingFace |

---

## 4. Function Reference

### `fetch_rte_co2_factor(start_date, end_date, fallback_gco2_per_kwh)`

Downloads the latest 15-minute CO₂ emission factors (gCO₂/kWh) from RTE's éCO2mix real-time zip file and filters to the requested date range. If the download fails or the date range falls outside the file's coverage window (~1.5 months), the function falls back to a constant value (default: 17.0 gCO₂/kWh, the approximate French grid average).

### `load_usage(fs, bucket, files)`

Reads one or more usage CSV files and concatenates them. Normalises column names, parses dates, and coerces numeric columns. Returns a DataFrame with columns: `Date`, `Team_ID`, `Model`, `Spend`, `Requests`, `Successful`, `Failed`, `Total_tokens`.

### `load_energy(fs, bucket)`

Scans the data source for energy files matching the regex pattern `NNN_XN_VoieN_YYYYMMDD.csv`. Reads each file, deduplicates, and computes incremental energy usage (Wh) per 5-minute interval from the cumulative `Input Cumulated Energy Total (Wh)` column using a 2-period diff per voie. Also produces a time-series plot of energy by voie and writes a local `energy.csv`.

### `aggregate_energy_daily(energy_df)`

Filters the interval energy data to a single voie (currently hardcoded to `101_J37_Voie1`) and aggregates to daily totals. This filter must be updated when expanding to multiple voies.

### `estimate_idle_energy(energy_df, idle_quantile, idle_window_days)`

Currently a stub that returns `None` and prints a warning. The function is intended to estimate server idle energy consumption by computing a rolling lower quantile of `energy_used`. It requires model run timestamps to mask active periods before computing the baseline. See TODO item 6.

### `apply_idle_subtraction(energy_daily, energy_df)`

Calls `estimate_idle_energy()` and, if a result is returned, subtracts idle energy from total daily energy to produce `active_energy_wh`. Currently passes through `total_energy_wh` unchanged because idle estimation is disabled.

### `apply_co2_factors(energy_daily, energy_df)`

Fetches 15-minute RTE CO₂ factors, matches each energy interval to the closest preceding CO₂ reading using a backward `merge_asof` join, then computes an energy-weighted daily average CO₂ factor. Adds `co2_g_per_kwh`, `co2_source`, and `co2_kg` columns to the daily energy DataFrame.

### `merge_usage_and_energy(usage_df, energy_daily)`

Left-joins daily usage records to daily energy data on date. Computes `wh_per_1000_tokens` and `co2_g_per_1000_tokens` for each model-day record.

### `estimate_wh_per_1000_tokens(merged_df)`

Isolates records for `gte-Qwen2-1-5B-instruct` on 2026-02-01 and 2026-02-02 — the only two days in the dataset where a single model was running — and computes a point estimate of Wh per 1,000 tokens. This function is intentionally narrow and must be generalised once per-model hourly data is available.

### `build_params_map(models)`

Iterates over all model names found in the usage data. For each model, attempts to fetch parameter counts from the HuggingFace safetensors metadata via the API. Falls back to `MODEL_PARAMS_B_KNOWN` for models without HuggingFace entries. Returns a dictionary of model → parameter count (billions) and a list of excluded models.

### `compute_peak_analysis(energy_df)`

Returns two DataFrames: `monthly_peaks` (maximum `energy_used` per voie per calendar month) and `top5_peaks` (the five highest individual interval energy readings across all voies and the full observation window).

---

## 5. Assumptions

### 5.1 Energy Measurement

- Energy data is recorded as cumulative Wh at the PDU (Power Distribution Unit) level for each server rack lane (voie), not at the individual GPU level. A 2-period diff is applied to recover per-interval consumption.
- The relevant voie for the February 2026 estimate is `101_J37_Voie1`. All other voies are filtered out in `aggregate_energy_daily()`. This assumption must be revisited as more models and voies come online.
- Energy readings are taken every 5 minutes. The 2-period diff therefore yields energy consumed over a 10-minute window, which is then treated as the interval's consumption.
- Idle server energy is currently not subtracted. The estimate is therefore an upper bound on active inference energy. Idle power can account for up to approximately 10% of total consumption.

### 5.2 CO₂ Emissions

- CO₂ emission factors are sourced from RTE's éCO2mix real-time dataset at 15-minute intervals, representing the instantaneous carbon intensity of the French national electricity grid.
- The fallback CO₂ factor (17.0 gCO₂/kWh) is a reasonable approximation of the French grid average given its high nuclear share, but will underestimate emissions during high-demand periods and overestimate during periods of high renewable generation.
- If any model is deployed on infrastructure outside France, the French grid factor will underestimate carbon emissions significantly (e.g. the EU average is ~300 gCO₂/kWh).

### 5.3 Token and Usage Accounting

- All tokens on a given day are attributed equally to energy consumed that day, regardless of which model generated them. This is only valid on days when a single model is running.
- Input and output tokens are treated as equivalent in energy cost. In reality, output (generation) tokens require considerably more compute than input (prefill) tokens.
- The current anchor estimate (878.30 Wh / 1,000 tokens for `gte-Qwen2-1-5B-instruct`) is based on two days of data only and should be treated as illustrative rather than robust.

### 5.4 Model Parameters

- Parameter counts are retrieved from HuggingFace safetensors metadata where available. This reflects the number of weights stored on disk and may differ slightly from the number of active parameters if layers are skipped or quantised at runtime.
- For quantised models (e.g. FP8 variants), the stored weight count reflects the quantised format. Energy consumption may differ from a full-precision deployment of the same architecture.

---

## 6. Limitations

### 6.1 Time Framing

The RTE real-time data file covers only the past approximately 1.5 months. Analysis of historical periods (e.g. January 2026) requires switching to the consolidated historical archive. No January 2026 usage data was available at the time of writing, so the observation window is limited to February 2026.

### 6.2 Estimation Precision

Because idle energy cannot currently be separated from active inference energy, all energy attributed to model usage includes a non-trivial idle component. The study estimates this overstatement at up to 10%. Until model run timestamps are available, the `active_energy_wh` column is identical to `total_energy_wh`.

### 6.3 Data Granularity

The usage data does not currently provide model-level or intra-day granularity. On any day where more than one model is active, it is impossible to attribute energy to individual models. The clean estimate window is therefore limited to just two days — not sufficient to characterise variability in energy intensity across time of day, request type, or sequence length.

### 6.4 Model Characteristics

Parameter count is used as a proxy for relative model size but does not directly predict energy consumption. Architecture, quantisation level, batch size, sequence length, and hardware utilisation all affect real-world energy intensity. The HuggingFace API integration provides a foundation for comparing these characteristics, but ex-ante energy estimation from model metadata alone is inherently approximate.

### 6.5 GPU Assignment

The current pipeline has no information about which GPU is running which model at any given time. When multiple models are deployed simultaneously on a multi-GPU rack, energy cannot be disaggregated below the voie level without GPU-to-model assignment data.

### 6.6 Power Snapshot Limitation

Power estimates derived from a short snapshot period are not representative of sustained deployment load. Inference workloads are bursty and the power draw during a brief window may not reflect average consumption over a full day.

---

## 7. Outputs

`main()` returns the following objects and prints a summary to stdout:

| Variable | Description |
|---|---|
| `merged_df` | Full DataFrame joining usage and energy data, with `wh_per_1000_tokens` and `co2_g_per_1000_tokens` columns |
| `wh_per_1000` | Float, the anchor model Wh per 1,000 tokens estimate |
| `energy_daily` | Daily energy DataFrame with CO₂ factors attached |
| `monthly_peaks` | Monthly maximum energy per voie |
| `top5_peaks` | Five highest energy intervals across the observation window |
| `params_map` | Dictionary of model name → parameter count in billions |

The script also writes `energy.csv` to `/home/onyxia/work/` and produces a matplotlib time-series plot of energy by voie during `load_energy()`.

---

## 8. Contact and Ownership 
### This tool was developed by the ENSAE/DEPP team in collaboration with the DGFiP Datalab. For questions about the methodology, data pipeline, or energy estimation approach, contact the robert.powers@polytechnique.edu, anne.thebaud@polytechnique.edu, margot.martin@polytechnique.edu, or letizia.gaggiotti@polytechnique.edu.
