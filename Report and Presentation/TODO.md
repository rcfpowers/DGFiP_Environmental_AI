## TODO: Priority Improvements

The following items must be addressed to move from a proof-of-concept to a production-grade energy monitoring system.

### 1. Data Infrastructure

1. **Configure data source paths for DGFiP's environment**
   - Replace the hardcoded `USAGE_FILES` and `ENERGY_FILES` constants with environment variables or a config file
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