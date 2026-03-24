import subprocess
import sys
import os
import s3fs
import pandas as pd
import matplotlib.pyplot as plt
import re
import requests
import zipfile
import io


def install():
    pkgs = ["pandas", "numpy", "matplotlib", "huggingface_hub", "boto3", "s3fs", "xlrd"]
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + pkgs)


install()

from huggingface_hub import HfApi

"""
TODO: Add the files here, or set up an automatic pool to pull usage and energy data
"""
USAGE_FILES = ["usage_file_MM_1.csv",
               "usage_file_MM_2.csv"]
OUTPUT_CSV = None

# List of models matching DGFiP's data to HuggingFace ID
MODEL_HF_REPO = {
    "gte-Qwen2-1-5B-instruct": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "Qwen2.5-Coder-32B-Instruct-fp8-W8A16": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Llama-3-3-70B-128k": "meta-llama/Llama-3.3-70B-Instruct",
    "Mistral-Small-24B-Instruct-2501-FP8-dynamic": "mistralai/Mistral-Small-24B-Instruct-2501",
    "/model/deepdml-faster-whisper-large-v3-turbo-ct2": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "qwen3vl32binstruct": "Qwen/Qwen3-VL-32B-Instruct",
    "gptoss20b": "openai/gpt-oss-20b",
    "gptoss120b": "openai/gpt-oss-120b",
    "dgfip-e5-large": "intfloat/e5-large",
}

# List of models whose parameters are not retievable from HuggingFace API
MODEL_PARAMS_B_KNOWN = {
    "Llama-3-3-70B-128k": 70.0,
    "Mistral-Small-24B-Instruct-2501-FP8-dynamic": 24.0,
    "/model/deepdml-faster-whisper-large-v3-turbo-ct2": 0.809,
    "dgfip-e5-large": 0.335,
    "qwen3vl32binstruct":   32.8,
}


def fetch_rte_co2_factor(start_date, end_date, fallback_gco2_per_kwh=17.0):
    """
    Fetch 15-minute interval CO2 emission factors (gCO2/kWh) from RTE.
    Source: https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-TR.zip

    For historical data, please use the following source:
    Source: https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-Consolide.zip

    Returns interval data
    Uses hardcoded French grid average if the download fails.

    Parameters
    ----------
    start_date : str, format 'YYYY-MM-DD'
    end_date : str, format 'YYYY-MM-DD'
    fallback_gco2_per_kwh: float, gCO2eq/kWh

    Returns
    -------
    co2_df : DataFrame with columns [datetime, co2_g_per_kwh, co2_source]
    """
    ZIP_URL = "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-TR.zip"
    XLS_NAME = "eCO2mix_RTE_En-cours-TR.xls"

    try:
        print("  [CO2]  Downloading RTE éCO2mix zip...")
        response = requests.get(ZIP_URL, timeout=30)
        response.raise_for_status()

        # Reading zipfile with French structure, requires additional encoding steps
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open(XLS_NAME) as xls_file:
                raw_bytes = xls_file.read()

        # trailing tab on rows
        cols = pd.read_csv(io.BytesIO(raw_bytes),
                           sep="\t", encoding="latin-1", nrows=0).columns.tolist()
        cols_with_extra = cols + ["_extra"]
        df = pd.read_csv(io.BytesIO(raw_bytes), sep="\t", encoding="latin-1",
                         low_memory=False, names=cols_with_extra, skiprows=1)
        df = df.drop(columns=["_extra"], errors="ignore")

        # build dataframe with co2 and datetime intervals
        co2_col = next((c for c in df.columns if "co2" in str(c).lower()), None)
        date_col = next((c for c in df.columns if str(c).strip().lower() == "date"), None)
        t_col = next((c for c in df.columns if str(c).strip().lower() in ["heures", "heure"]), None)

        if not co2_col or not date_col:
            raise ValueError(f"Could not identify required columns. Found: {list(df.columns)}")

        # Parse datetime where multiple datatypes are present
        if t_col:
            df["datetime"] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[t_col].astype(str),
                format="%Y-%m-%d %H:%M",
                errors="coerce"
            )
        else:
            df["datetime"] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")

        df["co2_g_per_kwh"] = pd.to_numeric(df[co2_col], errors="coerce")
        df = df.dropna(subset=["datetime", "co2_g_per_kwh"])

        # Filter to requested date range
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        df = df[(df["datetime"] >= start) & (df["datetime"] <= end)]

        if df.empty:
            raise ValueError(
                f"No CO2 data found for {start_date} to {end_date} in the zip file. "
                "The real-time file covers approximately the last 1.5 months."
            )

        co2_df = df[["datetime", "co2_g_per_kwh"]].copy()
        co2_df["co2_source"] = "rte_zip"
        co2_df = co2_df.sort_values("datetime").reset_index(drop=True)

        print(f"  [CO2]  {len(co2_df)} interval CO2 factors from RTE zip "
              f"(avg: {co2_df['co2_g_per_kwh'].mean():.1f} gCO2/kWh)")

        return co2_df

    except Exception as e:
        print(f"  [CO2]  RTE zip unavailable ({e}) — using {fallback_gco2_per_kwh} gCO2/kWh")
        # Generate fallback at 15-min intervals
        datetimes = pd.date_range(start=start_date, end=end_date, freq="15min")
        return pd.DataFrame({
            "datetime": datetimes,
            "co2_g_per_kwh": fallback_gco2_per_kwh,
            "co2_source": "fallback"
        })


def load_usage(fs, bucket, files):
    """
    Pulls then reformats model usage data from the S3 bucket
    TODO: need to change to the data sources locations for DGFiP

    Parameters
    ----------
    fs : S3FileSystem
    bucket : str, location of files
    files: list of str, name of each csv file

    Returns
    -------
    usage : DataFrame with columns [Date, Team_ID, Model, Spend, Requests, Successful, Failed,
                                    Total_tokens]
    """
    dfs = []
    for f in files:
        print(f"  Reading: s3://{bucket}/{f}")
        with fs.open(f"{bucket}/{f}") as fh:
            dfs.append(pd.read_csv(fh))
    usage = pd.concat(dfs, ignore_index=True)
    usage.columns = usage.columns.str.strip()
    usage["Date"] = pd.to_datetime(usage["Date"])
    usage["Total_tokens"] = pd.to_numeric(usage["Total Tokens"], errors="coerce").fillna(0)
    usage["Requests"] = pd.to_numeric(usage["Requests"], errors="coerce").fillna(0)
    usage["Spend"] = pd.to_numeric(usage["Spend ($)"], errors="coerce").fillna(0)
    usage["Team_ID"] = usage["Team ID"]
    usage.drop(columns=['Team', 'Spend ($)', 'Total Tokens', 'Team ID'], inplace=True)
    return usage


def load_energy(fs, bucket):
    """
    Pulls then reformats data center energy data from the S3 bucket

    Will pull all csv files with the naming convention
    [3_Digit_Number]_[Letter][Digit]_[Voie]_[YYYYMMDD]

    TODO: need to change to the data sources locations for DGFiP

    Parameters
    ----------
    fs : S3FileSystem
    bucket : str, location of files

    Returns
    -------
    usage : DataFrame with columns [source_file, Datetime, ID_Voie, energy_used]
    """
    pattern = re.compile(r"^\d{3}_[A-Za-z]\d+_Voie\d+_\d{8}\.csv$")
    all_files = fs.ls(bucket)
    energy_files = [f for f in all_files if pattern.match(os.path.basename(f))]

    print(f"  Found {len(energy_files)} energy files matching naming convention")

    dfs = []
    for f in energy_files:
        with fs.open(f) as fh:
            df = pd.read_csv(fh, skiprows=1)
        df["timestamp"] = pd.to_datetime(df["Time (UTC)"])
        df["source_file"] = os.path.basename(f)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    cols_to_check = [c for c in df.columns if c != "source_file"]
    df = df.drop_duplicates(subset=cols_to_check)

    df = df.sort_values(["source_file", "timestamp"])
    df["Datetime"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna()
    df["ID_Voie"] = df["source_file"].str.replace(".csv", "",
                                                  regex=False).str.rsplit("_", n=1).str[0]
    df = df.sort_values(["ID_Voie", "Datetime"]).reset_index(drop=True)
    df['energy_used'] = df.groupby('ID_Voie')['Input Cumulated Energy Total (Wh)'].diff(periods=2)
    df = df[['source_file', 'Datetime', 'ID_Voie', 'energy_used']]

    # Move to final visualization part that Margot recommended
    for id_val, group in df.groupby("ID_Voie"):
        plt.plot(group["Datetime"], group["energy_used"], label=id_val)

    plt.xlabel("Time")
    plt.ylabel("Energy Used (Wh)")
    plt.title("Energy Usage Over Time by ID")
    plt.legend(title="ID")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    df.to_csv("/home/onyxia/work/energy.csv", index=False)

    return df


def fetch_hf_params(model_name, hf_repo):
    """
    Pulls the number of parameters for a model from HuggingFace API, with a fallback
    of desk research on the number of parameters that is hard coded in the MODEL_PARAMS_B_KNOWN
    mapping

    Relies on the HuggingFace safe model names as stored in the MODEL_HF_REPO mapping

    Parameters
    ----------
    model_name : name of model following HuggingFace naming convention
    hf_repo : dict, mapping the model to hardcoded parameter numbers

    Returns
    -------
    p : number of model's parameters
    """
    try:
        info = HfApi().model_info(hf_repo)
        if info.safetensors and info.safetensors.total:
            p = info.safetensors.total / 1e9
            print(f"  [HF]   {model_name}: {p:.3f}B (safetensors)")
            return p
        config = getattr(info, "config", None) or {}
        if "num_parameters" in config:
            p = config["num_parameters"] / 1e9
            print(f"  [HF]   {model_name}: {p:.3f}B (config)")
            return p
        print(f"  [--]   {model_name}: no parameter data on HF ({hf_repo})")
        return None
    except Exception as e:
        print(f"  [ERR]  {model_name}: HF error → {e}")
        return None


def build_params_map(models):
    """
    Calls the fetch_hf_params for each model and organizes the results into different
    data structures depending on if parameter counts are found. Communicates these
    results to the user via terminal output.

    Relies on the HuggingFace safe model names as stored in the MODEL_HF_REPO mapping
    and desk research in MODEL_PARAMS_B_KNOWN if the parameters are not in HuggingFace

    Parameters
    ----------
    models : list of str, which represent the DGFiP deployed models following internal naming
             conventions

    Returns
    -------
    params_map : dict representing the models with known parameter counts
    excluded : list of models with no known parameter counts
    """
    params_map = {}
    excluded = []
    for model in sorted(models):
        hf_repo = MODEL_HF_REPO.get(model)
        if hf_repo:
            val = fetch_hf_params(model, hf_repo)
            if val is not None:
                params_map[model] = val
                continue
        if model in MODEL_PARAMS_B_KNOWN:
            src = "known" if not hf_repo else "known (HF has no safetensors)"
            print(f"  [OK]   {model}: {MODEL_PARAMS_B_KNOWN[model]}B ({src})")
            params_map[model] = MODEL_PARAMS_B_KNOWN[model]
            continue
        reason = "internal model" if not hf_repo else "not found on HF"
        print(f"  [SKIP] {model}: excluded ({reason})")
        excluded.append(model)
    return params_map, excluded


def estimate_idle_energy(energy_df, idle_quantile=0.05, idle_window_days=7):
    """
    Estimate idle energy consumption per voie.

    TODO: enable this function when any of the following become available:
        - Model run start/end timestamps to mask active periods before computing baseline
        - An explicit idle/active state flag per Voie in the energy data
        - Sub-hourly energy data with known idle periods (e.g. overnight with no requests)

    Current limitation: servers run continuously at consistent load, so the lower
    quantile of energy_used is not meaningfully different from the median — leading
    to ~99% of energy being classified as idle.

    Parameters
    ----------
    energy_df : energy df with [Datetime, ID_Voie, energy_used] columns
    idle_quantile : lower quantile of energy_used to use as idle estimate
    idle_window_days : number of days to use for rolling idle baseline

    Returns
    -------
    None until data quality allows reliable idle detection
    """
    print("  [idle] Idle estimation disabled — model run timestamps required for baseline.")
    print("  [idle] TODO: provide model run start/end times to isolate idle from active use.")
    return None


def compute_peak_analysis(energy_df):
    """
    Compute monthly peak energy and top 5 quarterly peaks per Voie.

    Parameters
    ----------
    energy_df : energy dataframe with [Datetime, ID_Voie, energy_used] columns

    Returns
    -------
    monthly_peaks : df with [ID_Voie, month, peak_energy_wh]
    top5_peaks    : df with [ID_Voie, Datetime, peak_energy_wh for top 5 peaks]
    """
    energy_df = energy_df.copy()
    energy_df["month"] = energy_df["Datetime"].dt.to_period("M")

    monthly_peaks = (
        energy_df.groupby(["ID_Voie", "month"])["energy_used"]
        .max()
        .reset_index()
        .rename(columns={"energy_used": "peak_energy_wh"})
    )

    top5_peaks = (
        energy_df.nlargest(5, "energy_used")
        [["ID_Voie", "Datetime", "energy_used"]]
        .rename(columns={"energy_used": "peak_energy_wh"})
        .reset_index(drop=True)
    )

    return monthly_peaks, top5_peaks


def aggregate_energy_daily(energy_df):
    """
    Filter to relevant Voies and aggregate energy readings to daily level.

    Parameters
    ----------
    energy_df : energy df with [Datetime, ID_Voie, energy_used] columns

    Returns
    -------
    energy_daily : df with [date, total_energy_wh]
    energy_df : filtered and date-annotated energy df
    """
    energy_df = energy_df.copy()
    energy_df["date"] = energy_df["Datetime"].dt.tz_localize(None).dt.normalize()
    energy_df = energy_df[energy_df["ID_Voie"].isin(["101_J37_Voie1"])]

    energy_daily = (
        energy_df
        .groupby("date")
        .agg(total_energy_wh=("energy_used", "sum"))
        .reset_index()
    )

    return energy_daily, energy_df


def apply_idle_subtraction(energy_daily, energy_df):
    """
    Subtract idle energy from total to estimate active energy consumption.
    Currently disabled pending model run timestamp availability.

    Parameters
    ----------
    energy_daily : daily aggregated energy df
    energy_df    : interval-level energy df

    Returns
    -------
    energy_daily : original df with [idle_energy_wh, active_energy_wh] columns added
    """
    idle_df = estimate_idle_energy(energy_df)
    if idle_df is not None:
        idle_daily = idle_df.groupby("date").agg(idle_energy_wh=("idle_energy_wh",
                                                                 "sum")).reset_index()
        energy_daily = energy_daily.merge(idle_daily, on="date", how="left")
        energy_daily["idle_energy_wh"] = energy_daily["idle_energy_wh"].fillna(0)
        energy_daily["active_energy_wh"] = (
            energy_daily["total_energy_wh"] - energy_daily["idle_energy_wh"]
        ).clip(lower=0)
        print("  [idle] Idle energy subtracted from total — using active_energy_wh for estimate.")
    else:
        energy_daily["idle_energy_wh"] = None
        energy_daily["active_energy_wh"] = energy_daily["total_energy_wh"]
        print("  [idle] Using total_energy_wh as proxy — includes idle consumption (upper bound).")

    return energy_daily


def apply_co2_factors(energy_daily, energy_df):
    """
    Fetch 15-minute CO2 factors from RTE and attach energy-weighted daily
    averages to the energy_daily DataFrame.

    Parameters
    ----------
    energy_daily : daily aggregated energy df
    energy_df    : interval-level energy df with Datetime column

    Returns
    -------
    energy_daily : original df with [co2_g_per_kwh, co2_source, co2_kg columns] added
    """
    start = (energy_daily["date"].min() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = energy_df["Datetime"].dt.tz_localize(None).max().strftime("%Y-%m-%d")
    co2_df = fetch_rte_co2_factor(start, end)

    # Match each 5-min energy reading to the last overlapping 15-min CO2 interval
    energy_df_co2 = energy_df.copy()
    energy_df_co2["datetime_naive"] = energy_df_co2["Datetime"].dt.tz_localize(None)
    energy_df_co2 = pd.merge_asof(
        energy_df_co2.sort_values("datetime_naive"),
        co2_df.sort_values("datetime"),
        left_on="datetime_naive",
        right_on="datetime",
        direction="backward"
    )

    # Aggregate to daily energy-weighted CO2 factor
    co2_daily = (
        energy_df_co2.groupby(energy_df_co2["datetime_naive"].dt.normalize())
        .apply(lambda g: pd.Series({
            "co2_g_per_kwh": (g["co2_g_per_kwh"] * g["energy_used"]).sum() / g["energy_used"].sum(),
            "co2_source": g["co2_source"].iloc[0]
        }))
        .reset_index()
        .rename(columns={"datetime_naive": "date"})
    )

    energy_daily = energy_daily.merge(co2_daily, on="date", how="left")
    energy_daily["co2_kg"] = (
        (energy_daily["active_energy_wh"] / 1000) * (energy_daily["co2_g_per_kwh"] / 1000)
    )

    return energy_daily


def build_energy_daily(energy_df):
    """
    Produce  daily energy df with idle subtraction and CO2 factors attached.

    Parameters
    ----------
    energy_df : raw interval-level energy df

    Returns
    -------
    energy_daily : df with [date, total_energy_wh, idle_energy_wh,
                   active_energy_wh, co2_g_per_kwh, co2_source, co2_kg]
    """
    energy_daily, energy_df_filtered = aggregate_energy_daily(energy_df)
    energy_daily = apply_idle_subtraction(energy_daily, energy_df_filtered)
    energy_daily = apply_co2_factors(energy_daily, energy_df_filtered)

    return energy_daily


def merge_usage_and_energy(usage_df, energy_daily):
    """
    Join daily usage records with daily energy data on date.

    Parameters
    ----------
    usage_df     : usage df with [Date, Model, Total_tokens] columns
    energy_daily : daily energy df from build_energy_daily

    Returns
    -------
    merged : usage_df joined with energy_daily with [wh_per_1000_tokens,
             co2_g_per_1000_tokens] columns added
    """
    usage_df = usage_df.copy()
    usage_df["date"] = pd.to_datetime(usage_df["Date"]).dt.normalize()

    merged = usage_df.merge(energy_daily, on="date", how="left")
    merged["wh_per_1000_tokens"] = (merged["active_energy_wh"] / merged["Total_tokens"]) * 1000
    merged["co2_g_per_1000_tokens"] = (merged["co2_kg"] * 1000 / merged["Total_tokens"]) * 1000

    return merged


def estimate_wh_per_1000_tokens(merged_df):
    """
    Estimate Wh per 1000 tokens for the clean anchor model window.
    TODO: expand with access to more granular data

    Parameters
    ----------
    merged_df : merged usage and energy DataFrame from merge_usage_and_energy

    Returns
    -------
    overall_estimate : float, Wh per 1000 tokens for the anchor model
    clean            : df filtered to the clean estimation window
    """
    # TODO: remove this filter as more clean single-model days are identified
    clean_dates = pd.to_datetime(["2026-02-01", "2026-02-02"])
    clean_model = "gte-Qwen2-1-5B-instruct"

    clean = merged_df[
        (merged_df["date"].isin(clean_dates)) &
        (merged_df["Model"] == clean_model)
    ]

    overall_estimate = (clean["active_energy_wh"].sum() / clean["Total_tokens"].sum()) * 1000

    return overall_estimate, clean


def main():
    df_usage = load_usage(fs, S3_BUCKET, USAGE_FILES)
    df_energy = load_energy(fs, S3_BUCKET)

    print("\n[1] Building daily energy and CO2 factors...")
    energy_daily = build_energy_daily(df_energy)

    print("\n[2] Merging usage and energy data...")
    merged_df = merge_usage_and_energy(df_usage, energy_daily)

    print("\n[3] Estimating Wh per 1000 tokens...")
    wh_per_1000, clean = estimate_wh_per_1000_tokens(merged_df)

    total_tokens = clean["Total_tokens"].sum()
    total_energy = clean["active_energy_wh"].sum()
    total_co2_kg = clean["co2_kg"].sum()
    overall_co2_per_1000 = (total_co2_kg * 1000 / total_tokens) * 1000

    print(clean[["date", "Model", "Total_tokens", "total_energy_wh", "active_energy_wh",
                 "wh_per_1000_tokens", "co2_kg", "co2_g_per_1000_tokens"]])
    print("\n  Model: gte-Qwen2-1-5B-instruct")
    print(f"  Total Tokens:               {total_tokens:,}")
    print(f"  Total Energy (active):      {total_energy:,.1f} Wh")
    print(f"  Estimate:                   {wh_per_1000:.4f} Wh / 1000 tokens")
    print(f"  Total CO2:                  {total_co2_kg:.4f} kgCO2e")
    print(f"  CO2 Intensity:              {overall_co2_per_1000:.4f} gCO2e / 1000 tokens")

    print("\n[4] Peak analysis...")
    monthly_peaks, top5_peaks = compute_peak_analysis(df_energy)
    print("\n  Monthly peaks per Voie:")
    print(monthly_peaks.to_string(index=False))
    print("\n  Top 5 quarterly peaks:")
    print(top5_peaks.to_string(index=False))

    print("\n[5] Looking up model parameters (HuggingFace + fallback)...")
    models = sorted(df_usage["Model"].unique())
    params_map, excluded = build_params_map(models)
    print(f"  Models with data: {len(params_map)} | Excluded: {len(excluded)}")

    return merged_df, wh_per_1000, energy_daily, monthly_peaks, top5_peaks, params_map


merged_df, wh_per_1000, energy_daily, monthly_peaks, top5_peaks, params_map = main()
