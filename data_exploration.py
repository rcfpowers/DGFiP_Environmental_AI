import subprocess
import sys
import os
import s3fs
import pandas as pd
import matplotlib.pyplot as plt
import re


def install():
    pkgs = ["pandas", "numpy", "matplotlib", "huggingface_hub", "boto3", "s3fs"]
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + pkgs)


install()

from huggingface_hub import HfApi

S3_BUCKET = "projet-datalab-depp-dgfip"
USAGE_FILES = ["diane_usage_daily_with_models_fevrier.csv",
               "diane_usage_daily_with_models_2026-01-28 (1).csv"]
OUTPUT_CSV = None

"""
Comment out when publishing and add descriptions of how to establish connection to underlying data
"""

os.environ["AWS_ACCESS_KEY_ID"] = '6JUTTZCSVX8TUV6TX10N'
os.environ["AWS_SECRET_ACCESS_KEY"] = '4Y0LaYdkGgOXtfqP6FlcrCt3xBr+hs4YtJhKEYvy'
os.environ["AWS_SESSION_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiI2SlVUVFpDU1ZYOFRVVjZUWDEwTiIsImFjciI6IjAiLCJhbGxvd2VkLW9yaWdpbnMiOlsiKiJdLCJhdWQiOlsibWluaW8iLCJhY2NvdW50Il0sImF1dGhfdGltZSI6MTc3MzQ4NjQ4MSwiYXpwIjoib255eGlhLW1pbmlvIiwiZW1haWwiOiJyb2JlcnQucG93ZXJzQGVuc2FlLmZyIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImV4cCI6MTc3NDY5NzM0MywiZmFtaWx5X25hbWUiOiJQT1dFUlMiLCJnaXZlbl9uYW1lIjoiUm9iZXJ0IiwiZ3JvdXBzIjpbImRhdGFsYWItZGVwcC1kZ2ZpcCJdLCJpYXQiOjE3NzM0ODc3NDEsImlzcyI6Imh0dHBzOi8vYXV0aC5ncm91cGUtZ2VuZXMuZnIvcmVhbG1zL2dlbmVzIiwianRpIjoiMWY3NDA4OWQtZDg4Yi00NzMxLThhNTUtNzQzZWEyYzQ5Zjg4IiwibmFtZSI6IlJvYmVydCBQT1dFUlMiLCJwb2xpY3kiOiJzdHNvbmx5IiwicHJlZmVycmVkX3VzZXJuYW1lIjoicnBvd2Vycy1lbnNhZSIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsImRlZmF1bHQtcm9sZXMtZ2VuZXMiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwiLCJzaWQiOiJmNmU1ZGRmMi1kYjE3LTRjZDEtOWE2NS0wOGQyZjc4ZjM3N2QiLCJzdWIiOiI2NzYzNDFlZC1mM2FlLTQzZDgtOTMzMS0zMTZjMzA4NDRlNjgiLCJ0eXAiOiJCZWFyZXIifQ.PjLM_zxZSLNu4kbEZ9VeV4R8WJsO3EOn2W3hn6y0Ylja8b_-qph7pD-I4jvJ4csac51GyGAEeuvbffPAdyVHLQ'
os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'

fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio-simple.lab.groupe-genes.fr'},
    key = os.environ["AWS_ACCESS_KEY_ID"],
    secret = os.environ["AWS_SECRET_ACCESS_KEY"],
    token = os.environ["AWS_SESSION_TOKEN"])

"""
Is it possible to make MODEL_HF_REPO and MODEL_PARAMS_B_KNOWN dynamic?
"""

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

MODEL_PARAMS_B_KNOWN = {
    # Retrieved from HuggingFace safetensors
    "gte-Qwen2-1-5B-instruct": 1.776,
    "Qwen2.5-Coder-32B-Instruct-fp8-W8A16": 32.764,
    "Llama-3-3-70B-128k": 70.0,
    "Mistral-Small-24B-Instruct-2501-FP8-dynamic": 24.0,
    "/model/deepdml-faster-whisper-large-v3-turbo-ct2": 0.809,
    "dgfip-e5-large": 0.335,
    # Desk Research
    "gptoss20b": 20.0,
    "gptoss120b": 120.0,
    "qwen3vl32binstruct":   32.8,
}

"""
Make function comments
"""
def load_usage(fs, bucket, files):
    dfs = []
    for f in files:
        print(f"  Reading: s3://{bucket}/{f}")
        with fs.open(f"{bucket}/{f}") as fh:
            dfs.append(pd.read_csv(fh))
    usage = pd.concat(dfs, ignore_index=True)
    usage.columns = usage.columns.str.strip()
    usage["Date"] = pd.to_datetime(usage["Date"])
    usage["Total Tokens"] = pd.to_numeric(usage["Total Tokens"], errors="coerce").fillna(0)
    usage["Requests"] = pd.to_numeric(usage["Requests"], errors="coerce").fillna(0)
    usage["Spend ($)"] = pd.to_numeric(usage["Spend ($)"], errors="coerce").fillna(0)
    return usage

"""
Make function comments
"""
def load_energy(fs, bucket):
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
    df["ID_Voie"] = df["source_file"].str.replace(".csv", "", regex=False).str.rsplit("_", n=1).str[0]
    df = df.sort_values(["ID_Voie", "Datetime"]).reset_index(drop=True)

    df['energy_used'] = df.groupby('ID_Voie')['Input Cumulated Energy Total (Wh)'].diff(periods=2)

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


def build_params_map(models, use_hf):
    params_map = {}
    excluded = []
    for model in sorted(models):
        hf_repo = MODEL_HF_REPO.get(model)
        if use_hf and hf_repo:
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


def estimate_wh_per_1000_tokens(energy_df, usage_df):
    """
    Estimate Wh per 1000 tokens for all models on all days.
    Currently filtered to clean single-model days for gte-Qwen2-1-5B-instruct.
    TODO: expand as data quality improves.
    """
    # --- Aggregate energy to daily level ---
    # Strip timezone before normalising to avoid merge type mismatch
    energy_df["date"] = energy_df["Datetime"].dt.tz_localize(None).dt.normalize()
    energy_df = energy_df[energy_df["ID_Voie"].isin(["101_J37_Voie1"])]
    energy_daily = (
        energy_df
        .groupby("date")
        .agg(total_energy_wh=("energy_used", "sum"))
        .reset_index()
    )

    # --- Normalise date column in usage data ---
    usage_df["date"] = pd.to_datetime(usage_df["Date"]).dt.normalize()

    # --- Join on date ---
    merged = usage_df.merge(energy_daily, on="date", how="left")

    # --- Compute Wh per 1000 tokens for all rows ---
    merged["wh_per_1000_tokens"] = (merged["total_energy_wh"] / merged["Total Tokens"]) * 1000

    # --- Filter to clean estimation window: single-model days for gte-Qwen2-1-5B-instruct ---
    # TODO: remove this filter as more clean single-model days are identified
    clean_dates = pd.to_datetime(["2026-02-01", "2026-02-02"])
    clean_model = "gte-Qwen2-1-5B-instruct"
    clean = merged[
        (merged["date"].isin(clean_dates)) &
        (merged["Model"] == clean_model)
    ]

    # --- Summary estimate ---
    total_tokens = clean["Total Tokens"].sum()
    total_energy = clean["total_energy_wh"].sum()
    overall_estimate = (total_energy / total_tokens) * 1000

    print(clean[["date", "Model", "Total Tokens", "total_energy_wh", "wh_per_1000_tokens"]])
    print(f"\n  Model: {clean_model}")
    print(f"  Total Tokens: {total_tokens:,}")
    print(f"  Total Energy: {total_energy:,.1f} Wh")
    print(f"  Estimate: {overall_estimate:.4f} Wh / 1000 tokens")

    return merged, overall_estimate


df_usage = load_usage(fs, S3_BUCKET, USAGE_FILES)
df_energy = load_energy(fs, S3_BUCKET)

result_df, wh_per_1000 = estimate_wh_per_1000_tokens(df_energy, df_usage)

use_hf = True

print(f"\n[2/6] Looking up model parameters "
      f"{'(HuggingFace + fallback)' if use_hf else '(fallback only)'}...")
models = sorted(df_usage["Model"].unique())
params_map, excluded = build_params_map(models, use_hf)
print(f"  Models with data : {len(params_map)} | Excluded : {len(excluded)}")


ai_energy = pd.read_csv("https://raw.githubusercontent.com/Nidhal-Jegham/HowHungryisAIDashboard/main/output/artificialanalysis_environmental.csv")

folder_path = "/home/onyxia/work/energy_score_data_02_26/*.csv"
"""
aie_models = []

for file in glob.glob(folder_path):
    m = pd.read_csv(file)

    m["type"] = os.path.splitext(os.path.basename(file))[0]

    aie_models.append(m)

aie_models = pd.concat(aie_models, ignore_index=True)

aie_models[['Company', 'model_name']] = aie_models['model'].str.split('/', expand=True)

aie_models['variation'] = aie_models['model_name'].str.extract(r'\b(low|medium|high)\b')
aie_models['variation'] = aie_models['variation'].fillna('medium')
aie_models['base_model'] = aie_models['model_name'].str.replace(r'\s+(low|medium|high)\b', '', regex=True)

metrics = ['total_gpu_energy', 'energy_score', 'test date']

pivot_df = aie_models.pivot_table(
    index='base_model',
    columns='variation',
    values=metrics,
    aggfunc='first'
)

pivot_df.columns = [
    f"{metric}" if var == "medium" else f"{metric}_{var}"
    for metric, var in pivot_df.columns
]

pivot_df = pivot_df.reset_index()

aie_models = aie_models.drop(columns=metrics + ['variation']).drop_duplicates('base_model') \
    .merge(pivot_df, on='base_model', how='left')
aie_models['model_name'] = aie_models['model_name'].str.replace(r'\smedium$', '', regex=True)

"""

df_usage.loc[df_usage['Model'] == "Llama-3-3-70B-128k", 'Task'] = 'Text_Generation'
URL = 'https://huggingface.co/meta-llama/Meta-Llama-3-70B'
df_usage.loc[df_usage['Model'] == 'Llama-3-3-70B-128k', 'HF_URL'] = URL

df_usage.loc[df_usage['Model'] == "gptoss120b", 'Task'] = 'Reasoning'
URL = 'https://huggingface.co/openai/gpt-oss-120b'
df_usage.loc[df_usage['Model'] == 'gptoss120b', 'HF_URL'] = URL

df_usage.loc[df_usage['Model'] == "gptoss20b", 'Task'] = 'Reasoning'
URL = "https://huggingface.co/openai/gpt-oss-20b"
df_usage.loc[df_usage['Model'] == 'gptoss20b', 'HF_URL'] = URL

df_usage.loc[df_usage['Model'] == '/model/deepdml-faster-whisper-large-v3-turbo-ct2', 'Task'] = 'Automatic_Speech_Recognition'
URL = 'https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2'
df_usage.loc[df_usage['Model'] == '/model/deepdml-faster-whisper-large-v3-turbo-ct2', 'HF_URL'] = URL

df_usage.loc[df_usage['Model'] == 'Qwen2.5-Coder-32B-Instruct-fp8-W8A16', 'Task'] = 'Text_Generation'
URL = 'https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct'
df_usage.loc[df_usage['Model'] == 'Qwen2.5-Coder-32B-Instruct-fp8-W8A16', 'HF_URL'] = URL

df_usage.loc[df_usage['Model'] == 'gte-Qwen2-1-5B-instruct', 'Task'] = 'Sentence_Similarity'
URL = 'https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct'
df_usage.loc[df_usage['Model'] == 'gte-Qwen2-1-5B-instruct', 'HF_URL'] = URL

df_usage.loc[df_usage['Model'] == 'Mistral-Small-24B-Instruct-2501-FP8-dynamic', 'Task'] = 'Reasoning'
URL = 'https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501'
df_usage.loc[df_usage['Model'] == 'Mistral-Small-24B-Instruct-2501-FP8-dynamic', 'HF_URL'] = URL

df_usage.loc[df_usage['Model'] == 'dgfip-e5-large', 'Task'] = 'Sentence_Similarity'
URL = 'https://huggingface.co/intfloat/e5-large'
df_usage.loc[df_usage['Model'] == 'dgfip-e5-large', 'HF_URL'] = URL


df_usage['AIE_name'] = df_usage['HF_URL'].str.split("/").str[-1]

df_usage.loc[df_usage['AIE_name'] == 'e5-large', 'AIE_name'] = "e5-large-v2"


#model_data = df_usage.merge(aie_models, left_on='AIE_name', right_on='model_name', how='left')


print(f"\n[2/6] Looking up model parameters "
      f"{'(HuggingFace + fallback)' if use_hf else '(fallback only)'}...")
models = sorted(df_usage["Model"].unique())

print(models)

print(df_usage.head())
