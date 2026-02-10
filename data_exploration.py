import os
import s3fs
import glob
import pandas as pd
from datetime import datetime

S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

BUCKET = "projet-datalab-depp-dgfip/"
FILE_KEY_S3_energy = "Energy_Data_Clean.csv"
FILE_PATH_S3_energy = BUCKET + "/" + FILE_KEY_S3_energy

with fs.open(FILE_KEY_S3_energy, mode="rb") as file_in:
    df_energy = pd.read_csv(file_in)

print(df_energy.columns)
print(df_energy.shape)

BUCKET = "projet-datalab-depp-dgfip/"
FILE_KEY_S3_model = "diane_usage_daily_with_models_2026-01-28 (1).csv"
FILE_PATH_S3_model = BUCKET + "/" + FILE_KEY_S3_model

with fs.open(FILE_KEY_S3_model, mode="rb") as file_in:
    df_model = pd.read_csv(file_in)

folder_path = "/home/onyxia/work/energy_score_data_02_26/*.csv"

aie_models = []

for file in glob.glob(folder_path):
    m = pd.read_csv(file)

    m["type"] = os.path.splitext(os.path.basename(file))[0]

    aie_models.append(m)

aie_models = pd.concat(aie_models, ignore_index=True)

aie_models[['Company', 'model_name']] = aie_models['model'].str.split('/', expand=True)

df_model.loc[df_model['Model'] == "Llama-3-3-70B-128k", 'Task'] = 'Text_Generation'
URL = 'https://huggingface.co/meta-llama/Meta-Llama-3-70B'
df_model.loc[df_model['Model'] == 'Llama-3-3-70B-128k', 'HF_URL'] = URL

df_model.loc[df_model['Model'] == "gptoss120b", 'Task'] = 'Reasoning'
URL = 'https://huggingface.co/openai/gpt-oss-120b'
df_model.loc[df_model['Model'] == 'gptoss120b', 'HF_URL'] = URL

df_model.loc[df_model['Model'] == "gptoss20b", 'Task'] = 'Reasoning'
URL = "https://huggingface.co/openai/gpt-oss-20b"
df_model.loc[df_model['Model'] == 'gptoss20b', 'HF_URL'] = URL

df_model.loc[df_model['Model'] == '/model/deepdml-faster-whisper-large-v3-turbo-ct2', 'Task'] = 'Automatic_Speech_Recognition'
URL = 'https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2'
df_model.loc[df_model['Model'] == '/model/deepdml-faster-whisper-large-v3-turbo-ct2', 'HF_URL'] = URL

df_model.loc[df_model['Model'] == 'Qwen2.5-Coder-32B-Instruct-fp8-W8A16', 'Task'] = 'Text_Generation'
URL = 'https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct'
df_model.loc[df_model['Model'] == 'Qwen2.5-Coder-32B-Instruct-fp8-W8A16', 'HF_URL'] = URL

df_model.loc[df_model['Model'] == 'gte-Qwen2-1-5B-instruct', 'Task'] = 'Sentence_Similarity'
URL = 'https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct'
df_model.loc[df_model['Model'] == 'gte-Qwen2-1-5B-instruct', 'HF_URL'] = URL

df_model.loc[df_model['Model'] == 'Mistral-Small-24B-Instruct-2501-FP8-dynamic', 'Task'] = 'Reasoning'
URL = 'https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501'
df_model.loc[df_model['Model'] == 'Mistral-Small-24B-Instruct-2501-FP8-dynamic', 'HF_URL'] = URL

df_model.loc[df_model['Model'] == 'dgfip-e5-large', 'Task'] = 'Sentence_Similarity'
URL = 'https://huggingface.co/intfloat/e5-large'
df_model.loc[df_model['Model'] == 'dgfip-e5-large', 'HF_URL'] = URL

df_model['AIE_name'] = df_model['HF_URL'].str.split("/").str[-1]

model_data = df_model.merge(aie_models, left_on='AIE_name', right_on='model_name', how='left')

model_data['Date'] = pd.to_datetime(model_data['Date'], errors='coerce')

has_current_month_data = (
    (model_data['Date'].dt.year == datetime.now().year) &
    (model_data['Date'].dt.month == datetime.now().month)
).any()

if not has_current_month_data:
    print(
        "The energy benchmark data appears to be stale.\n\n"
        "No records found for the current month.\n\n"
        "Please go to https://huggingface.co/spaces/AIEnergyScore/Leaderboard,"
        "download the latest data, and upload it to this tool using \nthe following naming convention:.\n"
        '\tenergy_score_data_MM_YY'
    )
