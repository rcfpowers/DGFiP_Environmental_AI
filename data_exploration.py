import os
import s3fs
import pandas as pd

S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

BUCKET = "projet-datalab-depp-dgfip/"
FILE_KEY_S3 = "diane_usage_daily_with_models_2026-01-28 (1).csv"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3

with fs.open(FILE_PATH_S3, mode="rb") as file_in:
    df = pd.read_csv(file_in, sep=",")



df.loc[df['Model'] == "Llama-3-3-70B-128k", 'Task'] = 'Text_Generation'
URL = 'https://huggingface.co/meta-llama/Meta-Llama-3-70B'
df.loc[df['Model'] == 'Llama-3-3-70B-128k', 'HF_URL'] = URL

df.loc[df['Model'] == "gptoss120b", 'Task'] = 'Reasoning'
URL = 'https://huggingface.co/openai/gpt-oss-120b'
df.loc[df['Model'] == 'gptoss120b', 'HF_URL'] = URL

df.loc[df['Model'] == "gptoss20b", 'Task'] = 'Reasoning'
URL = "https://huggingface.co/openai/gpt-oss-20b"
df.loc[df['Model'] == 'gptoss20b', 'HF_URL'] = URL

df.loc[df['Model'] == '/model/deepdml-faster-whisper-large-v3-turbo-ct2', 'Task'] = 'Automatic_Speech_Recognition'
URL = 'https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2'
df.loc[df['Model'] == '/model/deepdml-faster-whisper-large-v3-turbo-ct2', 'HF_URL'] = URL

df.loc[df['Model'] == 'Qwen2.5-Coder-32B-Instruct-fp8-W8A16', 'Task'] = 'Text_Generation'
URL = 'https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct'
df.loc[df['Model'] == 'Qwen2.5-Coder-32B-Instruct-fp8-W8A16', 'HF_URL'] = URL

df.loc[df['Model'] == 'gte-Qwen2-1-5B-instruct', 'Task'] = 'Sentence_Similarity'
URL = 'https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct'
df.loc[df['Model'] == 'gte-Qwen2-1-5B-instruct', 'HF_URL'] = URL

df.loc[df['Model'] == 'Mistral-Small-24B-Instruct-2501-FP8-dynamic', 'Task'] = 'Reasoning'
URL = 'https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501'
df.loc[df['Model'] == 'Mistral-Small-24B-Instruct-2501-FP8-dynamic', 'HF_URL'] = URL

df.loc[df['Model'] == 'dgfip-e5-large', 'Task'] = 'Sentence_Similarity'
URL = 'https://huggingface.co/intfloat/e5-large'
df.loc[df['Model'] == 'dgfip-e5-large', 'HF_URL'] = URL

print(df['HF_URL'].unique())
