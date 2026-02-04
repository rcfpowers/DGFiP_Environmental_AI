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

print(df.shape)