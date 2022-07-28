## Airlines Delay Project

The goal of this project is to build ml pipeline and track experimentation on Aws

### Run this project

#### Install dependencies

```
pip install -r requirements.txt
```

### Run MlFLow remotly

```
cd src ; mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/DB_NAME --default-artifact-root s3://S3_BUCKET_NAME
```

[link for remote tracking setup](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md)

Do not forget to install the aws cli localy, [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) in order to fill your credential 

### Run training script

```
python train.py
```
