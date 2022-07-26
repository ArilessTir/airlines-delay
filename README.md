## Airlines Delay Project

The goal of this project is to build ml pipline and track experimentation localy

### Run this project

#### Install dependencies

```
pip install -r requirements.txt
```

### Run MlFLow ui

```
cd src ; mlflow server --backend-store-uri sqlite:///mlflow.db  --default-artifact-root ./artifact
```

### Run training script

```
python train.py
```
