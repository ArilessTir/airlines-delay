from sklearn.metrics import (precision_score,
                            accuracy_score,
                            recall_score,
                            classification_report)
from pipeline import prepross, model, pipeline 
import Config
import mlflow
from handle_data import handle_data
from decouple import config

TRACKING_URI_DNS = config('TRACKING_URI_DNS')

mlflow.set_tracking_uri(f"http://{TRACKING_URI_DNS}:5000")
mlflow.set_experiment("airlines-experiment")


X_train, X_test, y_train, y_test = handle_data('../Data/Airlines.csv',
                                                Config.TARGET)



with mlflow.start_run():

    mlflow.set_tag("developer", "Ariless")
    mlflow.log_param("data-path", "./Data/Airlines.csv")

    pipeline = pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    ## Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('precision',precision)
    mlflow.log_metric('recall',recall)
    mlflow.log_metric('accuracy',accuracy)

    ## Params

    mlflow.log_params(prepross)
    mlflow.log_params(model['model_params'])

    ## Model
    mlflow.sklearn.log_model(pipeline,'artifact')