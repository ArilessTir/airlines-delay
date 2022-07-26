from sklearn.metrics import (precision_score,
                            accuracy_score,
                            recall_score,
                            classification_report)
from pipeline import model
import Config
import mlflow
from handle_data import handle_data

#mlflow.set_tracking_uri("sqlite:///mlflow.db")
#mlflow.set_experiment("airlines-experiment")


X_train, X_test, y_train, y_test = handle_data('../Data/Airlines.csv',
                                                Config.TARGET)



#with mlflow.start_run():

#    mlflow.set_tag("developer", "Ariless")
#    mlflow.log_param("data-path", "./Data/Airlines.csv")

model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)

## Metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(precision)
print(recall)
print(accuracy)

#    mlflow.log_metrics([precision,recall,accuracy])

    ## Model
