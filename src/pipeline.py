from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocessing import preprocessor


model = Pipeline(
    steps=[
        ('preprocessing', preprocessor),
        ('model',LogisticRegression(max_iter=500,random_state=400))
    ]
)