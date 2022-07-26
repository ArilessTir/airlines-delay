from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocessing import preprocessor

model = {
    'name':LogisticRegression,
    'model_params':{    
        'max_iter':200,
        'random_state':400,
        'C':10
    }
}

pipeline = Pipeline(
    steps=[
        ('preprocessing', preprocessor),
        ('model',model['name'](**model['model_params']))
    ]
)

prepross_list = pipeline.named_steps['preprocessing'].transformers
prepross = {}

for i in range(len(prepross_list)):
    prepross[prepross_list[i][0]] = prepross_list[i]

