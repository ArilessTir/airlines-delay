from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from preprocessing import preprocessor

model = {
    'name':RidgeClassifier,
    'model_params':{    
        'max_iter':200,
        'random_state':400,
        'alpha':1
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

