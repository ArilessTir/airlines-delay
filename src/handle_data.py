from sklearn.model_selection import train_test_split
import pandas as pd

def handle_data(path, target):

    data = pd.read_csv(path)

    X = data.drop(target,axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                        test_size=0.2,
                                                        random_state=400)
    return X_train, X_test, y_train, y_test