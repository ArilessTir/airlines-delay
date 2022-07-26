from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
import Config

preprocessor = ColumnTransformer(
    [
        ('OHE', OneHotEncoder(handle_unknown="ignore", sparse=False),Config.OHE),
        ('drop_feature','drop',Config.REMOVE),
        ('unchanged','passthrough', Config.KEEP)
    ]
)