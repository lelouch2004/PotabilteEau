from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

def preprocessing(data):
    num_features = [
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ]
    
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy='mean')),
        ("scaler", MinMaxScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_features)
    ])
    
    # Appliquer le prétraitement sur tout le DataFrame
    X_processed = preprocessor.fit_transform(data)
    
    # Convertir en DataFrame
    X_processed = pd.DataFrame(X_processed, columns=num_features)

    return X_processed
