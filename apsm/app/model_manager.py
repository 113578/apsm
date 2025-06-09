import pandas as pd
import pickle as pkl
from apsm.utils import (
    get_model_path,
    get_transformer_path
)
from apsm.app.data_preprocessing import (
    preprocess_time_series,
    extract_time_series_features,
    inverse_preprocess_time_series
)


# Глобальный менеджер модели для хранения активной ML-модели.
class ModelManager:
    model = None
    data_type: str
    transformers = None
    
    @staticmethod
    def transform_data(data, ticker, is_train=True):    
        df = pd.DataFrame({'value': data})
        df['Date'] = pd.date_range(start='2022-01-01', periods=len(data), freq='D')
      
        features = extract_time_series_features(df[['Date', 'value']])
       
        features['ticker'] = ticker
        
        data, ModelManager.transformers = preprocess_time_series(df=features, target='target', is_train=is_train, transformers=ModelManager.transformers)
        return data
    
    @staticmethod
    def load_transformers():
        transformers = get_transformer_path(data_type=ModelManager.data_type)
        
        with open(transformers, 'rb') as file:
            ModelManager.transformers = pkl.load(file)
        