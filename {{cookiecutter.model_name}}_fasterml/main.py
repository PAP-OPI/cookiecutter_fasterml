import pandas as pd
from src.classes import Trainable

class {{cookiecutter.model_name}}(Trainable):
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        preprocessed_data = df.copy()

        # AUI REALIZA TODO EL PREPROCESAMIENTO DE TUS DATOS

        return preprocessed_data

    def train(self, df: pd.DataFrame) -> any:

        self.model = #Aqui mandas llamar tu modelo (sklearn, statsmodels, etc.)
        self.model.fit(df['X'], df['y'])