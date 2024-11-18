from __future__ import annotations

import logging
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Trainable(ABC):
    model: Any = field(init=False)
    metrics: list[str] = field(default_factory=list)

    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Function to apply the preprocessing needed in the dataset to be ingested to the model

        Args:
            df (pd.DataFrame): Dataset to preprocess

        Returns:
            pd.DataFrame: Dataset clean
        """
        pass

    @abstractmethod
    def train(self, df: pd.DataFrame) -> any:
        """Function to train the model with a given dataset

        Args:
            df (pd.DataFrame): Dataset to be train

        Returns:
            pd.DataFrame: Dataset with the model output
        """
        pass

    def export(self, **kwargs) -> None:
        """Private function to save the trained model in a serialized file format

        Args:
            filepath (str): Name of the file with the model. Defaults to '%THE_MODEL_NAME%_model.pkl'
        """
        assert self.model is not None, ValueError("The model doesn't exist")

        # filePath = kwargs.get("filePath", f"{self.model}_model.pkl")
        filePath = "model.pkl"

        # assert filePath.split(".")[-1] == "pkl", ValueError(
        #     "The extension is not supported in the current version of FasterML."
        # )

        # TODO: Create config with the path variables to save/load files
        with open(file=os.path.join(".artifacts", filePath), mode="wb") as file:
            pickle.dump(self.model, file)

        logging.info(f"Model {self.name} exported to {filePath}")

    def predict(self, df: pd.DataFrame):
        data = self.preprocess_data(df)
        return self.model.predict(data)

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Private function to retrieve the evaluation metrics of the model

        Args:
            df (pd.DataFrame): Dataset to evaluate

        Returns:
            dict[str, float]: Dictionary with the metric name as a key and the result as value
        """
        # TODO: Implement this method later, check with the guys. God please help ;-;. Check graphana implementation due dependencies
        pass


class DatabaseConfig:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)
