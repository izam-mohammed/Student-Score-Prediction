import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationconfig
from src.components.model_trainer import ModelTrainer

from typing import Tuple
from typing_extensions import Annotated


@dataclass
class DataIngestionConfig:
    """
    Config class for Data Ingestion
    """

    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    """
    Class that performs Data Ingestion
    """

    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(
        self,
    ) -> Tuple[
        Annotated[str, "Trainig data path"],
        Annotated[str, "Testing data path"],
    ]:
        """
        Initializing and preparing the data for preprocessing
        """

        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read hte dataset as dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=True, header=True)

            logging.info("Train test split initialized")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=True, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=True, header=True
            )

            logging.info("Ingestion of data compleated")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)
