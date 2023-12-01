import numpy as np
import pandas as pd
import sys

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    '''
    A class for training the data pipeline
    '''
    def __init__(self) -> None:
        pass
    
    def train(self) -> None:
        '''
        Train the model
        '''
        try:
            obj = DataIngestion()
            train_data, test_data = obj.initiate_data_ingestion()
            
            data_transformation = DataTransformation()
            train_data, test_data, _ = data_transformation.initiate_data_transformation(train_data, test_data)

            model_trainer = ModelTrainer()
            r2_score = model_trainer.initialize_model_trainer(train_data, test_data, )
            logging.info(f"Trained model with R2 score of {r2_score}")
            
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)