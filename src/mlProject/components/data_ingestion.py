import os
import sys
from src.mlProject.exception import CustomException
from src.mlProject.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifact','train.csv')
    test_data_path = os.path.join('artifact','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config =  DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            data = pd.read_csv('Housing.csv')
            df = data[["area","stories","bedrooms","bathrooms","airconditioning","parking","prefarea","furnishingstatus","price"]]
            print('reading dataset')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            print('Data Ingestion is completed you can see artifact folder')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
        except Exception as  e:
            print("Custom exception")

            raise CustomException(e,sys)