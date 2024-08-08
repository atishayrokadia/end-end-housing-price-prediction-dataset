from src.mlProject.components.data_ingestion import DataIngestion
from src.mlProject.components.data_ingestion import DataIngestionConfig
from src.mlProject.components.data_transformation import DataTransformation
from src.mlProject.components.data_transformation import DataTranformationConfig
from src.mlProject.components.model_trainer import ModelTrainer
from src.mlProject.components.model_trainer import ModelTrainerConfig
from src.mlProject.logger import logging
from src.mlProject.exception import CustomException
import sys

if __name__=='__main__':
    logging.info("The execution has started")

    try:
        # data_ingection_config =  DataIngestionConfig()
        data_ingestion =  DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()  
        print(train_data_path)
        print(test_data_path)
        # data_transformation_config = DataTranformationConfig()
        data_transformation = DataTransformation()
        train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        print(train_arr,'~~~~~~~~~~~~~~~~~~~~~~~')
        print(test_arr,'~~~~~~~~~~~~~~~~~~~~~~~`')
        print(train_arr.shape,'~~~~~~~~~~~~~~~~~~~~~~~')
        print(test_arr.shape,'~~~~~~~~~~~~~~~~~~~~~~~`')
        # model_trainer
        model_trainer =  ModelTrainer()
        print("MAPE value is")
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))



    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)