from src.mlProject.components.data_ingestion import DataIngestion
from src.mlProject.components.data_ingestion import DataIngestionConfig
from src.mlProject.components.data_transformation import DataTransformation
from src.mlProject.components.data_transformation import DataTranformationConfig
from src.mlProject.logger import logging
from src.mlProject.exception import CustomException
import sys

if __name__=='__main__':
    logging.info("The execution has started")

    try:
        # data_ingection_config =  DataIngestionConfig()
        data_ingestion =  DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()  

        # data_transformation_config = DataTranformationConfig()
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path,test_data_path)



    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)