from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformationConfig
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainerConfig
from src.mlproject.components.model_trainer import ModelTrainer


if __name__=="__main__":
    logging.info("Starting the ML Project application.")

    try:
    #    data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully.")

        # data_transformation_config=DataTransformationConfig
        data_transformation=DataTransformation()
        train_arr,test_arr,preprocessor_path=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))   

    except Exception as e:
        logging.info("An error occurred in the application.")
        raise CustomException(e, sys)