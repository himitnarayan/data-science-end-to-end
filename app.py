from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformationConfig
from src.mlproject.components.data_transformation import DataTransformation
if __name__=="__main__":
    logging.info("Starting the ML Project application.")

    try:
    #    data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully.")

        # data_transformation_config=DataTransformationConfig
        data_transformation=DataTransformation()
        data_transformation.initiate_data_transformation(
            train_data_path,
            test_data_path
        )

    except Exception as e:
        logging.info("An error occurred in the application.")
        raise CustomException(e, sys)