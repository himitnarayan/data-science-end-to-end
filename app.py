from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_ingestion import DataIngestion
if __name__=="__main__":
    logging.info("Starting the ML Project application.")

    try:
    #    data_ingestion_config=DataIngestionConfig()
       data_ingestion=DataIngestion()
       data_ingestion.initiate_data_ingestion()
       logging.info("Data ingestion completed successfully.")
    except Exception as e:
        logging.info("An error occurred in the application.")
        raise CustomException(e, sys)