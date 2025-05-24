import requests
from bs4 import BeautifulSoup
import re
import os
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml
from utils.helpers import fetch_and_clean

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.urls = self.config['urls']
        self.bucket_name = self.config['bucket_name']
        self.file_name = self.config['content_file_name']

        os.makedirs(RAW_DIR,exist_ok =True)

        logger.info("Data Ingestion Started")

    def download_data_from_urls(self):
        try:
            if os.path.isfile(os.path.join(RAW_DIR,self.file_name)):
                logger.info("Content file already exists")
            else:
                logger.info("Content file Doesn't exists")
                with open(os.path.join(RAW_DIR,self.file_name),'w', encoding='utf-8') as file:

                    for url in self.urls:

                        clean_article_text = fetch_and_clean(url)
                        file.write(clean_article_text + '\n')

            logger.info("Data from Urls is written into content file")

        except Exception as e:
            logger.error("Error while downloading the data from the URLs ")
            raise CustomException("failed to download data from the URLs", e)
        
    def run(self):
        try:
            logger.info("starting Data Ingestion Process")
            self.download_data_from_urls()
            logger.info("Data Ingestion Completed.......")

        except CustomException as e:
            logger.error(f"CustomException : {str(e)}")
        
        finally:
            logger.info("Data Ingestion DONE...")

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
