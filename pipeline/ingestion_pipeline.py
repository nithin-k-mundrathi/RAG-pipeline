from utils.common_functions import read_yaml
from config.paths_config import *
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor
from src.data_retrieval import DataRetriever

if __name__ == "__main__":

    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    data_processor = DataProcessor(read_yaml(CONFIG_PATH),CONTENT_DATA_TXT,PROCESSED_DIR,CHUNKS_DF_PATH,VECTORDB_PATH)
    data_processor.run()

    data_retriever = DataRetriever(read_yaml(CONFIG_PATH),VECTORDB_PATH,RETRIEVED_DF_PATH,RETRIEVAL_DIR)
    data_retriever.run()

    
    
