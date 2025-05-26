from utils.common_functions import read_yaml
from config.paths_config import *
from src.data_retrieval import DataRetriever
from src.data_generator import DataGenerator
from src.data_evaluation import Evaluation
import asyncio

if __name__ == "__main__":
    data_retriever = DataRetriever(read_yaml(CONFIG_PATH),VECTORDB_PATH,RETRIEVED_DF_PATH,RETRIEVAL_DIR)
    data_retriever.run()

    data_generator = DataGenerator(read_yaml(CONFIG_PATH), RETRIEVED_DF_PATH, GENERATOR_DIR, GENERATOR_DF_PATH)
    data_generator.run()
    
    evaluation = Evaluation(read_yaml(CONFIG_PATH),GENERATOR_DF_PATH, RETRIEVED_DF_PATH)
    asyncio.run(evaluation.run())
