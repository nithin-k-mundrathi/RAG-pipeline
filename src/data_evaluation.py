import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall
from ragas.metrics import NonLLMContextPrecisionWithReference
from utils.common_functions import read_yaml
import asyncio  # Required for running async code

logger = get_logger(__name__)

class Evaluation:
    def __init__(self, config, generator_df_path, retrieved_df_path):
        self.config = config
        self.generator_df_path = generator_df_path
        self.retrieved_df_path = retrieved_df_path

        self.question = None
        self.context_chunks = None
        self.generated_text = None
        logger.info("Initialized Evaluation: starting Evaluation")

    def load_generation(self):
        try:
            logger.info(f"Loading retrieved data from file: {self.retrieved_df_path}")
            retrieved_df = pd.read_csv(self.retrieved_df_path)
            logger.info(f"Retrieved data loaded with {len(retrieved_df)} rows.")

            self.query = retrieved_df['query'][retrieved_df.shape[0] - 1]
            logger.info(f"Extracted query for processing: '{self.query}'")

            self.context_chunks = retrieved_df[retrieved_df['query'] == self.query]['document'].tolist()
            logger.info(f"Loaded {len(self.context_chunks)} context chunks for answering.")

            generator_df = pd.read_csv(self.generator_df_path)
            self.generated_text = generator_df[generator_df['query'] == self.query]['answer'].tolist()
        except Exception as e:
            logger.error("Failed to load retrieved data from CSV.")
            raise CustomException("Failed to load retrieved data", e)

    async def metrics(self):
        sample = SingleTurnSample(
            retrieved_contexts=self.generated_text,
            reference_contexts=self.context_chunks
        )

        context_recall = NonLLMContextRecall()
        recall = await context_recall.single_turn_ascore(sample)

        context_precision = NonLLMContextPrecisionWithReference()
        precision = await context_precision.single_turn_ascore(sample)
        print(recall,precision)


    async def run(self):
        try:
            logger.info("Starting full data evaluation workflow")
            self.load_generation()
            await self.metrics()
            logger.info("Completed data evaluation workflow successfully")
        except CustomException as e:
            logger.error(f"Data evaluation encountered an error: {str(e)}")
        finally:
            logger.info("Data evaluation process finished")


if __name__ == "__main__":
    evaluation = Evaluation(read_yaml(CONFIG_PATH),GENERATOR_DF_PATH, RETRIEVED_DF_PATH)
    asyncio.run(evaluation.run())
