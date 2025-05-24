import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml
from config.paths_config import *

logger = get_logger(__name__)

class DataGenerator:
    def __init__(self, config, retrieved_df_path):
        # Load relevant configuration for data generation and model name
        self.config = config["data_generator"]
        self.llm_generator = self.config["text_to_text_model"]

        # Initialize tokenizer and model once at class instantiation
        logger.info(f"Loading tokenizer and model from '{self.llm_generator}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_generator)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_generator)
        logger.info("Tokenizer and model loaded successfully.")

        self.retrieved_df_path = retrieved_df_path

        self.query = None
        self.context_chunks = None
        self.answer = None

        logger.info("DataGenerator instance created and ready.")

    def load_retrieval(self):
        try:
            logger.info(f"Loading retrieved data from file: {self.retrieved_df_path}")
            retrieved_df = pd.read_csv(self.retrieved_df_path)
            logger.info(f"Retrieved data loaded with {len(retrieved_df)} rows.")

            # Extract the query text (assumed first row)
            self.query = retrieved_df['query'][0]
            logger.info(f"Extracted query for processing: '{self.query}'")

            # Extract context documents as a list of text chunks
            self.context_chunks = retrieved_df['document'].tolist()
            logger.info(f"Loaded {len(self.context_chunks)} context chunks for answering.")

        except Exception as e:
            logger.error("Failed to load retrieved data from CSV.")
            raise CustomException("Failed to load retrieved data", e)

    def generate_answer(self):
        try:
            # Combine all context chunks into one string separated by blank lines
            total_context = "\n\n".join(self.context_chunks)
            logger.info("Prepared prompt by combining question and context chunks.")

            # Format prompt input for model consumption
            prompt = f"Question: {self.query}\n\nContext:\n{total_context}"

            # Tokenize prompt with padding and truncation for fixed max length input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
                padding="max_length"
            )
            logger.info("Tokenized prompt ready for model generation.")

            # Generate output tokens with specified decoding parameters
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.75,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
            logger.info("Model generation completed. Decoding output tokens...")

            # Decode output tokens into human-readable string
            self.answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Print the final question and generated answer
            print(f'Question: {self.query}\n')
            print(f'Answer: {self.answer}\n')

            logger.info("Answer generated and output successfully.")

        except Exception as e:
            logger.error("An error occurred during answer generation.")
            raise CustomException("Failed to generate answer from context", e)

    def run(self):
        try:
            logger.info("Starting full data generation workflow...")
            self.load_retrieval()
            self.generate_answer()
            logger.info("Data generation workflow completed successfully.")

        except CustomException as e:
            logger.error(f"CustomException caught during data generation: {str(e)}")

        finally:
            logger.info("Data generation process finished (success or failure).")


if __name__ == "__main__":
    data_generator = DataGenerator(read_yaml(CONFIG_PATH), RETRIEVED_DF_PATH)
    data_generator.run()
