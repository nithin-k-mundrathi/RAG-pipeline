import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml
from config.paths_config import *

logger = get_logger(__name__)

class DataGenerator:
    def __init__(self, config, retrieved_df_path,output_dir, generator_df_path):
        # Load relevant configuration for data generation and model name
        self.llm_generator = config["text_to_text_model"]
        self.config = config["data_generator"]
        
        # Initialize tokenizer and model once at class instantiation
        logger.info(f"Loading tokenizer and model from '{self.llm_generator}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_generator)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_generator)
        logger.info("Tokenizer and model loaded successfully.")

        self.retrieved_df_path = retrieved_df_path
        self.generator_df_path = generator_df_path
        self.output_dir = output_dir
        self.query = None
        self.context_chunks = None
        self.answer = None

        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("DataGenerator instance created and ready.")

    def load_retrieval(self):
        try:
            logger.info(f"Loading retrieved data from file: {self.retrieved_df_path}")
            retrieved_df = pd.read_csv(self.retrieved_df_path)
            logger.info(f"Retrieved data loaded with {len(retrieved_df)} rows.")

            # Extract the query text (assumed first row)
            
            self.query = retrieved_df['query'][retrieved_df.shape[0]-1]
            logger.info(f"Extracted query for processing: '{self.query}'")

            # Extract context documents as a list of text chunks
            self.context_chunks = retrieved_df[retrieved_df['query']==self.query]['document'].tolist()
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
            prompt = f"<|user|> Relevant information: {total_context} Provide answer to question with relevant information provided above: {self.query}<|end|> <|assistant|>"

            # Tokenize prompt with padding and truncation
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
                padding="max_length"
            )
            logger.info("Tokenized prompt ready for model generation.")

            # Generate output tokens
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.75,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
            logger.info("Model generation completed. Decoding output tokens...")

            # Decode output tokens into human-readable string
            self.answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Print the final question and generated answer
            print(f'Question: {self.query}\n')
            print(f'Answer: {self.answer}\n')

            # Prepare new data row
            new_data = {
                "query": self.query,
                "context": total_context,
                "answer": self.answer
            }

            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.generator_df_path), exist_ok=True)
            new_row = pd.DataFrame([new_data])

            # Append if file exists, else create
            if os.path.exists(self.generator_df_path):
                try:
                    existing_df = pd.read_csv(self.generator_df_path)
                    if set(existing_df.columns) == set(new_row.columns):
                        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
                    else:
                        logger.warning("⚠️ Column mismatch detected. Overwriting existing file.")
                        updated_df = new_row
                except Exception as e:
                    logger.warning(f"⚠️ Failed to read existing file. Overwriting. Reason: {e}")
                    updated_df = new_row
            else:
                updated_df = new_row

            # Save to CSV
            updated_df.to_csv(self.generator_df_path, index=False)
            logger.info(f"Saved generated output to: {self.generator_df_path}")
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
    data_generator = DataGenerator(read_yaml(CONFIG_PATH), RETRIEVED_DF_PATH, GENERATOR_DIR, GENERATOR_DF_PATH)
    data_generator.run()
