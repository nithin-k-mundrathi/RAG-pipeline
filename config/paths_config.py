import os

########################  DATA INGESTION ############################
RAW_DIR = "artifacts/raw"
CONFIG_PATH = "config/config.yaml"
CONTENT_DATA_TXT = os.path.join(RAW_DIR,"content_data.txt")

########################  DATA PROCESSING ############################
PROCESSED_DIR = "artifacts/processed"
CHUNKS_DF_PATH = os.path.join(PROCESSED_DIR,"chunks.csv")
VECTORDB_PATH = os.path.join(PROCESSED_DIR,"vector_db")

########################  DATA RETRIEVAL ############################
RETRIEVAL_DIR = "artifacts/retrieval"
RETRIEVED_DF_PATH = os.path.join(RETRIEVAL_DIR,"retrived_df.csv")

########################  DATA RETRIEVAL ############################
GENERATOR_DIR = "artifacts/generator"
GENERATOR_DF_PATH = os.path.join(GENERATOR_DIR,"generator_df.csv")