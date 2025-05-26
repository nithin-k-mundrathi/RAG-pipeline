import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml
from langchain_huggingface  import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self,config,input_file,output_dir,chunks_df_path,vectordb_path):
        self.embedding_model = config["embedding_model"]
        self.config = config["data_processing"]
        self.chunk_size = self.config["chunk_size"]
        self.overlap_chunk_size = self.config['overlap_chunk_size']
        
        self.input_file = input_file
        self.output_dir = output_dir
        self.vectordb_path = vectordb_path
        self.chunks_df_path = chunks_df_path

        self.content_data = None
        self.chunked_data = None 
        self.model_embedding = None
        self.vector_db =  None

        os.makedirs(self.output_dir,exist_ok =True)

        logger.info("Data Processing Started")

    def load_data(self):
        try:
            with open(self.input_file,'r',encoding='utf-8') as file:
                self.content_data = file.read()

            logger.info("Data loaded succesfully for data processing from input file")

        except Exception as e:
            logger.error("Error while loading data for data procesing from input file")
            raise CustomException("failed to load data for data procesing from input file", e)

    def chunking_data(self):
        try:
            logger.info("Chunking the data based on the chunk size with overlapping chunks")

            CHUNK_SIZE = self.chunk_size
            self.chunked_data = [self.content_data[i:i+CHUNK_SIZE] for i in range(0,len(self.content_data), CHUNK_SIZE-self.overlap_chunk_size)]
            
            # Convert to DataFrame
            df_chunks = pd.DataFrame(self.chunked_data , columns=["chunks_text"])

            # Store the chunks in a Processed folder
            df_chunks.to_csv(self.chunks_df_path, index = False)
            
            logger.info("Chunking of data with overlapping is successfully completed")
        except Exception as e:
            logger.error(f"Error while Chunking data during data procesing from input file: {e}")
            raise CustomException("failed to chunk data during data procesing from input file", e)

    def chunk_to_embedding_model(self):
        try:
            logger.info("starting the Loading of the Embedding Model")
            chunked_texts = [t.replace('\n'," ") for t in self.chunked_data]
            # print(len(texts[0]),len(texts[1]),len(texts[2]),len(texts[3]))
            # print(texts[0])
            self.model_embedding = HuggingFaceEmbeddings(model_name = self.embedding_model)
            logger.info("Successful in Loading the Embedding Model")
            # embeddings = self.embedding_model.embed_documents(texts)

            logger.info("Started Loading the embeddings into Faiss vector DB")
            self.vector_db = FAISS.from_texts(chunked_texts, self.model_embedding)
            logger.info("Successful in Loading the embeddings into Faiss vector DB")

            logger.info("DONE with converting chunks into Embeddings")
        except Exception as e:
            logger.error(f"Error while Converting chunks into Embeddings: {e}")
            raise CustomException("failed to  Convert chunks into Embeddings", e)
        
    def save_vector_db(self):
        try:
            logger.info("start Saving the VectorDb file to Disk")
            self.vector_db.save_local(self.vectordb_path)
            logger.info("Succesful in saving the VectorDb to Disk")
        except Exception as e:
            logger.error(f"Error while Saving the vectorDb file to Disk: {e}")
            raise CustomException("failed to Save the vectorDb file to Disk", e)


    def run(self):
        try:
            logger.info("starting Data Processing Process")
            self.load_data()
            self.chunking_data()
            self.chunk_to_embedding_model()
            self.save_vector_db()
            logger.info("Data Processing Completed.......")

        except CustomException as e:
            logger.error(f"CustomException : {str(e)}")
        
        finally:
            logger.info("Data Processing DONE...")

if __name__ == "__main__":

    data_processor = DataProcessor(read_yaml(CONFIG_PATH),CONTENT_DATA_TXT,PROCESSED_DIR,CHUNKS_DF_PATH,VECTORDB_PATH)
    data_processor.run()
