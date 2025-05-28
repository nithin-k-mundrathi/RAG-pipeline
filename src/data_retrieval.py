import os
import numpy as np
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger(__name__)

class DataRetriever:
    def __init__(self, config, vectordb_path, retrieved_df_path, output_dir):
        # Load config parameters relevant to data retrieval
        self.embedding_model = config["embedding_model"]
        self.config = config["data_retriever"]
        self.top_k = self.config["top_k"]
        self.output_dir = output_dir

        self.vectordb_path = vectordb_path
        self.retrieved_df_path = retrieved_df_path

        self.question = None
        self.model_embedding = None
        self.vector_db = None
        self.similarities = None

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("Initialized DataRetriever: starting data retrieval setup")

    def load_query_question(self):
        try:
            # Prompt user for the input question to retrieve relevant chunks
            self.question = input("Please enter your question: ").strip()
            logger.info("Successfully loaded input query question")

        except Exception as e:
            logger.error("Failed to load input query question")
            raise CustomException("Error while loading input query question", e)

    def load_vectordb(self):
        try:
            # Initialize embedding model and load vector database from local storage
            self.model_embedding = HuggingFaceEmbeddings(model_name=self.embedding_model)
            self.vector_db = FAISS.load_local(
                self.vectordb_path,
                self.model_embedding,
                allow_dangerous_deserialization=True
            )
            logger.info("Successfully loaded embedding model and vector database")

        except Exception as e:
            logger.error("Failed to load embedding model and vector database")
            raise CustomException("Error while loading embedding model and vector database", e)

    def get_cosine_similarity(self):
        try:
            logger.info("Computing embedding for input question")
            query_embedding = self.model_embedding.embed_query(self.question)

            # Reconstruct all stored chunk embeddings from FAISS index
            chunks_embedding = self.vector_db.index.reconstruct_batch(
                list(range(self.vector_db.index.ntotal))
            )

            logger.info("Converting question and chunk embeddings into numpy arrays")
            query_vector = np.array(query_embedding).reshape(1, -1)
            chunks_vectors = np.array(chunks_embedding)

            logger.info("Calculating cosine similarity between question and chunk embeddings")
            self.similarities = cosine_similarity(query_vector, chunks_vectors)[0]

        except Exception as e:
            logger.error("Failed to calculate cosine similarity between question and chunks")
            raise CustomException("Error while calculating similarity between question and chunks", e)

    def save_retrieved_chunks(self):
        try:
            logger.info(f"Selecting top {self.top_k} chunks most similar to the question")
            # gives u the index ascending order
            top_k_indices = self.similarities.argsort()[-self.top_k:][::-1]
            # extract the scores from the index's
            top_k_scores = self.similarities[top_k_indices]

            # use the top-k index's to getback the chunks from vectorDB.
            top_k_chunks = []
            for i in top_k_indices:
                doc_id = self.vector_db.index_to_docstore_id.get(i)
                if doc_id and doc_id in self.vector_db.docstore._dict:
                    top_k_chunks.append(self.vector_db.docstore._dict[doc_id])

            logger.info("Compiling top chunks and similarity scores into dataframe for downstream processing")

            new_data = [{
                "query": self.question,
                "score": score,
                "document": doc.page_content
            } for doc, score in zip(top_k_chunks, top_k_scores)]

            new_df = pd.DataFrame(new_data)

            os.makedirs(os.path.dirname(self.retrieved_df_path), exist_ok=True)

            if os.path.exists(self.retrieved_df_path):
                try:
                    existing_df = pd.read_csv(self.retrieved_df_path)

                    if set(existing_df.columns) == set(new_df.columns):
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    else:
                        logger.warning("⚠️ Column mismatch. Overwriting retrieval file.")
                        combined_df = new_df
                except Exception as e:
                    logger.warning(f"⚠️ Failed to read existing file. Overwriting. Reason: {e}")
                    combined_df = new_df
            else:
                combined_df = new_df

            combined_df.to_csv(self.retrieved_df_path, index=False)
            logger.info(f"Retrieved chunks saved to: {self.retrieved_df_path}")

        except Exception as e:
            logger.error("Failed to save retrieved chunks into dataframe")
            raise CustomException("Error while saving retrieved chunks into dataframe", e)

    def run(self):
        try:
            logger.info("Starting full data retrieval workflow")
            self.load_query_question()
            self.load_vectordb()
            self.get_cosine_similarity()
            self.save_retrieved_chunks()
            logger.info("Completed data retrieval workflow successfully")

        except CustomException as e:
            logger.error(f"Data retrieval encountered an error: {str(e)}")

        finally:
            logger.info("Data retrieval process finished")

if __name__ == "__main__":
    data_retriever = DataRetriever(read_yaml(CONFIG_PATH), VECTORDB_PATH, RETRIEVED_DF_PATH, RETRIEVAL_DIR)
    data_retriever.run()
