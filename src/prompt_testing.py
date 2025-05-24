from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import os
from src.custom_exception import CustomException
from config.paths_config import *
from langchain_huggingface  import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

embedding_model = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    
vector_db_loaded = FAISS.load_local(
    'artifacts/processed/vector_db', embedding_model,allow_dangerous_deserialization=True)

template = """<|user|> Relevant information: {context} Provide answer to question with relevant information provided above: {question}<|end|> <|assistant|>"""

prompt = PromptTemplate(
 template=template,
 input_variables=["context", "question"]
)

# Load small LLM (FLAN-T5 Base)
llm_pipeline = pipeline(
    "text2text-generation",  # For models like FLAN-T5
    model="google/long-t5-tglobal-base",
    tokenizer="google/long-t5-tglobal-base",
    max_new_tokens=512,
    do_sample =True,
    temperature=0.75,
    device=-1  # CPU; use 0 if you have a GPU
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

rag = RetrievalQA.from_chain_type(
 llm=llm,
 chain_type='stuff',
 retriever=vector_db_loaded.as_retriever(search_kwargs={"k": 10}),
 chain_type_kwargs={
 "prompt": prompt
 },
 verbose=True
)

x = rag.invoke('tell me something about kpmg')
output_text = x['result']
end_token = "|end|"

if end_token in output_text:
    output_text = output_text.split(end_token)[0].strip()

print(output_text)
