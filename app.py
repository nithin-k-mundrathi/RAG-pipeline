import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from config.paths_config import *
from utils.common_functions import read_yaml

config = read_yaml(CONFIG_PATH)

# Title
st.set_page_config(page_title="RAG QA App")
st.title("📚 RAG Question Answering")

# Load embeddings and vector DB
@st.cache_resource
def load_vector_db():
    embedding_model = HuggingFaceEmbeddings(model_name=config['embedding_model'])
    vector_db = FAISS.load_local(
        VECTORDB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vector_db

# Load LLM
@st.cache_resource
def load_llm():
    llm_pipeline = pipeline(
        "text2text-generation",
        model=config['text_to_text_model'],
        tokenizer=config['text_to_text_model'],
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        device=-1  # CPU; change to 0 for GPU
    )
    return HuggingFacePipeline(pipeline=llm_pipeline)

# Load prompt template
def get_prompt_template():
    template = """<|user|> Relevant information: {context} Provide answer to question with relevant information provided above: {question}<|end|> <|assistant|>"""

    return PromptTemplate(template=template, input_variables=["context", "question"])

# Initialize RAG chain
@st.cache_resource
def build_rag_chain():
    retriever = load_vector_db().as_retriever(search_kwargs={"k": config['data_generator']['top_k']})
    prompt = get_prompt_template()
    llm = load_llm()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )
    return rag_chain

# Input box
query = st.text_input("Ask a question:")

if query:
    with st.spinner("Generating answer..."):
        rag = build_rag_chain()
        result = rag.invoke(query)
        output_text = result['result']
        end_token = "|end|"

        if end_token in output_text:
            output_text = output_text.split(end_token)[0].strip()
        st.success(output_text)
