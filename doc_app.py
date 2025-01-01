import os
import time
import streamlit as st
from dotenv import load_dotenv
from documents_llm.st_helpers import run_query

# Load environment variables
load_dotenv()

# Load model parameters
MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = os.getenv("OPENAI_URL")

st.title("🐍 VCF Document Analyzer")
st.write(
    "This is a simple document analyzer that uses LLM models to summarize and answer questions about documents. "
    "You can upload a PDF or text file and the model will summarize the document and answer questions about it."
)

with st.sidebar:
    st.header("Model")
    model_name = st.text_input("Model name", value=MODEL_NAME)
    temperature = st.slider("Temperature", value=0.1, min_value=0.0, max_value=1.0)

    st.header("Document")
    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    start_page = st.number_input("Start page:", value=0, min_value=0)
    end_page = st.number_input("End page:", value=-1)

    query_type = st.radio("Query type", ["Summarize", "Query"])
    user_query = st.text_area("Query", value="What is the data used?") if query_type == "Query" else ""

if st.button("Run"):
    if not file:
        st.error("Please upload a file.")
    else:
        result = run_query(
            file, query_type == "Summarize", user_query, start_page, end_page,
            model_name, OPENAI_API_KEY, OPENAI_URL, temperature
        )
        st.markdown(result)
