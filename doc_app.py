import os
import time
import streamlit as st
from dotenv import load_dotenv
from documents_llm.st_helpers import run_query

# Load environment variables
load_dotenv()

# Model and API Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "mixtral:latest")
BASE_URL = os.getenv("BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit App
st.title("PDF Analyzer App")
st.write(
    "Upload a PDF to summarize or query using a large language model."
)

# Sidebar Input
with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input("Model Name", value=MODEL_NAME)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.subheader("Page Range")
    start_page = st.number_input("Start Page", min_value=0, value=0)
    end_page = st.number_input("End Page", value=-1)

    query_type = st.radio("Task", ["Summarize", "Query"])

if query_type == "Query":
    user_query = st.text_area("Enter your query")

if st.button("Run"):
    if not uploaded_file:
        st.error("Please upload a PDF file.")
    else:
        with st.spinner("Processing..."):
            try:
                result = run_query(
                    uploaded_file=uploaded_file,
                    summarize=query_type == "Summarize",
                    user_query=user_query if query_type == "Query" else "",
                    start_page=start_page,
                    end_page=end_page,
                    model_name=model_name,
                    base_url=BASE_URL,
                    temperature=temperature,
                )
                st.success("Task Completed!")
                st.text_area("Result", result, height=300)
            except Exception as e:
                st.error(f"An error occurred: {e}")
