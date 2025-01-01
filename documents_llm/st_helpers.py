from pathlib import Path
import streamlit as st
import logging
from .summarize import summarize_document
from .query import query_document
from .document import load_pdf

logging.basicConfig(level=logging.DEBUG)

def save_uploaded_file(uploaded_file, output_dir=Path("/tmp")) -> Path:
    output_path = output_dir / uploaded_file.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return output_path

def run_query(
    uploaded_file, summarize, user_query, start_page, end_page, model_name, base_url, temperature
):
    logging.info("Saving uploaded file...")
    file_path = save_uploaded_file(uploaded_file)

    logging.info("Loading PDF...")
    docs = load_pdf(file_path, start_page=start_page, end_page=end_page)

    file_path.unlink()  # Delete the temporary file after processing

    if summarize:
        logging.info("Summarizing the document...")
        return summarize_document(docs, model_name=model_name, base_url=base_url, temperature=temperature)

    logging.info("Querying the document...")
    return query_document(docs, user_query=user_query, model_name=model_name, base_url=base_url, temperature=temperature)
