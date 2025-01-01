from pathlib import Path
import streamlit as st
from .document import load_pdf
from .query import query_document
from .summarize import summarize_document


def save_uploaded_file(uploaded_file: "UploadedFile", output_dir: Path = Path("/tmp")) -> Path:
    output_path = output_dir / uploaded_file.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return output_path


def run_query(uploaded_file: "UploadedFile", summarize: bool, user_query: str,
              start_page: int, end_page: int, model_name: str,
              openai_api_key: str, openai_url: str, temperature: float) -> str:
    file_path = save_uploaded_file(uploaded_file, output_dir=Path("/tmp"))
    docs = load_pdf(file_path, start_page=start_page, end_page=end_page)
    file_path.unlink()

    if summarize:
        return summarize_document(docs, model_name, openai_api_key, openai_url, temperature)
    return query_document(docs, user_query, model_name, openai_api_key, openai_url, temperature)
