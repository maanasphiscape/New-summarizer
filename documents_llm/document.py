from PyPDF2 import PdfReader
from langchain_core.documents.base import Document

def load_pdf(file_path, start_page=0, end_page=-1):
    reader = PdfReader(file_path)
    pages = reader.pages[start_page:end_page] if end_page != -1 else reader.pages[start_page:]
    docs = [Document(content=page.extract_text()) for page in pages]
    return docs
