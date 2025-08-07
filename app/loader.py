# loader.py
# Extract text from PDF using PyMuPDF

import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from the given PDF file using PyMuPDF.
    :param pdf_path: Path to the PDF file.
    :return: Combined text from all pages.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text()
    doc.close()
    return full_text

if __name__ == "__main__":
    sample_path = "../sample_docs/Financial_Report_2023.pdf"
    text = extract_text_from_pdf(sample_path)
    print(f"Extracted {len(text)} characters from the PDF.")
