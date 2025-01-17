from docx import Document
import PyPDF2
import pdfplumber
import json
import pandas as pd
from bs4 import BeautifulSoup

def read_docx(file_path):
    """
    Reads content from a .docx file and returns structured data.
    """
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return [{"content": text, "meta": {"name": file_path}}]
    except Exception as e:
        raise ValueError(f"Error reading .docx file: {e}")

def read_pdf(file_path):
    """
    Reads content from a .pdf file using PyPDF2 or pdfplumber if PyPDF2 fails.
    """
    try:
        text = ""
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        
        if not text.strip():
            raise ValueError("PyPDF2 failed to extract text, trying pdfplumber.")

        return [{"content": text.strip(), "meta": {"name": file_path}}]
    
    except Exception as e:
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            if not text.strip():
                raise ValueError("pdfplumber also failed to extract text.")
            return [{"content": text.strip(), "meta": {"name": file_path}}]
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {e}")

def read_txt(file_path):
    """
    Reads content from a .txt file and returns structured data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as txt_file:
            text = txt_file.read().strip()
        if not text:
            raise ValueError("The text file is empty.")
        return [{"content": text, "meta": {"name": file_path}}]
    except Exception as e:
        raise ValueError(f"Error reading text file: {e}")

def read_html(file_path):
    """
    Reads content from an HTML file and returns structured data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as html_file:
            soup = BeautifulSoup(html_file, 'html.parser')
            text = soup.get_text(separator='\n').strip()
            return [{"content": text, "meta": {"name": file_path}}]
    except Exception as e:
        raise ValueError(f"Error reading HTML file: {e}")

def read_json(file_path):
    """
    Reads content from a JSON file and returns structured data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            text = json.dumps(data, indent=2)
            return [{"content": text, "meta": {"name": file_path}}]
    except Exception as e:
        raise ValueError(f"Error reading JSON file: {e}")

def read_excel(file_path):
    """
    Reads content from an Excel file and returns structured data.
    """
    try:
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)
        return [{"content": text, "meta": {"name": file_path}}]
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")

def getData(file_path):
    """
    Determines the file type by its extension and calls the appropriate reader function.
    Returns structured data or raises an error for unsupported file types.
    """
    if file_path.endswith('.docx'):
        return read_docx(file_path)
    elif file_path.endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.endswith('.txt'):
        return read_txt(file_path)
    elif file_path.endswith('.html'):
        return read_html(file_path)
    elif file_path.endswith('.json'):
        return read_json(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a .docx, .pdf, .txt, .html, .json, or .xlsx file.")