import os
import docx
import PyPDF2
import pdfplumber
import json
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set device
device = 0 if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load models
retriever_model_mpnet = SentenceTransformer("multi-qa-mpnet-base-dot-v1", device="cuda" if device == 0 else "cpu")
retriever_model_minilm = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if device == 0 else "cpu")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device=device)
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

retriever_model = retriever_model_mpnet

# File reading functions
def read_txt(file_path):
    """Read content from a .txt file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Error reading .txt file: {e}")

def read_pdf(file_path):
    """Read content from a .pdf file using PyPDF2 or pdfplumber."""
    try:
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"

        if not text.strip():
            raise ValueError("PyPDF2 failed to extract text, trying pdfplumber.")

        return text.strip()

    except Exception as e:
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            if not text.strip():
                raise ValueError("pdfplumber also failed to extract text.")
            return text.strip()
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {e}")

def read_docx(file_path):
    """Read content from a .docx file."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        raise ValueError(f"Error reading .docx file: {e}")

def read_html(file_path):
    """Read content from an HTML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text(separator="\n").strip()
    except Exception as e:
        raise ValueError(f"Error reading .html file: {e}")

def read_json(file_path):
    """Read content from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return json.dumps(data, indent=2)
    except Exception as e:
        raise ValueError(f"Error reading .json file: {e}")

def read_excel(file_path):
    """Read content from an Excel file."""
    try:
        df = pd.read_excel(file_path)
        return df.to_string(index=False)
    except Exception as e:
        raise ValueError(f"Error reading .xlsx file: {e}")

def read_file(file_path):
    """Determine the file type and read its content."""
    ext = file_path.rsplit(".", 1)[1].lower()
    if ext == "txt":
        return read_txt(file_path)
    elif ext == "pdf":
        return read_pdf(file_path)
    elif ext == "docx":
        return read_docx(file_path)
    elif ext == "html":
        return read_html(file_path)
    elif ext == "json":
        return read_json(file_path)
    elif ext in ("xlsx", "xls"):
        return read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def split_text(text, chunk_size=150, overlap=50):
    """Split text into chunks with overlap."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def build_retrieval_pipeline(text, embedder="mpnet"):
    """Build a retrieval pipeline using FAISS and a retriever model."""
    global retriever_model
    retriever_model = retriever_model_mpnet if embedder == "mpnet" else retriever_model_minilm

    chunks = split_text(text)
    embeddings = retriever_model.encode(chunks, convert_to_tensor=True, batch_size=8).cpu().numpy().astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efSearch = 64
    index.add(embeddings)

    return {"chunks": chunks, "index": index}

def improved_rag_pipeline(question, pipeline_data, top_k=15):
    """Retrieve and generate answers using the RAG pipeline."""
    try:
        question_embedding = retriever_model.encode([question], convert_to_tensor=True).cpu().numpy().astype("float32")
        distances, indices = pipeline_data["index"].search(question_embedding, k=top_k)
        retrieved_chunks = [pipeline_data["chunks"][i] for i in indices[0]]

        rerank_scores = reranker.predict([(question, chunk) for chunk in retrieved_chunks])
        ranked_chunks = [chunk for _, chunk in sorted(zip(rerank_scores, retrieved_chunks), reverse=True)]

        context = " ".join(ranked_chunks[:5])
        qa_result = qa_model(question=question, context=context, max_answer_len=100)
        direct_answer = qa_result["answer"]
        enriched_answer = summarizer(context, min_length=25, max_length=100, do_sample=False)[0]["summary_text"]

        return {
            "Direct_Answer": direct_answer,
            "Enriched_Contextual_Answer": enriched_answer,
        }
    except Exception as e:
        logging.error(f"Error in RAG pipeline: {e}")
        return {
            "Direct_Answer": "An error occurred while processing your question.",
            "Enriched_Contextual_Answer": "An error occurred while generating the enriched answer.",
        }

def main():
    print("Welcome to the RAG Question-Answering System")
    file_paths = input("Enter the paths of the files to process (comma-separated): ").strip().split(",")

    combined_text = ""
    for file_path in file_paths:
        file_path = file_path.strip()
        if not os.path.isfile(file_path):
            print(f"Invalid file path: {file_path}. Please try again.")
            exit(1)

        try:
            text = read_file(file_path)
            combined_text += text + "\n"
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            exit(1)

    try:
        embedder_choice = input("Choose embedder model ('mpnet' or 'minilm'): ").strip().lower() or "mpnet"
        pipeline_components = build_retrieval_pipeline(combined_text, embedder_choice)

        print("Files processed successfully. You can now ask questions.")
        while True:
            question = input("\nEnter your question (or type 'exit' to quit): ").strip()
            if question.lower() == "exit":
                print("Exiting the system. Goodbye!")
                break

            answer = improved_rag_pipeline(question, pipeline_components)
            print("\nDirect Answer: ", answer["Direct_Answer"])
            print("Enriched Contextual Answer: ", answer["Enriched_Contextual_Answer"])

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()