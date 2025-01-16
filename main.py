import os
import docx
import PyPDF2
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import torch

device = 0 if torch.cuda.is_available() else 'cpu'

retriever_model_mpnet = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
retriever_model_minilm = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device=device)
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

retriever_model = retriever_model_mpnet

def read_file(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == 'pdf':
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return ' '.join(page.extract_text() for page in reader.pages)
    elif ext == 'docx':
        doc = docx.Document(filepath)
        return ' '.join(paragraph.text for paragraph in doc.paragraphs)
    else:
        raise ValueError("Unsupported file format.")

def split_text(text, chunk_size=150, overlap=50):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def build_retrieval_pipeline(text, embedder='mpnet'):
    global retriever_model
    retriever_model = retriever_model_mpnet if embedder == 'mpnet' else retriever_model_minilm

    chunks = split_text(text)
    embeddings = retriever_model.encode(chunks, convert_to_tensor=True, batch_size=8).cpu().numpy().astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efSearch = 64
    index.add(embeddings)

    return {
        "chunks": chunks,
        "index": index
    }

def improved_rag_pipeline(question, pipeline_data, top_k=15):
    question_embedding = retriever_model.encode([question], convert_to_tensor=True).cpu().numpy().astype('float32')
    distances, indices = pipeline_data['index'].search(question_embedding, k=top_k)
    retrieved_chunks = [pipeline_data['chunks'][i] for i in indices[0]]

    rerank_scores = reranker.predict([(question, chunk) for chunk in retrieved_chunks])
    ranked_chunks = [chunk for _, chunk in sorted(zip(rerank_scores, retrieved_chunks), reverse=True)]

    context = " ".join(ranked_chunks[:5])
    qa_result = qa_model(question=question, context=context)
    direct_answer = qa_result['answer']
    enriched_answer = summarizer(context, min_length=25, do_sample=False)[0]['summary_text']

    return {
        "Direct_Answer": direct_answer,
        "Enriched_Contextual_Answer": enriched_answer
    }

if __name__ == "__main__":
    print("Welcome to the RAG Question-Answering System")
    file_path = input("Enter the path of the file to process (txt, pdf, docx): ").strip()

    if not os.path.isfile(file_path):
        print("Invalid file path. Please try again.")
        exit(1)

    try:
        embedder_choice = input("Choose embedder model ('mpnet' or 'minilm'): ").strip().lower() or 'mpnet'
        text = read_file(file_path)
        pipeline_components = build_retrieval_pipeline(text, embedder_choice)

        print("File processed successfully. You can now ask questions.")
        while True:
            question = input("\nEnter your question (or type 'exit' to quit): ").strip()
            if question.lower() == 'exit':
                print("Exiting the system. Goodbye!")
                break

            answer = improved_rag_pipeline(question, pipeline_components)
            print("\nDirect Answer: ", answer['Direct_Answer'])
            print("Enriched Contextual Answer: ", answer['Enriched_Contextual_Answer'])

    except Exception as e:
        print(f"An error occurred: {e}")
