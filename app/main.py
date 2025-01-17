from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import tempfile
from app.utils.data import getData
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import torch

app = Flask(__name__)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'html', 'json', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = 0 if torch.cuda.is_available() else 'cpu'

# Load models
retriever_model_mpnet = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
retriever_model_minilm = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device=device)
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

pipeline_components = {}
retriever_model = retriever_model_mpnet

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(error="No file uploaded."), 400

    files = request.files.getlist('file')  # Get list of uploaded files
    if not files or all(file.filename == '' for file in files):
        return jsonify(error="No selected file."), 400

    processed_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                embedder = request.form.get('embedder', 'mpnet')
                file_data = getData(filepath)
                processed_files.append({"filename": filename, "content": file_data[0]['content']})
            except Exception as e:
                return jsonify(error=str(e)), 500
        else:
            return jsonify(error="Unsupported file type. Only txt, pdf, docx, html, json, and xlsx are allowed."), 400

    # Combine content from all files
    combined_text = " ".join([file['content'] for file in processed_files])
    global pipeline_components
    pipeline_components = build_retrieval_pipeline(combined_text, embedder)

    return jsonify(message="Files uploaded and processed successfully.", embedder=embedder)

@app.route('/ask', methods=['POST'])
def ask_question():
    if not pipeline_components:
        return jsonify(error="No file uploaded or processed."), 400

    data = request.get_json()
    question = data.get('query')
    if not question:
        return jsonify(error="Question cannot be empty."), 400

    try:
        question_embedding = retriever_model.encode([question], convert_to_tensor=True).cpu().numpy().astype('float32')
        distances, indices = pipeline_components['index'].search(question_embedding, k=15)
        retrieved_chunks = [pipeline_components['chunks'][i] for i in indices[0]]

        rerank_scores = reranker.predict([(question, chunk) for chunk in retrieved_chunks])
        ranked_chunks = [chunk for _, chunk in sorted(zip(rerank_scores, retrieved_chunks), reverse=True)]

        context = " ".join(ranked_chunks[:5])
        qa_result = qa_model(question=question, context=context)
        direct_answer = qa_result['answer']
        enriched_answer = summarizer(context, min_length=25, do_sample=False)[0]['summary_text']

        return jsonify(answer=direct_answer, enriched=enriched_answer)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)