# RAG-Model Framework

This repository provides a framework for building a **Retrieval-Augmented Generation (RAG) model** for various use cases. It allows users to upload documents, process them, and ask context-aware questions. The project is fully open-source and does not rely on any paid libraries or APIs, making it accessible to everyone.

This project is licensed under the **MIT License**, so feel free to use, modify, and contribute. Contributions are highly encouraged, and contributors will be credited in the contributors' list.

---

## Features
- **File Upload**: Supports `.txt`, `.pdf`, and `.docx` formats.
- **Document Chunking**: Splits large documents into smaller chunks for efficient retrieval.
- **Pre-trained Models**: Utilizes pre-trained models for retrieval, reranking, and answering.
- **Retriever Options**: Choose between two retriever models: `mpnet` or `minilm`.
- **Answer Generation**: Generates direct answers and enriched contextual answers using summarization.

---

## Project Structure
- **templates/**: Contains UI files.
  - `index.html`: The main UI file for file upload and question answering.
- **utils/**: Contains utility functions and package management.
  - `data.py`: Handles data extraction from files (PDF, DOCX, TXT).
- **main.py**: The core file implementing the RAG model logic, from preprocessing to final output.
- **requirements.txt**: Lists all the required libraries to run the project.
- **sampleWebApp.py**: A Flask-based web application integrating the RAG model.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/akshaysatyam2/rag-model-framework.git
   cd rag-model-framework
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install `faiss`:
   - For CPU:
     ```bash
     pip install faiss-cpu
     ```
   - For GPU:
     ```bash
     pip install faiss-gpu
     ```

---

## Usage

### Command Line
Run the `main.py` script:
```bash
python main.py
```
Follow the prompts to upload a document and ask questions.

---

## Models Used
- **Retriever Models**:
  - `multi-qa-mpnet-base-dot-v1`
  - `all-MiniLM-L6-v2`
- **Reranker**:
  - `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **Question Answering**:
  - `deepset/roberta-base-squad2`
- **Summarization**:
  - `facebook/bart-large-cnn`

---

## Supported File Formats
- Text (`.txt`)
- PDF (`.pdf`)
- Word (`.docx`)

---

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions. Contributors will be credited in the contributors' list.

---

## Main Author/Creator
- **Akshay Kumar**  
  [GitHub Profile](https://github.com/akshaysatyam2)

---

## Contributors
- Akshay Kumar ([GitHub](https://github.com/akshaysatyam2)), 
