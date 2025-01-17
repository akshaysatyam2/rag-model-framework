# RAG-Model Framework

This repository provides a framework for building a **Retrieval-Augmented Generation (RAG) model** for various use cases. It allows users to upload documents, process them, and ask context-aware questions. The project is fully open-source and does not rely on any paid libraries or APIs, making it accessible to everyone.

This project is licensed under the **MIT License**, so feel free to use, modify, and contribute. Contributions are highly encouraged, and contributors will be credited in the contributors' list.

---

## Features
- **File Upload**: Supports `.txt`, `.pdf`, `.docx`, `.html`, `.json`, and `.xlsx` formats.
- **Multiple File Support**: Process multiple files simultaneously and combine their content for retrieval.
- **Document Chunking**: Splits large documents into smaller chunks for efficient retrieval.
- **Pre-trained Models**: Utilizes pre-trained models for retrieval, reranking, and answering.
- **Retriever Options**: Choose between two retriever models: `mpnet` or `minilm`.
- **Answer Generation**: Generates direct answers and enriched contextual answers using summarization.
- **Standalone Script**: Includes a standalone script (`main.py`) for command-line usage without Flask.

---

## Project Structure
```
your_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── utils/
│   │   └── data.py
│   └── templates/
│       └── index.html
│
├── requirements.txt
├── .gitignore
├── LICENCE.txt
├── README.md
├── CHANGELOG.md
├── main.py (standalone script for command-line usage)
└── run.py (entry point for Flask-based web application)
```

---

## Installation

### 1. **Clone the Repository**
   ```bash
   git clone https://github.com/akshaysatyam2/rag-model-framework.git
   cd rag-model-framework
   ```

### 2. **Set Up a Virtual Environment**
   A virtual environment is recommended to isolate dependencies for this project.

   #### On Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   #### On macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   > **Note**: If you don’t have `venv` installed, install it using:
   > ```bash
   > python -m ensurepip --upgrade
   > ```

### 3. **Install Dependencies**
   Install the required dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

### 4. **Install FAISS**
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

### Command Line (Standalone Script)
Run the `main.py` script:
```bash
python main.py
```
Follow the prompts to upload one or more documents and ask questions.

### Flask-Based Web Application
Run the `run.py` script to start the Flask web application:
```bash
python run.py
```
Open your browser and navigate to `http://127.0.0.1:5000/` to use the web interface.

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
- HTML (`.html`)
- JSON (`.json`)
- Excel (`.xlsx`, `.xls`)

---

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions. Contributors will be credited in the contributors' list.

---

## Main Author/Creator
**Akshay Kumar**   |
[Mail](mailto:akshaysatyam2003@gmail.com) |
[GitHub](https://github.com/akshaysatyam2) | 
[LinkedIn](https://www.linkedin.com/in/akshaysatyam2) | 
[X](https://x.com/akshaysatyam2) | 
[Kaggle](https://www.kaggle.com/akshaysatyam2) | 
[Instagram](https://www.instagram.com/akshaysatyam2/#) |

---

## Contributors
Akshay Kumar ([GitHub](https://github.com/akshaysatyam2))

---

## Changelog
For a detailed list of changes, see the [CHANGELOG.md](CHANGELOG.md) file.

### Recent Updates
#### [v1.1.0] - 17-01-2025
- **Added**:
  - Support for multiple file uploads.
  - Support for `.html`, `.json`, and `.xlsx` file formats.
  - Improved error handling and user interaction in the standalone script.
- **Changed**:
  - Updated repository structure for better organization.

#### [v1.0.0] - 14-01-2025
- **Added**:
  - Initial release with support for `.txt`, `.pdf`, and `.docx` files.
  - Flask-based web application for file upload and question answering.
  - Command-line interface for standalone usage.

---

## Versioning
This project follows [Semantic Versioning](https://semver.org/). All notable changes to this project are documented in the [CHANGELOG.md](CHANGELOG.md) file.

---

## How to Contribute
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and update the `CHANGELOG.md` file if necessary.
4. Submit a pull request with a detailed description of your changes.

---

## Reporting Issues
If you encounter any issues or have suggestions for improvements, please open an issue on the [GitHub Issues](https://github.com/akshaysatyam2/rag-model-framework/issues) page.

---

## Virtual Environment Tips
- **Activate the Virtual Environment**:
  - On Windows: `venv\Scripts\activate`
  - On macOS/Linux: `source venv/bin/activate`

- **Deactivate the Virtual Environment**:
  ```bash
  deactivate
  ```

- **Recreate the Virtual Environment**:
  If you need to recreate the virtual environment, delete the `venv` folder and follow the installation steps again.
