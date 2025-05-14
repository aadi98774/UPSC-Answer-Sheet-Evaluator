# 🧠 AI-Based UPSC Answer Sheet Evaluation with RAG

This project implements an end-to-end AI pipeline to evaluate scanned handwritten UPSC answer sheets using GPT-4o and a Retrieval-Augmented Generation (RAG) system grounded in an official SOP Evaluation PDF.

---

## 🚀 Getting Started

### 📦 Install Dependencies

This project uses `uv` for dependency management:

```
uv pip install -r pyproject.toml
```

▶️ Run the Application

```
streamlit run evaluation.py
```

Once the app launches, upload a scanned PDF of a student's handwritten UPSC answer sheet to generate a structured evaluation report.

### 🔧 Tech Stack

| Component  | Tool/Framework               |
| ---------- | ---------------------------- |
| LLM        | GPT-4o (OpenAI)              |
| OCR        | GPT-4o Vision                |
| Embeddings | OpenAI Embeddings            |
| Vector DB  | Qdrant (in-memory)           |
| Framework  | LangChain, LlamaIndex        |
| UI         | Streamlit                    |
| PDF Tools  | pdf2image, PyPDF2, reportlab |


### 🧠 RAG Architecture

          +--------------------+
          |  SOP (PDF Format)  |
          +--------------------+
                    |
            [Chunking & Embedding]
                    ↓
          +---------------------+
          | Vector DB (Qdrant)  |
          +---------------------+
                    ↑
           [Similarity Search]
                    ↑
         Query ← GPT-4o → Evaluation
                    ↑
          OCR-extracted answer text# 🧠 AI-Based UPSC Answer Sheet Evaluation with RAG

This project implements an end-to-end AI pipeline to evaluate scanned handwritten UPSC answer sheets using GPT-4o and a Retrieval-Augmented Generation (RAG) system grounded in an official SOP Evaluation PDF.



### 🔄 Evaluation Pipeline

1. OCR Extraction: Uses GPT-4o Vision to extract handwritten Hindi text from uploaded answer sheet.

2. Segmentation: Separates the answer into Introduction, Body, Conclusion.

3. Context Retrieval: Retrieves SOP chunks relevant to each section using a vector search on embedded SOP.

4. LLM-based Evaluation:

    a. Generates feedback.

    b. Assigns section-wise marks.

5. Presentation Evaluation: Generates feedback on handwriting quality and layout.

6. PDF Report Generation: Outputs a downloadable PDF report using reportlab.


