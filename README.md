# RAG Document Q&A Chatbot

Upload a PDF, ask questions about it, get direct answers. I tested it mostly on the Apple 10-K 2025 but it works on any text-heavy PDF.

The core idea is RAG (Retrieval Augmented Generation) — instead of feeding the whole document to the LLM (which won't fit), you find the relevant bits first using vector search, then pass only those to the model. Built entirely in Google Colab with pretrained models.

---

## How it works

1. **Extract** — reads all text from the PDF using PyMuPDF
2. **Clean** — strips page numbers, headers, footers and junk characters
3. **Chunk** — splits text into overlapping 800-character passages
4. **Embed** — converts every chunk into a vector using all-MiniLM-L6-v2
5. **Index** — stores vectors in FAISS for fast similarity search
6. **Retrieve** — on each question, fetches the top 3 most relevant chunks
7. **Generate** — passes chunks + conversation history to TinyLlama for the answer

It also keeps track of previous questions in the session so follow-ups like "how does that compare to last year?" work without you repeating the context.

---

## Models used

| Component | Model |
|---|---|
| Embeddings | all-MiniLM-L6-v2 (384-dim) |
| LLM | TinyLlama-1.1B-Chat |
| Vector search | FAISS |

TinyLlama is small enough to run on Colab free tier and the license is Apache 2.0. Not the most powerful model but good enough for factual extraction from documents.

---

## Tech stack

- Python
- PyTorch
- HuggingFace Transformers
- LangChain LCEL
- FAISS
- Sentence Transformers
- PyMuPDF
- Pydantic
- Gradio
- Google Colab

---

## How to run

**Colab (recommended)**

- Open `rag_document_qna.ipynb`
- Switch runtime to T4 GPU
- Run cells top to bottom
- Last cell opens a Gradio UI where you can upload your own PDF and start asking questions

**Local**

```
pip install -r requirements.txt
```

Then open the notebook in Jupyter and run all cells. You don't need a GPU but it'll be slow on CPU.

---

## Requirements

```
torch
transformers
accelerate
sentence-transformers
faiss-cpu
langchain
langchain-community
langchain-huggingface
langchain-text-splitters
PyMuPDF
pydantic
gradio
```

---

## Project structure

```
rag-document-qna/
│
├── rag_document_qna.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## What I learned

TinyLlama has a 2048 token context limit which sounds fine until you realize a full 10-K is 500k+ characters. Getting the chunk size right so financial sentences don't get cut in the middle was more annoying than expected. Too small and you lose context, too large and you blow the window.

The other thing that tripped me up was the prompt format. TinyLlama was fine-tuned with a specific chat template and if you don't use it exactly the answers get weird. Once I switched to the proper `<|system|>` / `<|user|>` / `<|assistant|>` format the quality jumped a lot.

---

## Author

Vipin Gupta
vipingupta.dev@gmail.com
github.com/vipingupta-dev
