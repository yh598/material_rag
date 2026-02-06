# Apparel Material Palette RAG (backend + UI)

This repo builds a lightweight Retrieval Augmented Generation (RAG) app over an apparel BOM / materials dataset.
It includes:
- FastAPI backend with `/chat` endpoint
- Simple retriever (TF-IDF cosine similarity) over key material fields
- Streamlit UI chat app that calls the backend

## Quick start

1) Create a virtual env and install deps

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) Put secrets in `.env`

Copy `.env.example` to `.env` and set:
- AI_GATEWAY_BASE_URL
- AI_GATEWAY_API_KEY
- AI_GATEWAY_MODEL

IMPORTANT: do not commit `.env`.

3) Place the CSV

Set `PALETTE_CSV_PATH` in `.env` to the dataset path.
If you keep the CSV in the repo root, use:

`PALETTE_CSV_PATH=apparel_material_cluster_sample.csv`

4) Run the backend

```bash
cd backend
uvicorn app:app --reload --port 8000
```

5) Run the UI (new terminal)

```bash
cd ui
streamlit run app.py --server.port 8501
```

Open the Streamlit URL printed in your terminal.

## What is indexed

We build one "document" per row using a handful of columns (material name, item description, content, benefits, supplier, etc).
You can add or remove fields in `backend/rag/index.py`.

## Security notes

- Keep your API key in environment variables or `.env`, never in source control.
- If you received a key in a file, load it at runtime and avoid pasting it into chat logs.
