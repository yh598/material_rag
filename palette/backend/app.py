import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.rag.index import PaletteIndex
from backend.llm.client import ChatLLM

load_dotenv()

app = FastAPI(title="Apparel Material Palette RAG")

# Lazy singletons
_INDEX: PaletteIndex | None = None
_LLM: ChatLLM | None = None


class ChatRequest(BaseModel):
    message: str
    top_k: int = 6


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]


def get_index() -> PaletteIndex:
    global _INDEX
    if _INDEX is None:
        csv_path = os.getenv("PALETTE_CSV_PATH", "apparel_material_cluster_sample.csv")
        _INDEX = PaletteIndex.from_csv(csv_path)
    return _INDEX


def get_llm() -> ChatLLM:
    global _LLM
    if _LLM is None:
        _LLM = ChatLLM.from_env()
    return _LLM


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    idx = get_index()
    llm = get_llm()

    hits = idx.search(req.message, top_k=req.top_k)

    context_blocks = []
    citations = []
    for h in hits:
        citations.append({
            "row_id": int(h["row_id"]),
            "score": float(h["score"]),
            "material": h.get("material", ""),
            "snippet": h.get("snippet", "")
        })
        context_blocks.append(
            f"[row_id={h['row_id']}] {h.get('material','')}\n{h.get('fulltext','')}"
        )

    system = (
        "You are an apparel materials assistant. Use ONLY the provided context to answer.\n"
        "If the answer isn't in the context, say you don't know and suggest what fields to search.\n"
        "When you cite, refer to row_id(s)."
    )
    user = (
        f"Question: {req.message}\n\n"
        f"Context:\n\n" + "\n\n---\n\n".join(context_blocks)
    )

    answer = llm.chat(system=system, user=user)

    return ChatResponse(answer=answer, citations=citations)
