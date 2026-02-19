import os
import json
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.rag.index import PaletteIndex
from backend.llm.client import ChatLLM

load_dotenv()

app = FastAPI(title="Apparel Material Palette RAG")

# Lazy singletons
_INDEX: PaletteIndex | None = None
_LLM: ChatLLM | None = None
_VOTE_LOCK = Lock()


class ChatRequest(BaseModel):
    message: str
    top_k: int = 6


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    recommendations: list[dict] = []
    mode: str = "llm"


class VoteRequest(BaseModel):
    item_id: str
    vote: str


def _is_cost_query(text: str) -> bool:
    q = (text or "").lower()
    keywords = [
        "cost",
        "price",
        "priced",
        "cheap",
        "cheapest",
        "low-cost",
        "lowest",
        "budget",
        "affordable",
    ]
    return any(k in q for k in keywords)


def _llm_configured() -> bool:
    return bool(
        os.getenv("AI_GATEWAY_BASE_URL", "").strip()
        and os.getenv("AI_GATEWAY_API_KEY", "").strip()
    )


def _fallback_answer_from_hits(question: str, hits: list[dict]) -> str:
    if not hits:
        return (
            "I could not find matching rows for your question in the indexed dataset. "
            "Try adding material family, benefits, supplier, or usage details."
        )

    top = hits[: min(3, len(hits))]
    lines = [
        "LLM generation is unavailable, so this is a retrieval-only recommendation summary.",
        f"Question: {question}",
        "",
        "Top matching rows:",
    ]
    for h in top:
        lines.append(
            f"- row_id={h['row_id']} | score={h['score']:.4f} | {h.get('material', 'Unknown material')}"
        )
        snippet = h.get("snippet", "").strip()
        if snippet:
            lines.append(f"  {snippet}")

    lines.append("")
    lines.append(
        "To enable generated narrative answers, set AI_GATEWAY_BASE_URL and AI_GATEWAY_API_KEY."
    )
    return "\n".join(lines)


def _to_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _weight_bucket(fields: dict) -> str:
    gsm = _to_float(fields.get("WEIGHT_GRAMS_PER_SQUARE_METER"))
    if gsm is not None:
        if gsm <= 120:
            return "Lightweight"
        if gsm <= 220:
            return "Medium"
        return "Heavy"

    text = " ".join(
        [
            str(fields.get("MATERIAL_BENEFITS_NM", "")),
            str(fields.get("MATERIAL_INTENT_DESCRIPTION", "")),
            str(fields.get("MATERIAL_CONTENT", "")),
        ]
    ).lower()
    if any(k in text for k in ["light", "mesh", "breathable", "cooling"]):
        return "Lightweight"
    if any(k in text for k in ["reinforcement", "protect", "abrasion"]):
        return "Medium"
    return "Medium"


def _strength_bucket(fields: dict) -> str:
    text = " ".join(
        [
            str(fields.get("MATERIAL_BENEFITS_NM", "")),
            str(fields.get("MATERIAL_INTENT_DESCRIPTION", "")),
        ]
    ).lower()
    if any(k in text for k in ["support", "stability", "propulsion", "reinforc"]):
        return "Stable"
    if any(k in text for k in ["breath", "light", "mesh", "cool"]):
        return "Flexible"
    if any(k in text for k in ["durable", "abrasion", "protection", "trail"]):
        return "Durable"
    return "Balanced"


def _durability_bucket(fields: dict) -> str:
    text = " ".join(
        [
            str(fields.get("MATERIAL_BENEFITS_NM", "")),
            str(fields.get("MATERIAL_INTENT_DESCRIPTION", "")),
            str(fields.get("MATL_COMMENT", "")),
        ]
    ).lower()
    if any(k in text for k in ["abrasion", "durable", "protection", "trail", "reinforc"]):
        return "High"
    if any(k in text for k in ["light", "mesh", "cooling"]):
        return "Medium"
    return "Medium"


def _format_cost(fields: dict) -> str:
    raw = fields.get("LATEST_PRICE_PER_UOM") or fields.get("PRICE_PER_UOM")
    uom = fields.get("PRICE_UOM") or "uom"
    if raw:
        value = _to_float(raw)
        if value is not None:
            return f"${value:.2f} / {uom}"
        return f"{raw} / {uom}"
    return "N/A"


def _recommendation_from_hit(hit: dict) -> dict:
    fields = hit.get("fields", {})
    row_id = int(hit.get("row_id", 0))

    material_number = str(fields.get("PCX_MATL_NBR", "")).strip()
    supplied_material_id = str(fields.get("SUPPLIED_MATERIAL_ID", "")).strip()
    rec_key = material_number or supplied_material_id or f"row{row_id}"

    name = (
        fields.get("MATL_ITM_DESC")
        or fields.get("SUPPLEMENTAL_MATERIAL_NM")
        or fields.get("MATERIAL_FAMILY_NM")
        or hit.get("material", "Material")
    )
    type_name = (
        fields.get("MATERIAL_FAMILY_NM")
        or fields.get("END_USE_NM")
        or "Material"
    )
    supplier = fields.get("SUPLR_LCTN_NM") or fields.get("VENDOR_CD") or "Unknown supplier"
    sustainability = fields.get("SUSTAINABILITY_RANKING") or "Unranked"
    description = (
        fields.get("MATERIAL_INTENT_DESCRIPTION")
        or fields.get("MATERIAL_BENEFITS_NM")
        or fields.get("MATL_COMMENT")
        or hit.get("snippet", "")
    )

    return {
        "id": f"row_{row_id}_{rec_key}",
        "row_id": row_id,
        "score": int(round(float(hit.get("score", 0.0)) * 100)),
        "name": str(name)[:180],
        "type": str(type_name)[:120],
        "division": fields.get("SEGMENT") or "Performance",
        "department": fields.get("DIMENSION") or fields.get("FOP") or "Running",
        "category": fields.get("SILHOUETTE_TYPE_DESCRIPTION") or fields.get("END_USE_NM") or "General",
        "supplier": str(supplier)[:180],
        "sustainability": str(sustainability)[:80],
        "cost": _format_cost(fields),
        "strength": _strength_bucket(fields),
        "weight": _weight_bucket(fields),
        "durability": _durability_bucket(fields),
        "description": str(description)[:320],
        "source": "rag",
    }


def _cost_from_hit(hit: dict) -> float | None:
    fields = hit.get("fields", {})
    return _to_float(fields.get("LATEST_PRICE_PER_UOM")) or _to_float(fields.get("PRICE_PER_UOM"))


def _vote_store_path() -> Path:
    configured = os.getenv("VOTE_STORE_PATH", "").strip()
    if configured:
        p = Path(configured)
        if not p.is_absolute():
            return Path(__file__).resolve().parent.parent / p
        return p
    return Path(__file__).resolve().parent.parent / "votes.json"


def _load_votes() -> dict[str, str]:
    p = _vote_store_path()
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_votes(votes: dict[str, str]) -> None:
    p = _vote_store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(votes, f, indent=2)


def _vote_counts(votes: dict[str, str]) -> dict:
    approved = sum(1 for v in votes.values() if v == "approved")
    disapproved = sum(1 for v in votes.values() if v == "disapproved")
    return {
        "approved": approved,
        "disapproved": disapproved,
        "total": len(votes),
    }


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
    return {
        "ok": True,
        "llm_configured": _llm_configured(),
        "palette_csv_path": os.getenv("PALETTE_CSV_PATH", "apparel_material_cluster_sample.csv"),
    }


@app.get("/votes")
def get_votes():
    with _VOTE_LOCK:
        votes = _load_votes()
    return {"ok": True, "votes": votes, "counts": _vote_counts(votes)}


@app.post("/votes")
def set_vote(req: VoteRequest):
    normalized_vote = str(req.vote).strip().lower()
    if normalized_vote not in {"approved", "disapproved", "pending"}:
        raise HTTPException(status_code=400, detail="vote must be approved, disapproved, or pending")

    item_id = str(req.item_id).strip()
    if not item_id:
        raise HTTPException(status_code=400, detail="item_id is required")

    with _VOTE_LOCK:
        votes = _load_votes()
        if normalized_vote == "pending":
            votes.pop(item_id, None)
        else:
            votes[item_id] = normalized_vote
        _save_votes(votes)

    return {"ok": True, "votes": votes, "counts": _vote_counts(votes)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    idx = get_index()
    top_k = max(1, int(req.top_k))
    search_k = top_k
    if _is_cost_query(req.message):
        search_k = max(top_k * 6, 40)

    hits = idx.search(req.message, top_k=search_k)
    if _is_cost_query(req.message):
        priced = [h for h in hits if _cost_from_hit(h) is not None]
        unpriced = [h for h in hits if _cost_from_hit(h) is None]
        priced.sort(key=lambda h: (_cost_from_hit(h), -float(h.get("score", 0.0))))
        unpriced.sort(key=lambda h: -float(h.get("score", 0.0)))
        hits = priced + unpriced
    hits = hits[:top_k]

    context_blocks = []
    citations = []
    recommendations = []
    for h in hits:
        citations.append({
            "row_id": int(h["row_id"]),
            "score": float(h["score"]),
            "material": h.get("material", ""),
            "snippet": h.get("snippet", ""),
        })
        recommendations.append(_recommendation_from_hit(h))
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

    mode = "llm"
    try:
        llm = get_llm()
        answer = llm.chat(system=system, user=user)
    except Exception:
        answer = _fallback_answer_from_hits(req.message, hits)
        mode = "retrieval_fallback"

    return ChatResponse(
        answer=answer,
        citations=citations,
        recommendations=recommendations,
        mode=mode,
    )
