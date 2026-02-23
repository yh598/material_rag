import os
import json
import re
import time
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

try:
    from scripts.generate_large_synthetic_dataset import (
        all_filter_combos,
        build_dataset as build_synthetic_dataset,
        rows_have_full_combo_coverage,
    )
except Exception:
    all_filter_combos = None
    build_synthetic_dataset = None
    rows_have_full_combo_coverage = None

load_dotenv()

st.set_page_config(page_title="Materials Recommendation", layout="wide")

FALLBACK_MATERIALS = [
    {
        "id": "m001",
        "name": "Flyknit Mesh",
        "type": "Upper Material",
        "category": "Marathon",
        "department": "Running",
        "division": "Performance",
        "supplier": "TextilePro Inc.",
        "sustainability": "88%",
        "cost": "$12.50 / meter",
        "score": 96,
        "strength": "Flexible",
        "weight": "Lightweight",
        "durability": "High",
        "description": "Optimized breathability and weight reduction for high-performance running shoes.",
    },
    {
        "id": "m002",
        "name": "EVA Foam Compound",
        "type": "Midsole",
        "category": "Marathon",
        "department": "Running",
        "division": "Performance",
        "supplier": "CushionTech Ltd.",
        "sustainability": "74%",
        "cost": "$8.75 / kg",
        "score": 93,
        "strength": "Durable",
        "weight": "Lightweight",
        "durability": "High",
        "description": "Industry-standard cushioning with excellent energy return and long-term comfort.",
    },
    {
        "id": "m003",
        "name": "Carbon Fiber Plate",
        "type": "Support Structure",
        "category": "Marathon",
        "department": "Running",
        "division": "Performance",
        "supplier": "AeroComposite",
        "sustainability": "62%",
        "cost": "$45.00 / piece",
        "score": 90,
        "strength": "Rigid",
        "weight": "Lightweight",
        "durability": "High",
        "description": "Enhances toe-off efficiency and running economy for sustained speed over long distances.",
    },
    {
        "id": "m004",
        "name": "Recycled Ripstop Nylon",
        "type": "Shell Layer",
        "category": "Trail",
        "department": "Outdoor",
        "division": "Performance",
        "supplier": "GreenWeave",
        "sustainability": "91%",
        "cost": "$9.20 / meter",
        "score": 89,
        "strength": "Stable",
        "weight": "Medium",
        "durability": "High",
        "description": "Abrasion resistance and weather protection using recycled feedstock.",
    },
    {
        "id": "m005",
        "name": "Thermo Wool Blend",
        "type": "Inner Layer",
        "category": "Winter",
        "department": "Lifestyle",
        "division": "Apparel",
        "supplier": "NordicTextiles",
        "sustainability": "79%",
        "cost": "$14.10 / meter",
        "score": 86,
        "strength": "Comfort",
        "weight": "Medium",
        "durability": "Medium",
        "description": "Soft thermal retention layer suitable for cooler weather footwear systems.",
    },
    {
        "id": "m006",
        "name": "Bio-TPU Outsole",
        "type": "Outsole",
        "category": "Training",
        "department": "Running",
        "division": "Performance",
        "supplier": "EcoPolymers",
        "sustainability": "84%",
        "cost": "$6.90 / kg",
        "score": 85,
        "strength": "Stable",
        "weight": "Medium",
        "durability": "High",
        "description": "Bio-based TPU compound balancing grip, resilience, and reduced fossil content.",
    },
]


def _read_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sustainability_pct(grade: str, seed: int) -> str:
    buckets = {"A": (88, 98), "B": (74, 89), "C": (60, 78)}
    low, high = buckets.get(grade, (70, 85))
    value = low + (seed % (high - low + 1))
    return f"{value}%"


def _pick(values: list[str], seed: int) -> str:
    return values[seed % len(values)]


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _role_candidates_for_family(family: str) -> list[str]:
    by_family = {
        "Upper": ["Upper Base", "Upper Overlay", "Upper Reinforcement"],
        "Lining": ["Lining", "Inner Lining"],
        "Midsole": ["Primary Midsole", "Midsole Insert"],
        "Midsole Plate": ["Plate", "Propulsion Plate"],
        "Outsole": ["Outsole Forefoot", "Outsole Heel", "Outsole"],
        "Insole": ["Insole", "Heel Pad Insole"],
        "Reinforcement": ["Toe Protection", "Heel Counter", "Midfoot Support"],
        "Lace": ["Lace", "Lace System"],
    }
    return by_family.get(family, [family or "Material"])


def _unit_for_family(family: str) -> str:
    by_family = {
        "Upper": "meter",
        "Lining": "meter",
        "Midsole": "kg",
        "Midsole Plate": "piece",
        "Outsole": "kg",
        "Insole": "pair",
        "Reinforcement": "piece",
        "Lace": "pair",
    }
    return by_family.get(family, "unit")


@st.cache_data(show_spinner=False)
def build_synthetic_materials() -> list[dict]:
    root = Path(__file__).resolve().parent.parent
    default_cached = root / "synthetic_materials_large.json"
    target_size = _safe_int(os.getenv("SYNTHETIC_TARGET_SIZE", "12000"), 12000)
    target_size = max(300, target_size)
    min_per_combo = _safe_int(os.getenv("SYNTHETIC_MIN_PER_COMBO", "6"), 6)
    min_per_combo = max(1, min_per_combo)

    expected_combos = []
    try:
        products = _read_json(root / "products.json")
        if callable(all_filter_combos):
            expected_combos = all_filter_combos(products)
    except Exception:
        expected_combos = []

    def cache_is_valid(rows: list[dict]) -> bool:
        if not isinstance(rows, list) or not rows:
            return False
        first = rows[0] if isinstance(rows[0], dict) else {}
        required = {"id", "name", "division", "department", "category", "score"}
        if not required.issubset(set(first.keys())):
            return False
        if expected_combos and callable(rows_have_full_combo_coverage):
            return rows_have_full_combo_coverage(rows, expected_combos)
        return True

    candidates = []
    env_data_path = os.getenv("SYNTHETIC_DATA_PATH", "").strip()
    if env_data_path:
        candidate = Path(env_data_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidates.append(candidate)
    candidates.append(default_cached)

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            cached_rows = _read_json(candidate)
            if cache_is_valid(cached_rows):
                return cached_rows
        except Exception:
            continue

    try:
        if not callable(build_synthetic_dataset):
            return FALLBACK_MATERIALS
        rows = build_synthetic_dataset(root=root, target_size=target_size, min_per_combo=min_per_combo)
        if cache_is_valid(rows):
            try:
                with default_cached.open("w", encoding="utf-8") as f:
                    json.dump(rows, f, indent=2)
            except Exception:
                pass
            return rows
    except Exception:
        pass

    return FALLBACK_MATERIALS


MATERIALS = build_synthetic_materials()

if "votes" not in st.session_state:
    st.session_state.votes = {}
if "assistant_answer" not in st.session_state:
    st.session_state.assistant_answer = ""
if "assistant_citations" not in st.session_state:
    st.session_state.assistant_citations = []
if "assistant_error" not in st.session_state:
    st.session_state.assistant_error = ""
if "assistant_mode" not in st.session_state:
    st.session_state.assistant_mode = ""
if "assistant_query" not in st.session_state:
    st.session_state.assistant_query = ""
if "last_vote_action" not in st.session_state:
    st.session_state.last_vote_action = ""
if "rag_recommendations" not in st.session_state:
    st.session_state.rag_recommendations = []
if "backend_votes" not in st.session_state:
    st.session_state.backend_votes = {}
if "vote_error" not in st.session_state:
    st.session_state.vote_error = ""
if "vote_counts" not in st.session_state:
    st.session_state.vote_counts = {"approved": 0, "disapproved": 0, "total": 0}

st.markdown(
    """
    <style>
    .stApp { background: #040507; color: #eef2ff; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { max-width: 1600px; padding-top: 0.9rem; padding-bottom: 1rem; }

    [data-testid="stSidebar"] { background: #070a10; border-right: 1px solid #1c212d; }
    [data-testid="stSidebar"] .block-container { padding-top: 0.7rem; }

    .topbar {
        display: flex; justify-content: space-between; align-items: center;
        border: 1px solid #1d2330; border-radius: 12px;
        background: #070b12; padding: 0.65rem 0.9rem; margin-bottom: 0.65rem;
    }
    .title-main { font-size: 0.95rem; font-weight: 700; color: #f3f7ff; }
    .title-sub { font-size: 0.7rem; color: #8894ab; }
    .metrics { display: flex; gap: 1rem; color: #a9b2c4; font-size: 0.72rem; }
    .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 5px; }

    .section-box {
        border: 1px solid #1d2432;
        background: #0a0f19;
        border-radius: 10px;
        padding: 0.55rem 0.65rem;
        margin-bottom: 0.5rem;
    }
    .section-title { font-size: 0.77rem; font-weight: 700; color: #e6edff; margin-bottom: 0.25rem; }
    .section-sub { font-size: 0.67rem; color: #8691a7; margin-bottom: 0.45rem; }
    .field-label { font-size: 0.62rem; color: #7f8aa2; margin-top: 0.35rem; margin-bottom: 0.2rem; text-transform: uppercase; }

    .recommend-header {
        border: 1px solid #1f2634;
        background: #080d16;
        border-radius: 10px;
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.45rem;
    }
    .recommend-title { font-size: 0.82rem; font-weight: 700; color: #f0f5ff; margin-bottom: 0.35rem; }
    .pills { display: flex; gap: 0.33rem; }
    .pill {
        border: 1px solid #2b3243;
        background: #111827;
        color: #c3cee2;
        border-radius: 999px;
        padding: 0.09rem 0.52rem;
        font-size: 0.62rem;
    }

    .card {
        border: 1px solid #232b3a;
        background: linear-gradient(180deg, #101723 0%, #0a0f18 100%);
        border-radius: 10px;
        padding: 0.62rem;
        min-height: 252px;
    }
    .card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 0.45rem; margin-bottom: 0.38rem; }
    .card-name { font-size: 0.8rem; font-weight: 700; color: #f2f6ff; }
    .card-type { font-size: 0.66rem; color: #98a2b7; }
    .badge {
        border: 1px solid #284f8d;
        background: #0a1d39;
        color: #90bdf7;
        border-radius: 999px;
        font-size: 0.58rem;
        padding: 0.08rem 0.45rem;
        white-space: nowrap;
    }
    .kv-grid { display: grid; grid-template-columns: auto 1fr; gap: 0.16rem 0.36rem; margin-top: 0.3rem; }
    .k { font-size: 0.62rem; color: #77829a; }
    .v { font-size: 0.62rem; color: #dde6f6; font-weight: 600; }
    .desc { margin-top: 0.38rem; font-size: 0.62rem; color: #9ca8bf; line-height: 1.3; min-height: 58px; }

    .stButton > button {
        border-radius: 7px;
        border: 1px solid #2a3344;
        background: #0f1522;
        color: #d3ddef;
        font-size: 0.67rem;
        height: 1.85rem;
    }
    .stButton > button:hover { border-color: #3a4a63; background: #141d2e; }

    .ai-panel {
        border: 1px solid #1e2533;
        border-radius: 10px;
        background: #070c14;
        padding: 0.58rem 0.7rem;
        margin-bottom: 0.5rem;
    }
    .ai-title { font-size: 0.76rem; font-weight: 700; color: #edf2ff; margin-bottom: 0.3rem; }

    div[data-baseweb="select"] > div { min-height: 34px; background: #0d1320; border-color: #253043; }
    .stTextInput input, .stTextArea textarea {
        background: #0d1320 !important;
        color: #edf2ff !important;
        border: 1px solid #253043 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def query_backend(url: str, prompt: str, top_k: int) -> tuple[str, list[dict], list[dict], str, str]:
    endpoint = f"{url.rstrip('/')}/chat"
    payload = {"message": prompt, "top_k": int(top_k)}
    retry_statuses = {429, 502, 503, 504}
    max_attempts = 4

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(endpoint, json=payload, timeout=120)
        except requests.exceptions.RequestException as exc:
            if attempt < max_attempts:
                time.sleep(min(8.0, 1.2 * (2 ** (attempt - 1))))
                continue
            return "", [], [], f"Backend request failed: {exc}", "error"

        if resp.status_code in retry_statuses:
            retry_after = _safe_float(resp.headers.get("Retry-After"), 0.0)
            wait_seconds = retry_after if retry_after > 0 else min(8.0, 1.2 * (2 ** (attempt - 1)))
            if attempt < max_attempts:
                time.sleep(wait_seconds)
                continue
            detail = ""
            try:
                detail = (resp.text or "").strip()
            except Exception:
                detail = ""
            msg = (
                f"Backend is rate-limiting requests (HTTP {resp.status_code}) "
                f"after {max_attempts} retries. Please wait 20-40 seconds and try again."
            )
            if detail:
                msg = f"{msg} Detail: {detail[:180]}"
            return "", [], [], msg, "error"

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            return "", [], [], f"Backend request failed: {exc}", "error"

        data = resp.json()
        return (
            data.get("answer", ""),
            data.get("citations", []),
            data.get("recommendations", []),
            "",
            data.get("mode", "llm"),
        )

    return "", [], [], "Backend request failed: retry attempts exhausted.", "error"


def normalize_backend_url(raw_url: str) -> str:
    value = (raw_url or "").strip()
    if not value:
        return "http://127.0.0.1:8000"
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"http://{value}"


@st.cache_data(ttl=5, show_spinner=False)
def fetch_votes(url: str) -> tuple[dict, dict, str]:
    try:
        resp = requests.get(f"{url.rstrip('/')}/votes", timeout=6)
        resp.raise_for_status()
        data = resp.json()
        return data.get("votes", {}), data.get("counts", {}), ""
    except Exception as exc:
        return {}, {}, f"Vote sync unavailable: {exc}"


def submit_vote(url: str, item_id: str, vote: str) -> tuple[dict, dict, str]:
    try:
        resp = requests.post(
            f"{url.rstrip('/')}/votes",
            json={"item_id": item_id, "vote": vote},
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()
        fetch_votes.clear()
        return data.get("votes", {}), data.get("counts", {}), ""
    except Exception as exc:
        return {}, {}, f"Vote update failed: {exc}"


STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "show", "what", "which", "are", "into", "your",
    "about", "under", "best", "option", "options", "materials", "material", "recommend", "recommended",
    "give", "need", "want", "shoe", "shoes",
}


def tokenize_query(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]


def material_search_blob(item: dict) -> str:
    fields = [
        item.get("name", ""),
        item.get("type", ""),
        item.get("product_name", ""),
        item.get("product_code", ""),
        item.get("target_consumer", ""),
        item.get("season", ""),
        item.get("category", ""),
        item.get("department", ""),
        item.get("division", ""),
        item.get("supplier", ""),
        item.get("strength", ""),
        item.get("weight", ""),
        item.get("durability", ""),
        item.get("description", ""),
    ]
    return " ".join(str(x) for x in fields).lower()


def query_relevance(item: dict, query_tokens: list[str]) -> tuple[float, list[str]]:
    if not query_tokens:
        return 0.0, []

    blob = material_search_blob(item)
    name_type = f"{item.get('name', '')} {item.get('type', '')}".lower()
    matched = []
    score = 0.0
    for token in query_tokens:
        if token in name_type:
            score += 3.0
            matched.append(token)
        elif token in blob:
            score += 1.2
            matched.append(token)

    score += float(item.get("score", 0)) / 500.0
    return score, matched[:4]


def rationale_line(item: dict, matched_terms: list[str], query_text: str) -> str:
    if matched_terms:
        return f"Why: matches query terms {', '.join(matched_terms)}."
    if query_text.strip():
        return f"Why: ranked by overall fit and score for '{query_text[:60]}'."
    return "Why: high baseline performance score."


def run_assistant_query(prompt: str, backend_ok: bool, backend_status: str, backend_url: str, top_k: int) -> None:
    st.session_state.assistant_query = prompt
    if not backend_ok:
        st.session_state.assistant_error = backend_status or "Backend unavailable."
        st.session_state.assistant_mode = "error"
        return

    answer, citations, recommendations, error, mode = query_backend(backend_url, prompt, top_k)
    if error:
        # Preserve prior successful state on transient backend failures.
        st.session_state.assistant_error = error
        st.session_state.assistant_mode = mode
        return

    st.session_state.assistant_answer = answer
    st.session_state.assistant_citations = citations
    st.session_state.rag_recommendations = recommendations
    st.session_state.assistant_error = ""
    st.session_state.assistant_mode = mode


@st.cache_data(ttl=6, show_spinner=False)
def backend_health(url: str) -> tuple[bool, str]:
    health_url = f"{url.rstrip('/')}/health"
    last_timeout = False
    for timeout_sec in (4, 12):
        try:
            resp = requests.get(health_url, timeout=timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            llm_configured = data.get("llm_configured")
            if llm_configured is True:
                return True, "Backend connected (LLM enabled)"
            if llm_configured is False:
                return True, "Backend connected (retrieval mode)"
            return True, "Backend connected"
        except requests.exceptions.Timeout:
            last_timeout = True
            continue
        except Exception as exc:
            return False, f"Backend unavailable: {exc}"

    if last_timeout:
        return (
            False,
            "Backend is waking up (cold start on free plan). Submit your question again in 30-60 seconds.",
        )
    return False, "Backend unavailable"


def matches_filters(
    item: dict,
    division: str,
    department: str,
    category: str,
    product_name: str,
    min_score: int,
) -> bool:
    item_division = str(item.get("division", ""))
    item_department = str(item.get("department", ""))
    item_category = str(item.get("category", ""))
    item_product = str(item.get("product_name", ""))
    item_score = _safe_int(item.get("score"), 0)

    if division != "All" and item_division.lower() != division.lower():
        return False
    if department != "All" and item_department.lower() != department.lower():
        return False
    if category != "All" and item_category.lower() != category.lower():
        return False
    if product_name != "All" and item_product.lower() != product_name.lower():
        return False
    return item_score >= min_score


with st.sidebar:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Product Selection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Choose criteria to view material recommendations</div>', unsafe_allow_html=True)

    universe_materials = MATERIALS + (st.session_state.get("rag_recommendations") or [])
    divisions = ["All"] + sorted({str(m.get("division", "")) for m in universe_materials if str(m.get("division", "")).strip()})
    departments = ["All"] + sorted({str(m.get("department", "")) for m in universe_materials if str(m.get("department", "")).strip()})
    categories = ["All"] + sorted({str(m.get("category", "")) for m in universe_materials if str(m.get("category", "")).strip()})

    st.markdown('<div class="field-label">Division</div>', unsafe_allow_html=True)
    selected_division = st.selectbox("Division", divisions, index=0, label_visibility="collapsed")
    st.markdown('<div class="field-label">Department</div>', unsafe_allow_html=True)
    selected_department = st.selectbox("Department", departments, index=0, label_visibility="collapsed")
    st.markdown('<div class="field-label">Category</div>', unsafe_allow_html=True)
    selected_category = st.selectbox("Category", categories, index=0, label_visibility="collapsed")
    min_score = st.slider("Min score", 60, 99, 80, label_visibility="collapsed")

    filtered_for_products = [
        m
        for m in universe_materials
        if (selected_division == "All" or str(m.get("division", "")).lower() == selected_division.lower())
        and (selected_department == "All" or str(m.get("department", "")).lower() == selected_department.lower())
        and (selected_category == "All" or str(m.get("category", "")).lower() == selected_category.lower())
        and _safe_int(m.get("score", 0), 0) >= min_score
    ]
    product_names = sorted(
        {
            str(m.get("product_name", "")).strip()
            for m in filtered_for_products
            if str(m.get("product_name", "")).strip()
        }
    )
    if not product_names:
        product_names = sorted(
            {
                str(m.get("product_name", "")).strip()
                for m in universe_materials
                if str(m.get("product_name", "")).strip()
            }
        )

    st.markdown('<div class="field-label">Product</div>', unsafe_allow_html=True)
    selected_product = st.selectbox(
        "Product",
        ["All"] + product_names,
        index=0,
        label_visibility="collapsed",
    )
    max_cards = st.slider("Max cards", 12, 180, 36, 12, label_visibility="collapsed")
    st.caption(f"Minimum Score: {min_score}")
    st.caption(f"Cards Shown: {max_cards}")

    st.markdown('</div>', unsafe_allow_html=True)

    sidebar_pool = st.session_state.get("rag_recommendations") or MATERIALS
    if st.session_state.get("rag_recommendations"):
        sidebar_pool_msg = f"{len(sidebar_pool)} RAG recommendations from latest question"
    else:
        combo_count = len(
            {
                (
                    str(m.get("division", "")),
                    str(m.get("department", "")),
                    str(m.get("category", "")),
                )
                for m in sidebar_pool
                if str(m.get("division", "")).strip()
                and str(m.get("department", "")).strip()
                and str(m.get("category", "")).strip()
            }
        )
        sidebar_pool_msg = (
            f"{len(sidebar_pool)} synthetic recommendations from JSON source files "
            f"({combo_count} filter combinations)"
        )
    st.markdown(
        f'<div class="section-box"><div class="section-title">All Recommendations</div><div class="section-sub">{sidebar_pool_msg}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Materials AI Assistant</div>', unsafe_allow_html=True)

    backend_url = st.text_input("Backend URL", value=os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
    resolved_backend_url = normalize_backend_url(backend_url)
    backend_ok, backend_status = backend_health(resolved_backend_url)
    st.caption(backend_status)
    synced_votes, synced_counts, vote_err = fetch_votes(resolved_backend_url)
    if isinstance(synced_votes, dict) and synced_votes:
        st.session_state.backend_votes = synced_votes
    if isinstance(synced_counts, dict) and synced_counts:
        st.session_state.vote_counts = synced_counts
    st.session_state.vote_error = vote_err if vote_err else ""
    top_k = st.slider("Top K", 3, 12, 6)
    assistant_prompt = st.text_area("Ask about materials", placeholder="Ask about materials...", height=80)

    if st.button("Generate recommendations", use_container_width=True):
        prompt = assistant_prompt.strip() or (
            f"Recommend materials for division={selected_division}, "
            f"department={selected_department}, category={selected_category}, "
            f"product={selected_product}. Cite row_id values."
        )
        run_assistant_query(
            prompt=prompt,
            backend_ok=backend_ok,
            backend_status=backend_status,
            backend_url=resolved_backend_url,
            top_k=top_k,
        )

    st.markdown('</div>', unsafe_allow_html=True)


quick_prompt = st.chat_input("Ask demo question here...")
if quick_prompt:
    run_assistant_query(
        prompt=quick_prompt,
        backend_ok=backend_ok,
        backend_status=backend_status,
        backend_url=resolved_backend_url,
        top_k=top_k,
    )
    st.rerun()


rag_materials = st.session_state.get("rag_recommendations", [])
source_materials = rag_materials if rag_materials else MATERIALS
effective_min_score = min_score if not rag_materials else 0

filtered = [
    m
    for m in source_materials
    if matches_filters(
        m,
        selected_division,
        selected_department,
        selected_category,
        selected_product,
        effective_min_score,
    )
]
rag_filter_fallback = False
if rag_materials and not filtered:
    rag_filter_fallback = True
    source_materials = MATERIALS
    effective_min_score = min_score
    filtered = [
        m
        for m in source_materials
        if matches_filters(
            m,
            selected_division,
            selected_department,
            selected_category,
            selected_product,
            effective_min_score,
        )
    ]

query_text = st.session_state.get("assistant_query", "").strip()
query_tokens = tokenize_query(query_text)
relevance_by_id = {}

if rag_materials and not rag_filter_fallback:
    filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
    for item in filtered:
        _, matched = query_relevance(item, query_tokens)
        relevance_by_id[item["id"]] = (item.get("score", 0), matched)
else:
    if query_tokens:
        scored = []
        for item in filtered:
            rel, matched = query_relevance(item, query_tokens)
            relevance_by_id[item["id"]] = (rel, matched)
            scored.append((item, rel, matched))
        scored.sort(key=lambda t: (t[1], t[0].get("score", 0)), reverse=True)
        if any(rel > 0 for _, rel, _ in scored):
            filtered = [item for item, rel, _ in scored if rel > 0]
        else:
            filtered = [item for item, _, _ in scored]
    else:
        filtered.sort(key=lambda x: x["score"], reverse=True)

visible_materials = filtered[:max_cards]

active_votes = st.session_state.backend_votes if backend_ok else st.session_state.votes
approved = sum(1 for v in active_votes.values() if v == "approved")
disapproved = sum(1 for v in active_votes.values() if v == "disapproved")
pending = max(0, len(source_materials) - approved - disapproved)
source_label = (
    "RAG-driven material table"
    if rag_materials and not rag_filter_fallback
    else "AI-powered material selector for footwear"
)

st.markdown(
    f"""
    <div class="topbar">
        <div>
            <div class="title-main">Materials Recommendation</div>
            <div class="title-sub">{source_label}</div>
        </div>
        <div class="metrics">
            <span><span class="dot" style="background:#22c55e"></span>Approved: {approved}</span>
            <span><span class="dot" style="background:#f59e0b"></span>Pending: {pending}</span>
            <span><span class="dot" style="background:#ef4444"></span>Disapproved: {disapproved}</span>
        </div>
    </div>
    <div class="recommend-header">
        <div class="recommend-title">Recommended Materials</div>
        <div class="pills">
            <span class="pill">Performance</span>
            <span class="pill">Rating</span>
            <span class="pill">Marathon</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.assistant_error:
    st.error(st.session_state.assistant_error)

if st.session_state.vote_error and backend_ok:
    st.caption(st.session_state.vote_error)

if st.session_state.last_vote_action:
    st.caption(st.session_state.last_vote_action)

if rag_filter_fallback:
    st.caption("No RAG rows matched the selected filters. Showing synthetic full-coverage dataset.")

if st.session_state.assistant_answer:
    st.markdown('<div class="ai-panel"><div class="ai-title">AI Summary</div></div>', unsafe_allow_html=True)
    if st.session_state.assistant_mode == "llm":
        st.success("LLM explanation enabled.")
    if st.session_state.assistant_mode == "retrieval_fallback":
        st.warning("LLM response unavailable. Showing retrieval-only fallback summary.")
    st.write(st.session_state.assistant_answer)
    if st.session_state.assistant_citations:
        with st.expander("Retrieved rows"):
            for item in st.session_state.assistant_citations:
                row_id = item.get("row_id", "-")
                material = item.get("material") or "Unknown material"
                score = float(item.get("score", 0.0))
                snippet = item.get("snippet") or "No snippet available."
                st.markdown(f"**row_id={row_id} | score={score:.4f} | {material}**")
                st.caption(snippet)
                st.divider()

if not filtered:
    st.info("No materials matched the selected filters.")
else:
    st.caption(f"Showing {len(visible_materials)} of {len(filtered)} materials")
    view_approved = sum(1 for m in filtered if active_votes.get(m["id"]) == "approved")
    view_disapproved = sum(1 for m in filtered if active_votes.get(m["id"]) == "disapproved")
    st.caption(f"In current view: Approved {view_approved}, Disapproved {view_disapproved}")
    if query_text:
        st.caption(f"Ranked by query relevance: '{query_text}'")
    cols = st.columns(3)
    for idx, item in enumerate(visible_materials):
        with cols[idx % 3]:
            matched_terms = relevance_by_id.get(item["id"], (0.0, []))[1]
            current_vote = active_votes.get(item["id"], "pending")
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-head">
                        <div>
                            <div class="card-name">{item['name']}</div>
                            <div class="card-type">{item['type']}</div>
                        </div>
                        <span class="badge">recommended</span>
                    </div>
                    <div class="kv-grid">
                        <div class="k">Product:</div><div class="v">{item.get('product_name', '-')}</div>
                        <div class="k">Consumer:</div><div class="v">{item.get('target_consumer', '-')}</div>
                        <div class="k">Season:</div><div class="v">{item.get('season', '-')}</div>
                        <div class="k">Supplier:</div><div class="v">{item.get('supplier', '-')}</div>
                        <div class="k">Sustainability:</div><div class="v">{item.get('sustainability', '-')}</div>
                        <div class="k">Cost:</div><div class="v">{item.get('cost', '-')}</div>
                        <div class="k">Strength:</div><div class="v">{item.get('strength', '-')}</div>
                        <div class="k">Weight:</div><div class="v">{item.get('weight', '-')}</div>
                        <div class="k">Durability:</div><div class="v">{item.get('durability', '-')}</div>
                        <div class="k">Status:</div><div class="v">{current_vote}</div>
                    </div>
                    <div class="desc">{item.get('description', '')}</div>
                    <div class="desc">{rationale_line(item, matched_terms, query_text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            left, right = st.columns(2)
            with left:
                if st.button("Approve", key=f"approve_{item['id']}", use_container_width=True):
                    synced_votes, synced_counts, vote_err = submit_vote(
                        resolved_backend_url, item["id"], "approved"
                    )
                    if vote_err:
                        st.session_state.votes[item["id"]] = "approved"
                        st.session_state.vote_error = f"{vote_err} (using local fallback)"
                    else:
                        st.session_state.vote_error = ""
                        if synced_votes:
                            st.session_state.backend_votes = synced_votes
                        if synced_counts:
                            st.session_state.vote_counts = synced_counts
                    st.session_state.last_vote_action = f"{item['name']}: approved"
                    st.rerun()
            with right:
                if st.button("Disapprove", key=f"disapprove_{item['id']}", use_container_width=True):
                    synced_votes, synced_counts, vote_err = submit_vote(
                        resolved_backend_url, item["id"], "disapproved"
                    )
                    if vote_err:
                        st.session_state.votes[item["id"]] = "disapproved"
                        st.session_state.vote_error = f"{vote_err} (using local fallback)"
                    else:
                        st.session_state.vote_error = ""
                        if synced_votes:
                            st.session_state.backend_votes = synced_votes
                        if synced_counts:
                            st.session_state.vote_counts = synced_counts
                    st.session_state.last_vote_action = f"{item['name']}: disapproved"
                    st.rerun()

    table_rows = []
    for item in visible_materials:
        table_rows.append(
            {
                "ID": item.get("id", ""),
                "Product": item.get("product_name", ""),
                "Product Code": item.get("product_code", ""),
                "Consumer": item.get("target_consumer", ""),
                "Season": item.get("season", ""),
                "Material": item.get("name", ""),
                "Type": item.get("type", ""),
                "Division": item.get("division", ""),
                "Department": item.get("department", ""),
                "Category": item.get("category", ""),
                "Supplier": item.get("supplier", ""),
                "Sustainability": item.get("sustainability", ""),
                "Cost": item.get("cost", ""),
                "Score": item.get("score", 0),
                "Vote": active_votes.get(item.get("id", ""), "pending"),
            }
        )
    if table_rows:
        st.markdown("#### Material Table")
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
