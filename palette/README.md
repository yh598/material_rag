# Apparel Material Palette (Backend + UI)

This project provides:
- `FastAPI` backend for retrieval and AI response generation (`/chat`)
- `Streamlit` UI styled as a materials recommendation dashboard
- Synthetic material recommendation cards generated from:
  - `materials.json`
  - `products.json`
  - `product_materials.json`
  - optional prebuilt cache `synthetic_materials_large.json`

## Generate a large synthetic dataset (optional)

Generate 20,000 rows with full filter-combination coverage:

```powershell
python scripts/generate_large_synthetic_dataset.py --size 20000 --min-per-combo 8 --output synthetic_materials_large.json
```

If `synthetic_materials_large.json` exists in repo root, the UI loads it automatically.
The generator expands records to product level and guarantees coverage for every
`division x department x category` combination.

## One-command local startup (Windows PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
```

This command:
- Creates `.venv` if missing
- Installs dependencies
- Stops prior app processes started by this repo
- Starts backend and UI
- Waits for health checks

Open:
- UI: `http://127.0.0.1:8501`
- Backend health: `http://127.0.0.1:8000/health`

Stop both:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop_all.ps1
```

## Share with others (Render deployment)

This repo includes a Render Blueprint file: `render.yaml`.

Steps:
1. Push this repo to GitHub.
2. In Render, click `New +` -> `Blueprint`.
3. Select your repo and deploy.
4. In Render service settings for `palette-backend`, set secret env values:
   - `AI_GATEWAY_BASE_URL`
   - `AI_GATEWAY_API_KEY`
   - optionally `AI_GATEWAY_MODEL`
5. Open the deployed `palette-ui` URL and share it.

Notes:
- UI reads backend URL from Render service discovery (`BACKEND_URL` in `render.yaml`).
- If AI gateway secrets are missing, backend still responds in retrieval-only fallback mode.

## One-command Docker startup

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1 -Docker
```

Stop Docker stack:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop_all.ps1 -Docker
```

## Environment variables

Copy `.env.example` to `.env` and update values as needed.

Required for AI generation:
- `AI_GATEWAY_BASE_URL`
- `AI_GATEWAY_API_KEY`
- `AI_GATEWAY_MODEL`

Used by backend retrieval:
- `PALETTE_CSV_PATH` (defaults to `apparel_material_cluster_sample.csv`)

Optional for UI:
- `BACKEND_URL` (defaults to `http://127.0.0.1:8000`)
- `SYNTHETIC_TARGET_SIZE` (defaults to `12000` when generating in memory)
- `SYNTHETIC_MIN_PER_COMBO` (defaults to `6`; minimum rows per `division x department x category` combination)
- `SYNTHETIC_DATA_PATH` (path to prebuilt synthetic JSON file)

## LLM fallback behavior

If AI gateway settings are missing or unavailable, `/chat` still works in retrieval mode and returns:
- ranked citations from the indexed CSV
- a fallback summary without LLM generation

## Project layout

- `backend/app.py`: API routes and chat orchestration
- `backend/rag/index.py`: TF-IDF retrieval index
- `backend/llm/client.py`: OpenAI-compatible gateway client
- `ui/app.py`: recommendation dashboard + backend integration
- `scripts/generate_large_synthetic_dataset.py`: large synthetic dataset generator
- `scripts/run_all.ps1`: local or Docker startup
- `scripts/stop_all.ps1`: local or Docker shutdown
- `Dockerfile.backend`, `Dockerfile.ui`, `docker-compose.yml`: container deployment
