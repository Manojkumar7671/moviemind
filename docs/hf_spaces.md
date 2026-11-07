# Deploying MovieMind to Hugging Face Spaces

MovieMind is designed to deploy seamlessly to **Hugging Face Spaces** using the **Docker SDK** on standard CPU hardware (free tier). Serving requires zero GPU resources and zero online model training because neural embeddings, retrieval indices, and candidate caches are precomputed and lightweight (< 35 MB total).

---

## Architecture in Production

- **Base Image**: `python:3.11-slim` running as non-root user `appuser` (UID `1000`).
- **Port**: Default port `7860` (standard for Hugging Face Spaces; configurable via `PORT` environment variable).
- **Healthcheck**: Regular polling of `GET /healthz` ensuring server readiness and reporting LLM configuration and catalog status.
- **Resource Footprint**: Starts in under 3 seconds, consumes ~120 MB RAM, and executes retrieval and ranking with < 20 ms latency on CPU.

---

## Deployment Walkthrough

### 1. Create a Space on Hugging Face
1. Navigate to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Give your Space a name (e.g. `moviemind`).
3. Select **Docker** as the Space SDK (Blank template).
4. Select the **Free CPU** tier (2 vCPU, 16 GB RAM).

### 2. Configure Environment Variables and Secrets
Under Space **Settings** -> **Variables and secrets**, add any of the following optional settings:

| Name | Type | Description | Default / Example |
| :--- | :--- | :--- | :--- |
| `LLM_BASE_URL` | Variable | OpenAI-compatible API base URL | `https://api.groq.com/openai/v1` |
| `LLM_API_KEY` | Secret | API key for LLM conversational features | *(Set as Secret; never committed)* |
| `LLM_MODEL` | Variable | Target chat/tool-calling model name | `openai/gpt-oss-20b` |
| `PORT` | Variable | Web server port | `7860` |
| `MIN_VOTES` | Variable | Minimum votes for world catalog builder | `2000` |

> [!NOTE]
> If `LLM_API_KEY` is omitted, MovieMind automatically engages its deterministic data-chip and rule-based fallback router with full functionality. Recommendation generation (`/api/recommend` and `/api/global/recommend`) makes zero LLM calls regardless.

### 3. Deploying Code
Clone the Space repository and push the MovieMind codebase:

```bash
# Add Hugging Face Space as a git remote
git remote add space https://huggingface.co/spaces/<your-username>/moviemind

# Push to the Space main branch
git push space main
```

### 4. World Cinema Catalog on Spaces
Because raw IMDb datasets and derived parquet files are untracked for licensing compliance:
1. If `artifacts/global_catalog.parquet` is not built, the app starts gracefully; Discover tab (MovieLens 1M) operates normally, while `/api/global/*` returns a clean `503 Service Unavailable` with instructions to run `python scripts/build_catalog.py`.
2. To enable World Cinema on the Space, run `python scripts/build_catalog.py` in your Space container startup script or pre-build stage.

---

## Health Check and Verification

Once deployed, verify container health via:
```bash
curl -f https://<your-space>.hf.space/healthz
```
Expected JSON response:
```json
{
  "status": "ok",
  "llm_configured": true,
  "world_catalog_loaded": true
}
```
