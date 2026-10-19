# Mini LLM Backend Service

High-performance, production-ready inference API for the Mini LLM models built with **FastAPI**, **Pydantic v2**, and **PyTorch**.

## Architecture & Features

- **Pre-loaded Checkpoints**: Loads both the `mini-llm-25k` and `mini-llm-100k` models at server startup via FastAPI lifespan management. Model switching is instant with zero disk I/O on inference requests.
- **Hardware Acceleration**: Automatic CUDA detection with seamless CPU fallback.
- **Autoregressive Generation**: Custom sampling loop supporting temperature, top-k, top-p, and reproducibility seeds.
- **Strict Validation**: Enforces input prompt constraints (minimum 5 words) and token bounds (up to 5,000 tokens).
- **Fine-Grained Truncation**: Truncates output cleanly at the requested token count or character limit.
- **Environment Driven**: Fully decoupled configuration via `.env` file for painless deployment.

---

## Directory Structure

```
backend/
├── app/
│   ├── config.py              # Pydantic Settings & environment resolver
│   ├── exceptions.py          # Custom domain exceptions & HTTP handlers
│   ├── main.py                # FastAPI factory, CORS, lifespan & router mounting
│   ├── routers/
│   │   ├── generation.py      # POST /api/v1/generate
│   │   ├── health.py          # GET  /api/v1/health, GET /health
│   │   └── models.py          # GET  /api/v1/models, GET /api/v1/models/{id}
│   ├── schemas/
│   │   ├── generation.py      # GenerationRequest & GenerationResponse schemas
│   │   └── models.py          # ModelInfo & ModelListResponse schemas
│   └── services/
│       ├── model_registry.py  # Model loading, metadata catalog & caching
│       └── text_generator.py  # Autoregressive generation & truncation service
├── .env                       # Active environment configuration
├── .env.example               # Reference template for environment variables
└── requirements.txt           # Python dependencies
```

---

## Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` if not already present:
```bash
cp .env.example .env
```

Review and adjust variables:
- `APP_HOST`: Server bind host (default `0.0.0.0`)
- `APP_PORT`: Server bind port (default `8000`)
- `CORS_ALLOWED_ORIGINS`: Comma-separated frontend URLs (default `http://localhost:5173`)
- `DEVICE`: `auto`, `cuda`, or `cpu`

### 3. Run Development Server
From the repository root (`mini-llm/`):
```bash
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Or from the `backend/` directory:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Reference

### 1. Health Check
`GET /api/v1/health`

Response:
```json
{
  "status": "healthy",
  "device": "cuda",
  "cuda_available": true,
  "cuda_device_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
  "loaded_models": ["mini-llm-25k", "mini-llm-100k"]
}
```

### 2. List Models
`GET /api/v1/models`

Response:
```json
{
  "models": [
    {
      "model_id": "mini-llm-25k",
      "display_name": "Mini LLM — 25K Baseline",
      "description": "Trained on a 25k stories subset...",
      "parameter_count": 3258624,
      "context_length": 128,
      "validation_loss": 0.7277,
      "validation_perplexity": 2.0703
    },
    {
      "model_id": "mini-llm-100k",
      "display_name": "Mini LLM — 100K Advanced",
      "description": "Trained on a 100k stories subset...",
      "parameter_count": 3258624,
      "context_length": 128,
      "validation_loss": 0.6862,
      "validation_perplexity": 1.9862
    }
  ]
}
```

### 3. Generate Completion
`POST /api/v1/generate`

Request Body:
```json
{
  "model_id": "mini-llm-100k",
  "prompt": "The little rabbit hopped across the green meadow",
  "max_tokens": 150,
  "temperature": 0.8,
  "top_k": 20,
  "top_p": 0.9,
  "seed": 42
}
```

Response Body:
```json
{
  "model_id": "mini-llm-100k",
  "prompt": "The little rabbit hopped across the green meadow",
  "completion": " and found a bright red flower sitting near the brook...",
  "full_text": "The little rabbit hopped across the green meadow and found a bright red flower sitting near the brook...",
  "prompt_tokens": 48,
  "completion_tokens": 150,
  "total_tokens": 198,
  "character_count": 520,
  "truncated": true,
  "inference_duration_ms": 320.45
}
```
