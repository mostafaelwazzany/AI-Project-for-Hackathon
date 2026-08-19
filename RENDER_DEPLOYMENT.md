# Render Deployment

## Recommended deployment

Use Render as a Docker Web Service.

## Required environment variable

Add this in Render dashboard:

```text
GROQ_API_KEY=your_groq_key
```

Do not commit `.env`.

## Why Docker?

The app needs both:

- Next.js frontend/API
- Python RAG process with Chroma and sentence-transformers

Docker keeps both runtimes in one deployable service.

## Render settings

If you deploy manually instead of using `render.yaml`:

- Service type: Web Service
- Runtime: Docker
- Plan: Free
- Health check path: `/`

## First boot note

The first request can be slow because Render downloads the `intfloat/multilingual-e5-base` model into the new server cache. After that, the long-running Python bridge keeps the model loaded while the service is awake.

## Important files included in deployment

- `data/vector_store/chroma`
- `data/chunks`
- `data/evaluation`
- `web`
- Python RAG files in the project root

