# ai-application-layer
# Gemma Sci/Tech RAG Assistant (AG News)

A Gemma-powered retrieval-augmented fact-checking / topic assistant built for
**Hack Fiesta Miami — Build w Gemma (August 2026)**. It ingests only the
**Sci/Tech** slice of the AG News dataset directly from Hugging Face, indexes
it in **ChromaDB** using **LlamaIndex.TS** for chunking, and answers questions
about the articles using a **Gemma** model called through the **Vercel AI SDK
(ai-sdk)**.

## Live demo

- Hosted app: `<add your Vercel URL here>`
- Kaggle Notebook / Writeup: `<add your Kaggle link here>`
- Demo video (≤3 min): `<add your YouTube link here>`

## Architecture

```
Hugging Face (ag_news, Sci/Tech only)
        │  datasets-server REST API
        ▼
  lib/dataset.js  ── fetch + filter label==3
        ▼
  lib/vectorstore.js
        │  LlamaIndex SentenceSplitter → chunks
        │  ai-sdk embed() → text-embedding-004
        ▼
      ChromaDB (vector store)
        ▲
        │  similarity search (top-k)
  lib/vectorstore.js retrieve()
        │
        ▼
  index.js  /api/query
        │  builds RAG prompt with numbered context
        ▼
  lib/gemma.js  ── ai-sdk generateText() → Gemma (gemma-3-27b-it)
        ▼
   Answer + cited sources → public/index.html dashboard
```

## Setup and execution

### 1. Install dependencies
```bash
npm install
```

### 2. Configure environment
```bash
cp .env.example .env
```
Fill in:
- `GOOGLE_GENERATIVE_AI_API_KEY` — free key from [Google AI Studio](https://aistudio.google.com/apikey). Used for both the Gemma generation calls and the `text-embedding-004` embedding calls.
- `CHROMA_URL` — for local dev, run a Chroma server: `pip install chromadb && chroma run --path ./chroma-data`, then leave the default `http://localhost:8000`. For the deployed Vercel app, point this at a hosted Chroma instance (e.g. Chroma Cloud), since Vercel's serverless functions are stateless and cannot host a persistent local Chroma process.

### 3. Run locally
```bash
npm start
# Server on http://localhost:3000
```

### 4. Ingest the dataset
Open the dashboard at `http://localhost:3000` and click **"Ingest Sci/Tech Articles"**,
or call the API directly:
```bash
curl -X POST http://localhost:3000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"limit": 200, "split": "test"}'
```
This pulls rows straight from `https://datasets-server.huggingface.co/rows?dataset=ag_news&config=default&split=test`, keeps only Sci/Tech-labeled rows, chunks them, embeds them, and upserts into ChromaDB.

### 5. Ask a question
```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What recent Sci/Tech stories mention AI chips?"}'
```

### 6. Deploy to Vercel
```bash
npm i -g vercel
vercel
```
Set the same environment variables in the Vercel project dashboard (Settings → Environment Variables). Point `CHROMA_URL`/`CHROMA_API_KEY` at your hosted Chroma instance before deploying.

## Gemma implementation details

- **Model used:** `gemma-3-27b-it` (configurable via `GEMMA_MODEL` env var; any instruction-tuned Gemma model id served by the same API will work, e.g. a smaller `gemma-3-4b-it` for lower latency).
- **How it's accessed:** through Google AI Studio's Generative Language API, called via the `@ai-sdk/google` provider inside the Vercel **ai-sdk** (`generateText`). An optional local fallback (`GEMMA_PROVIDER=ollama`) calls a locally-served Gemma model through Ollama for fully offline demos.
- **Approach:** Retrieval-Augmented Generation (RAG). The user's question is embedded and matched against Chroma-stored AG News Sci/Tech chunks; the top-k chunks are inserted into a numbered context block, and Gemma is instructed to answer **only** from that context and cite snippet numbers — this is the "fact-checking assistant" pattern rather than free-form generation, which reduces hallucination on a small, closed news corpus.
- **Why Gemma is essential:** Gemma's instruction-tuned reasoning is what turns raw retrieved snippets into a grounded, cited natural-language answer — the retrieval/vector layer alone (Chroma + embeddings) can only rank text, it cannot synthesize or explain it. Gemma is also small enough to run locally via Ollama, so the same RAG pipeline can be demoed fully offline if judging Wi-Fi is unreliable.
- **Key technical decisions:**
  - Embeddings are produced by `text-embedding-004`, not Gemma itself, because Gemma is a text-generation model and does not expose an embeddings endpoint — this is a standard RAG pattern (separate embedding model + separate generation model).
  - LlamaIndex's `SentenceSplitter` is used purely for chunking/document representation; ChromaDB is accessed directly through its official client for storage and similarity search, keeping the pipeline easy to reason about for a hackathon judge.
  - The system prompt forces citation-or-refusal behavior ("say so plainly if the context doesn't contain the answer") to keep answers grounded in the Sci/Tech corpus.

## Reused code and external resources

- **Dataset:** [`ag_news` on Hugging Face](https://huggingface.co/datasets/ag_news) (Zhang, Zhao & LeCun), accessed live via the [Hugging Face datasets-server REST API](https://huggingface.co/docs/datasets-server) — no local copy is checked into this repo.
- **Libraries:** [Vercel AI SDK (`ai`, `@ai-sdk/google`)](https://sdk.vercel.ai/), [ChromaDB JS client](https://docs.trychroma.com/), [LlamaIndex.TS](https://ts.llamaindex.ai/), Express, cors, dotenv — all standard open-source packages installed via npm, unmodified.
- **App scaffold pattern:** the Express-as-a-single-`index.js`-module structure and the dark, blurred-card dashboard styling were adapted from an earlier internal hackathon template (drought-monitoring dashboard) built by the same team, restructured here for a RAG/Gemma workflow instead of a Postgres CRUD app.
- All Gemma/AI Studio and embedding calls are original code written for this project; no third-party RAG boilerplate was copied.

## Repository structure

```
.
├── index.js                # Express app: /api/ingest, /api/query, /api/health
├── lib/
│   ├── dataset.js           # Hugging Face AG News Sci/Tech loader
│   ├── vectorstore.js        # LlamaIndex chunking + ChromaDB storage/retrieval
│   └── gemma.js              # ai-sdk Gemma generation + embeddings
├── public/index.html        # Dashboard UI
├── vercel.json               # Vercel deployment routing
├── .env.example
├── KAGGLE_WRITEUP.md
└── DEVPOST.md
```