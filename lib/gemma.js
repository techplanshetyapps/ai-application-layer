/**
 * lib/gemma.js
 *
 * Thin wrapper around Vercel's `ai` SDK (ai-sdk) so the rest of the app
 * never talks to a specific provider directly.
 *
 * - Generation (the RAG answer, and the retrieval-agent reasoning step)
 *   is done by a Gemma model (default: gemma-3-27b-it), served through
 *   Google AI Studio's Generative Language API — the same HTTPS API
 *   ai-sdk's @ai-sdk/google provider already speaks.
 * - Embeddings for the vector store are produced by Google's
 *   text-embedding-004 model, since Gemma is a generation-only model
 *   and does not expose an embeddings endpoint.
 * - For fully offline demos/testing, set GEMMA_PROVIDER=ollama to route
 *   generation to a local Ollama server instead (embeddings still use
 *   Google, or you can swap in any local embedding model).
 */

require('dotenv').config();
const { google } = require('@ai-sdk/google');
const { generateText, embed, embedMany } = require('ai');

const GEMMA_MODEL = process.env.GEMMA_MODEL || 'gemini-3.6-flash';
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || 'gemini-embedding-001';
const PROVIDER = process.env.GEMMA_PROVIDER || 'google';
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://127.0.0.1:11434';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'gemma3:4b';

/**
 * Generate text with Gemma. Returns { text, provider, model }.
 */
async function gemmaGenerate({ system, prompt, maxTokens = 512 }) {
  if (PROVIDER === 'ollama') {
    const res = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        prompt: system ? `${system}\n\n${prompt}` : prompt,
        stream: false,
      }),
    });
    if (!res.ok) throw new Error(`Ollama error ${res.status}`);
    const data = await res.json();
    return { text: (data.response || '').trim(), provider: 'ollama', model: OLLAMA_MODEL };
  }

  const { text } = await generateText({
    model: google(GEMMA_MODEL),
    system,
    prompt,
    maxOutputTokens: maxTokens,
  });
  return { text, provider: 'google-ai-studio', model: GEMMA_MODEL };
}

/**
 * Embed a single string for querying the vector store.
 */
async function embedQuery(text) {
  const { embedding } = await embed({
    model: google.textEmbeddingModel(EMBEDDING_MODEL),
    value: text,
  });
  return embedding;
}

/**
 * Embed many documents at once for ingestion into Chroma.
 */
async function embedDocuments(texts) {
  const { embeddings } = await embedMany({
    model: google.textEmbeddingModel(EMBEDDING_MODEL),
    values: texts,
  });
  return embeddings;
}

module.exports = { gemmaGenerate, embedQuery, embedDocuments, GEMMA_MODEL, EMBEDDING_MODEL };
