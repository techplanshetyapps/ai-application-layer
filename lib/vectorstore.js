/**
 * lib/vectorstore.js
 * 
 * Fully compatible CommonJS vector store wrapper using native fetch for Chroma Cloud 
 * and dynamic imports for ESM-only dependencies like LlamaIndex, preventing 
 * ERR_REQUIRE_ESM and native binary crashes on Vercel.
 */

require('dotenv').config();
const { embedDocuments, embedQuery } = require('./gemma');

const CHROMA_API_URL = process.env.CHROMA_API_URL;
const CHROMA_API_KEY = process.env.CHROMA_API_KEY;
const COLLECTION_NAME = process.env.CHROMA_COLLECTION || 'ag_news_scitech';

let cachedCollectionId = null;

async function getHeaders() {
  return {
    'Content-Type': 'application/json',
    ...(CHROMA_API_KEY ? { 'Authorization': `Bearer ${CHROMA_API_KEY}` } : {}),
  };
}

async function getOrCreateCollectionId() {
  if (cachedCollectionId) return cachedCollectionId;

  if (!CHROMA_API_URL) {
    throw new Error("Missing CHROMA_API_URL environment variable.");
  }

  const res = await fetch(`${CHROMA_API_URL}/api/v1/collections`, {
    method: 'POST',
    headers: await getHeaders(),
    body: JSON.stringify({ name: COLLECTION_NAME, get_or_create: true }),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Chroma API error (${res.status}): ${errText}`);
  }

  const data = await res.json();
  cachedCollectionId = data.id || data.collection_id;
  return cachedCollectionId;
}

async function ingestArticles(articles) {
  // Dynamically import LlamaIndex to bypass ESM require restrictions in CommonJS
  const { Document, SentenceSplitter } = await import('llamaindex');
  
  const splitter = new SentenceSplitter({ chunkSize: 256, chunkOverlap: 20 });
  const collectionId = await getOrCreateCollectionId();

  const chunks = [];
  for (const article of articles) {
    const doc = new Document({ text: article.text, id_: article.id });
    const nodes = splitter.getNodesFromDocuments([doc]);
    nodes.forEach((node, i) => {
      chunks.push({
        id: `${article.id}-chunk-${i}`,
        text: node.getContent ? node.getContent() : node.text,
        metadata: { title: article.title, label: article.label, sourceId: article.id },
      });
    });
  }

  const BATCH = 50;
  for (let i = 0; i < chunks.length; i += BATCH) {
    const batch = chunks.slice(i, i + BATCH);
    const embeddings = await embedDocuments(batch.map((c) => c.text));

    const res = await fetch(`${CHROMA_API_URL}/api/v1/collections/${collectionId}/upsert`, {
      method: 'POST',
      headers: await getHeaders(),
      body: JSON.stringify({
        ids: batch.map((c) => c.id),
        embeddings: embeddings,
        documents: batch.map((c) => c.text),
        metadatas: batch.map((c) => c.metadata),
      }),
    });

    if (!res.ok) {
      throw new Error(`Chroma upsert failed: ${await res.text()}`);
    }
  }

  return { articlesIngested: articles.length, chunksIngested: chunks.length };
}

async function retrieve(question, topK = 5) {
  const collectionId = await getOrCreateCollectionId();
  const queryEmbedding = await embedQuery(question);

  const res = await fetch(`${CHROMA_API_URL}/api/v1/collections/${collectionId}/query`, {
    method: 'POST',
    headers: await getHeaders(),
    body: JSON.stringify({
      query_embeddings: [queryEmbedding],
      n_results: topK,
      include: ['documents', 'metadatas', 'distances'],
    }),
  });

  if (!res.ok) {
    throw new Error(`Chroma query failed: ${await res.text()}`);
  }

  const results = await res.json();
  const docs = results.documents?.[0] || [];
  const metas = results.metadatas?.[0] || [];
  const dists = results.distances?.[0] || [];

  return docs.map((text, i) => ({
    text,
    title: metas[i]?.title,
    sourceId: metas[i]?.sourceId,
    distance: dists[i],
  }));
}

async function collectionCount() {
  const collectionId = await getOrCreateCollectionId();
  const res = await fetch(`${CHROMA_API_URL}/api/v1/collections/${collectionId}/count`, {
    method: 'GET',
    headers: await getHeaders(),
  });

  if (!res.ok) return 0;
  const count = await res.json();
  return typeof count === 'number' ? count : (count.count || 0);
}

module.exports = { ingestArticles, retrieve, collectionCount };
