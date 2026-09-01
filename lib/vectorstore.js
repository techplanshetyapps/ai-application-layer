require('dotenv').config();
const { embedDocuments, embedQuery } = require('./gemma');

const CHROMA_API_URL = process.env.CHROMA_API_URL;
const CHROMA_API_KEY = process.env.CHROMA_API_KEY;
const COLLECTION_NAME = process.env.CHROMA_COLLECTION || 'ag_news_scitech';

let cachedCollectionId = null;

async function getHeaders() {
  const headers = {
    'Content-Type': 'application/json',
  };

  if (CHROMA_API_KEY) {
    headers['Authorization'] = `Bearer ${CHROMA_API_KEY}`;
    headers['x-chroma-token'] = CHROMA_API_KEY;
  }

  return headers;
}

async function getOrCreateCollectionId() {
  if (cachedCollectionId) return cachedCollectionId;

  if (!CHROMA_API_URL) {
    throw new Error("Missing CHROMA_API_URL environment variable.");
  }

  const tenant = process.env.CHROMA_TENANT || 'default_tenant';
  const database = process.env.CHROMA_DATABASE || 'default_database';
  const baseUrl = `${CHROMA_API_URL}/api/v2/tenants/${tenant}/databases/${database}`;

  const listRes = await fetch(`${baseUrl}/collections`, {
    method: 'GET',
    headers: await getHeaders(),
  });

  if (listRes.ok) {
    const collections = await listRes.json();
    const existing = collections.find(c => c.name === COLLECTION_NAME);
    if (existing) {
      cachedCollectionId = existing.id || existing.collection_id;
      return cachedCollectionId;
    }
  }

  const res = await fetch(`${baseUrl}/collections`, {
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

function splitIntoChunks(text, chunkSize = 256, chunkOverlap = 20) {
  if (!text) return [];
  const words = text.split(/\s+/);
  const chunks = [];
  
  let i = 0;
  while (i < words.length) {
    const chunkWords = words.slice(i, i + chunkSize);
    chunks.push(chunkWords.join(' '));
    i += (chunkSize - chunkOverlap);
  }
  return chunks;
}

async function ingestArticles(articles) {
  const collectionId = await getOrCreateCollectionId();
  const tenant = process.env.CHROMA_TENANT || 'default_tenant';
  const database = process.env.CHROMA_DATABASE || 'default_database';
  const baseUrl = `${CHROMA_API_URL}/api/v2/tenants/${tenant}/databases/${database}`;

  const chunks = [];
  for (const article of articles) {
    const textContent = article.text || article.body || '';
    const textChunks = splitIntoChunks(textContent, 256, 20);
    
    textChunks.forEach((chunkText, i) => {
      chunks.push({
        id: `${article.id}-chunk-${i}`,
        text: chunkText,
        metadata: { title: article.title || 'Untitled', label: article.label || 'Sci/Tech', sourceId: article.id },
      });
    });
  }

  const BATCH = 50;
  for (let i = 0; i < chunks.length; i += BATCH) {
    const batch = chunks.slice(i, i + BATCH);
    const embeddings = await embedDocuments(batch.map((c) => c.text));

    const res = await fetch(`${baseUrl}/collections/${collectionId}/upsert`, {
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
  const tenant = process.env.CHROMA_TENANT || 'default_tenant';
  const database = process.env.CHROMA_DATABASE || 'default_database';
  const baseUrl = `${CHROMA_API_URL}/api/v2/tenants/${tenant}/databases/${database}`;
  
  const queryEmbedding = await embedQuery(question);

  const res = await fetch(`${baseUrl}/collections/${collectionId}/query`, {
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
  const tenant = process.env.CHROMA_TENANT || 'default_tenant';
  const database = process.env.CHROMA_DATABASE || 'default_database';
  const baseUrl = `${CHROMA_API_URL}/api/v2/tenants/${tenant}/databases/${database}`;

  const res = await fetch(`${baseUrl}/collections/${collectionId}`, {
    method: 'GET',
    headers: await getHeaders(),
  });

  if (!res.ok) return 0;
  const data = await res.json();
  if (typeof data.count === 'number') return data.count;

  const countRes = await fetch(`${baseUrl}/collections/${collectionId}/count`, {
    method: 'POST',
    headers: await getHeaders(),
    body: JSON.stringify({}),
  });
  if (!countRes.ok) return 0;
  const countData = await countRes.json();
  return typeof countData === 'number' ? countData : (countData.count || 0);
}

module.exports = { ingestArticles, retrieve, collectionCount };
