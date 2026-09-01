/**
 * lib/vectorstore.js
 *
 * - Uses LlamaIndex.TS's SentenceSplitter to chunk each AG News Sci/Tech
 *   article into retrieval-sized nodes (articles are short, so this mostly
 *   guards against occasional long outliers).
 * - Uses ai-sdk (lib/gemma.js) to embed each chunk with Google's
 *   text-embedding-004 model.
 * - Stores/queries vectors in ChromaDB via the official `chromadb` client.
 */

require('dotenv').config();
const { ChromaClient } = require('chromadb');
const { Document, SentenceSplitter } = require('llamaindex');
const { embedDocuments, embedQuery } = require('./gemma');

const CHROMA_URL = process.env.CHROMA_URL || 'http://localhost:8000';
const CHROMA_API_KEY = process.env.CHROMA_API_KEY || undefined;
const COLLECTION_NAME = process.env.CHROMA_COLLECTION || 'ag_news_scitech';

const client = new ChromaClient({
  path: CHROMA_URL,
  ...(CHROMA_API_KEY ? { auth: { provider: 'token', credentials: CHROMA_API_KEY } } : {}),
});

async function getCollection() {
  return client.getOrCreateCollection({ name: COLLECTION_NAME });
}

/**
 * Split each article into chunks using LlamaIndex's SentenceSplitter,
 * embed the chunks, and upsert them into the Chroma collection.
 */
async function ingestArticles(articles) {
  const splitter = new SentenceSplitter({ chunkSize: 256, chunkOverlap: 20 });

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

  const collection = await getCollection();
  const BATCH = 50;
  for (let i = 0; i < chunks.length; i += BATCH) {
    const batch = chunks.slice(i, i + BATCH);
    const embeddings = await embedDocuments(batch.map((c) => c.text));
    await collection.upsert({
      ids: batch.map((c) => c.id),
      embeddings,
      documents: batch.map((c) => c.text),
      metadatas: batch.map((c) => c.metadata),
    });
  }

  return { articlesIngested: articles.length, chunksIngested: chunks.length };
}

/**
 * Embed the user's question and retrieve the top-k most similar chunks.
 */
async function retrieve(question, topK = 5) {
  const collection = await getCollection();
  const queryEmbedding = await embedQuery(question);
  const results = await collection.query({
    queryEmbeddings: [queryEmbedding],
    nResults: topK,
  });

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
  const collection = await getCollection();
  return collection.count();
}

module.exports = { ingestArticles, retrieve, collectionCount };
