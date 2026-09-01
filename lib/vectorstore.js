require('dotenv').config();
const { CloudClient } = require('chromadb');
const { Document, SentenceSplitter } = require('llamaindex');
const { embedDocuments, embedQuery } = require('./gemma');

const COLLECTION_NAME = process.env.CHROMA_COLLECTION || 'ag_news_scitech';

let clientInstance = null;

function getClient() {
  if (!clientInstance) {
    if (!process.env.CHROMA_API_KEY) {
      throw new Error("Missing CHROMA_API_KEY environment variable.");
    }
    clientInstance = new CloudClient({
      apiKey: process.env.CHROMA_API_KEY,
      tenant: process.env.CHROMA_TENANT,
      database: process.env.CHROMA_DATABASE || 'default_database',
    });
  }
  return clientInstance;
}

async function getCollection() {
  const client = getClient();
  return client.getOrCreateCollection({ name: COLLECTION_NAME });
}

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
