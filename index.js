require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Safe loader to prevent top-level module crashes from killing the container silently
function loadModules() {
  const dataset = require('./lib/dataset');
  const vectorstore = require('./lib/vectorstore');
  const gemma = require('./lib/gemma');
  return { dataset, vectorstore, gemma };
}

// Health check endpoint
app.get('/api/health', async (req, res) => {
  try {
    const { vectorstore, gemma } = loadModules();
    const count = await vectorstore.collectionCount();
    res.json({ ok: true, gemmaModel: gemma.GEMMA_MODEL, vectorCount: count });
  } catch (err) {
    res.status(500).json({ 
      ok: false, 
      error: err.message, 
      stack: err.stack,
      envCheck: {
        hasChromaKey: !!process.env.CHROMA_API_KEY,
        hasGoogleKey: !!process.env.GOOGLE_GENERATIVE_AI_API_KEY
      }
    });
  }
});

// Ingest endpoint
app.post('/api/ingest', async (req, res) => {
  try {
    const { dataset, vectorstore } = loadModules();
    const limit = Math.min(parseInt(req.body?.limit, 10) || 10, 2000);
    const split = req.body?.split === 'train' ? 'train' : 'test';

    const articles = await dataset.loadSciTechArticles(limit, split);
    const result = await vectorstore.ingestArticles(articles);

    res.json({
      message: `Ingested ${result.articlesIngested} Sci/Tech articles (${result.chunksIngested} chunks) from AG News [${split}] into Chroma.`,
      ...result,
      sample: articles.slice(0, 3).map((a) => a.title),
    });
  } catch (err) {
    console.error('Ingest error:', err.message);
    res.status(500).json({ error: err.message, stack: err.stack });
  }
});

// Query endpoint
app.post('/api/query', async (req, res) => {
  const { question, topK } = req.body || {};
  if (!question || typeof question !== 'string') {
    return res.status(400).json({ error: 'question (string) is required.' });
  }

  try {
    const { vectorstore, gemma } = loadModules();
    const chunks = await vectorstore.retrieve(question, topK || 5);

    const context = chunks
      .map((c, i) => `[${i + 1}] (${c.title || 'untitled'}) ${c.text}`)
      .join('\n\n');

    const system =
      'You are a fact-checking and topic-routing assistant for Sci/Tech news. ' +
      'Answer ONLY using the numbered context snippets provided. ' +
      'Cite snippet numbers like [1], [2] that support each claim. ' +
      'If the context does not contain the answer, say so plainly instead of guessing.';

    const prompt = `Context snippets from AG News (Sci/Tech):\n\n${context}\n\nQuestion: ${question}\n\nAnswer:`;

    const { text, provider, model } = await gemma.gemmaGenerate({ system, prompt, maxTokens: 400 });

    res.json({
      answer: text,
      provider,
      model,
      sources: chunks.map((c) => ({ title: c.title, sourceId: c.sourceId, distance: c.distance })),
    });
  } catch (err) {
    console.error('Query error:', err.message);
    res.status(500).json({ error: err.message, stack: err.stack });
  }
});

const PORT = process.env.PORT || 3000;
if (require.main === module) {
  app.listen(PORT, () => console.log(`Gemma AG-News RAG app listening on :${PORT}`));
}

module.exports = app;
