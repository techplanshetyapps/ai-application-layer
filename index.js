require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Global catch to output any startup or runtime errors directly to the browser
app.get('/', async (req, res) => {
  try {
    // Test loading modules dynamically to isolate the crash
    const { loadSciTechArticles } = require('./lib/dataset');
    const { ingestArticles, retrieve, collectionCount } = require('./lib/vectorstore');
    const { gemmaGenerate, GEMMA_MODEL } = require('./lib/gemma');

    res.json({ 
      status: "Modules loaded successfully", 
      model: GEMMA_MODEL,
      envCheck: {
        hasChromaKey: !!process.env.CHROMA_API_KEY,
        hasGoogleKey: !!process.env.GOOGLE_GENERATIVE_AI_API_KEY
      }
    });
  } catch (err) {
    res.status(500).json({
      error: "FATAL BOOT CRASH",
      message: err.message,
      stack: err.stack
    });
  }
});

app.get('/api/health', async (req, res) => {
  try {
    const { collectionCount } = require('./lib/vectorstore');
    const { GEMMA_MODEL } = require('./lib/gemma');
    const count = await collectionCount();
    res.json({ ok: true, gemmaModel: GEMMA_MODEL, vectorCount: count });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message, stack: err.stack });
  }
});

const PORT = process.env.PORT || 3000;
if (require.main === module) {
  app.listen(PORT, () => console.log(`Listening on :${PORT}`));
}

module.exports = app;
