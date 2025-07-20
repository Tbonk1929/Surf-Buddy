const express = require('express');
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));
const cors = require('cors');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 4000;

// Replace with your actual Stormglass API key
const STORMGLASS_API_KEY = 'e0d9d084-5431-11f0-89b2-0242ac130006-e0d9d0de-5431-11f0-89b2-0242ac130006';

app.use(cors());
app.use(express.json());

// --- RAG setup ---
const docText = fs.existsSync('mydoc.txt') ? fs.readFileSync('mydoc.txt', 'utf-8') : '';
const docChunks = docText ? docText.match(/(.|[\r\n]){1,500}/g) : [];
let docVectors = null;

async function getEmbedding(text) {
  const res = await fetch('http://localhost:11434/api/embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'llama2', prompt: text })
  });
  const data = await res.json();
  console.log('Embedding response:', data);
  if (!data.embedding) {
    console.error('No embedding returned:', data);
    return null;
  }
  return data.embedding;
}

function cosineSimilarity(a, b) {
  if (!a || !b) return -Infinity;
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (normA * normB);
}

async function buildVectorStore() {
  if (!docChunks.length) return [];
  const vectors = [];
  for (const chunk of docChunks) {
    const embedding = await getEmbedding(chunk);
    vectors.push({ chunk, embedding });
  }
  return vectors;
}

async function retrieveRelevantChunk(query, vectors) {
  const queryEmbedding = await getEmbedding(query);
  let best = { score: -Infinity, chunk: '' };
  for (const v of vectors) {
    const score = cosineSimilarity(queryEmbedding, v.embedding);
    if (score > best.score) best = { score, chunk: v.chunk };
  }
  return { chunk: best.chunk, score: best.score };
}

// Build vector store at startup
(async () => {
  if (docChunks.length) {
    console.log('Building RAG vector store for mydoc.txt...');
    docVectors = await buildVectorStore();
    console.log('RAG vector store ready.');
  }
})();

app.post('/rag', async (req, res) => {
  const { question, fallbackPrompt } = req.body;
  if (!question) return res.status(400).json({ error: 'Missing question' });
  try {
    let prompt = fallbackPrompt;
    let usedRag = false;
    if (docVectors && docVectors.length) {
      const { chunk, score } = await retrieveRelevantChunk(question, docVectors);
      // If similarity is above a threshold, use RAG
      if (score > 0.7) {
        prompt = `Use the following context to answer the question.\n\n${chunk}\n\nQuestion: ${question}\nAnswer:`;
        usedRag = true;
      }
    }
    const ollamaRes = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'mistral', prompt, stream: false })
    });
    const data = await ollamaRes.json();
    console.log('Ollama LLM response:', data);
    res.json({ response: data.response, usedRag });
  } catch (err) {
    console.error('RAG endpoint error:', err);
    res.status(500).json({ error: 'Failed to generate response' });
  }
});

app.get('/api/stormglass', async (req, res) => {
  const { url } = req.query;
  if (!url) return res.status(400).json({ error: 'Missing url parameter' });

  try {
    const sgRes = await fetch(url, {
      headers: { 'Authorization': STORMGLASS_API_KEY }
    });
    const data = await sgRes.json();
    res.json(data);
  } catch (err) {
    console.error('Stormglass proxy error:', err);
    res.status(500).json({ error: 'Failed to fetch from Stormglass' });
  }
});

app.listen(PORT, () => {
  console.log(`Backend proxy running on http://localhost:${PORT}`);
}); 