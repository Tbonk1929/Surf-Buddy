const fetch = require('node-fetch');
const fs = require('fs');

// 1. Load and chunk your document
const text = fs.readFileSync('mydoc.txt', 'utf-8');
const chunks = text.match(/(.|[\r\n]){1,500}/g); // Split into ~500 char chunks

// 2. Get embeddings for each chunk using Ollama
async function getEmbedding(text) {
  const res = await fetch('http://localhost:11434/api/embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'llama2', prompt: text })
  });
  const data = await res.json();
  return data.embedding;
}

// 3. Build the vector store (in-memory for demo)
async function buildVectorStore(chunks) {
  const vectors = [];
  for (const chunk of chunks) {
    const embedding = await getEmbedding(chunk);
    vectors.push({ chunk, embedding });
  }
  return vectors;
}

// 4. Find the most relevant chunk for a query
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (normA * normB);
}

async function retrieveRelevantChunk(query, vectors) {
  const queryEmbedding = await getEmbedding(query);
  let best = { score: -Infinity, chunk: '' };
  for (const v of vectors) {
    const score = cosineSimilarity(queryEmbedding, v.embedding);
    if (score > best.score) best = { score, chunk: v.chunk };
  }
  return best.chunk;
}

// 5. Ask a question with RAG
async function askRAG(query, vectors) {
  const context = await retrieveRelevantChunk(query, vectors);
  const prompt = `Use the following context to answer the question:\n\n${context}\n\nQuestion: ${query}\nAnswer:`;
  const res = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'llama2', prompt, stream: false })
  });
  const data = await res.json();
  return data.response;
}

// 6. Demo
(async () => {
  console.log('Building vector store...');
  const vectors = await buildVectorStore(chunks);
  const question = 'What is the main topic of the document?';
  const answer = await askRAG(question, vectors);
  console.log('Q:', question);
  console.log('A:', answer);
})();
