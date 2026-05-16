# Crawl4AI Adaptive Research Notes

## 2026-05-15

- The adaptive crawler currently uses Crawl4AI's embedding strategy with local
  `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- This avoids the OpenRouter embedding-provider routing issue seen during early
  tests, while still allowing OpenRouter to be used for optional query expansion.
- Potential risk: local MiniLM embeddings may be less semantically accurate than
  stronger hosted embedding models. Relevance quality should be evaluated during
  testing before this crawler is trusted as production research input.
- If relevance is not good enough, revisit hosted embeddings through OpenRouter,
  direct OpenAI embeddings, or a stronger local embedding model.
