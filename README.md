# hm-rag-rust

Rust port of [`qdrant-boosted-search`](https://github.com/RGGH/qdrant-boosted-search)'s
retrieval pipeline, using [Rig](https://rig.rs) to add an LLM answer on top
of the results.

## What this does

1. Embeds your query locally with **BGE-small-en-v1.5** (384-dim), via
   `fastembed` — the Rust equivalent of the Python project's
   `sentence-transformers` call in `src/embeddings.py`.
2. Runs the same **popularity/recency boosted formula query** against your
   existing `hm_products` Qdrant collection, ported line-for-line from
   `src/boosted_search.py`'s `FormulaQuery`/`SumExpression`/`MultExpression`.
3. Hands the retrieved products to a local LLM (via Rig + Ollama) as
   context and asks it your question — a minimal RAG loop, structured like
   Rig's own
   [`rag_ollama` example](https://github.com/0xPlaygrounds/rig/blob/main/examples/rag_ollama/src/main.rs).

## Project layout

```
src/
  embedder.rs      -> local BGE-small-en-v1.5 embeddings   (embeddings.py)
  qdrant_store.rs  -> Qdrant client + boosted search        (qdrant.py, boosted_search.py)
  rag.rs           -> build context + ask the LLM (Rig)     (new)
  main.rs          -> CLI wiring it all together            (boosted_search.py's __main__)
```

## Important: gRPC vs REST port

The Python project connects to Qdrant's **REST** port:
`QdrantClient("http://localhost:6333")`.

The Rust `qdrant-client` crate talks **gRPC** instead, so point it at
**6334**:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

(if your existing container only publishes 6333, add `-p 6334:6334` or
recreate it — gRPC is enabled by default in the qdrant/qdrant image).

## Setup

1. Have your existing `hm_products` Qdrant collection populated (same one
   the Python project uses — nothing to re-ingest).
2. Install [Ollama](https://ollama.com) and pull a model:
   ```bash
   ollama pull llama3.2
   ```
   (You can swap Ollama for OpenAI/Anthropic/etc. — Rig supports both;
   see "Using a different LLM" below.)
3. Build & run:
   ```bash
   cargo run -- "summer dress for a wedding"
   ```

   Useful flags:
   ```bash
   cargo run -- "summer dress for a wedding" \
     --popularity-weight 0.10 \
     --recency-weight 0.05 \
     --limit 5 \
     --model llama3.2 \
     --qdrant-url http://localhost:6334

   # Just the boosted search results, no LLM call:
   cargo run -- "summer dress for a wedding" --no-llm
   ```

## Payload field names

`Product::from(&ScoredPoint)` in `qdrant_store.rs` reads the same payload
keys `app.py` does: `prod_name`/`name`, `product_type`, `colour`,
`detail_desc`, `popularity`, `recency`. If your collection uses slightly
different key names, adjust the `payload_str`/`payload_f64` lookups there.

## Using a different LLM

`rag.rs` uses `rig::providers::ollama` to match Rig's official RAG example
and keep everything local/free. To use OpenAI, Anthropic, etc. instead,
swap the client construction, e.g.:

```rust
use rig::providers::openai;
let client = openai::Client::from_env(); // needs OPENAI_API_KEY
let agent = client.agent("gpt-4o").preamble(&preamble).build();
```
