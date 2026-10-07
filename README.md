# Jetson 8GB AI Stack — Granite 4.0-3B + EmbeddingGemma 2 + Serena + Hermes

**The complete coexistence setup: RAG-enabled local AI, code intelligence, and an agent gateway — all on 8GB of unified memory.**

Verified live on Jetson Orin Nano 8GB (llama.cpp build 1098, commit 448147d4, 2026-10-07):
all four services running simultaneously with **1.0–1.1GB of RAM headroom**, full RAG pipeline at 4.4s per
32-chunk batch, generator at 15–16 tok/s under concurrent load.

| Component | Role | RAM (measured) |
|---|---|---|
| Granite 4.0-3B (32K ctx, q8_0 KV) | Generator LLM — chat, RAG answers, function calling (BFCL 96.2%) | 4,027 MB |
| EmbeddingGemma 2 (UD-Q4_K_XL) | Embedding model — retrieval index, semantic search | 518 MB |
| Serena MCP + pyright LSP | Code intelligence — symbol search, refactors, diagnostics | 144 MB |
| Hermes agent gateway | Agent runtime — tools, memory, Telegram, chat | ~400–600 MB |
| System (Ubuntu, GDM off) | | ~250 MB |
| **Total** | | **~5.4–5.6 GB used, ~1.0–1.1 GB free** |

## The three rules that make it work

1. **GDM stays off.** `sudo systemctl stop gdm.service` frees ~1.5GB. With the GUI on, this stack OOMs.
2. **Launch every llama-server with `setsid`** (via `execute_code`, never `terminal(background=true)`).
   The hermes-worker cgroup caps a single process at ~4.7GB; Granite needs ~4.0GB just for weights+KV.
   setsid launches land in the gateway cgroup instead and survive.
3. **q8_0 KV cache, not f16.** At 32K context the f16 KV cache leaves 96MB of headroom (one allocation
   from OOM); q8_0 leaves 1.38GB. Same speed, near-lossless quality.

## Quick start

```bash
# 0. Stop the GUI (free ~1.5GB)
sudo systemctl stop gdm.service

# 1. Start EmbeddingGemma 2 (retrieval) on :8091
bash scripts/launch_emb2.sh 8091

# 2. Start Granite 4.0-3B (generator) on :8090
bash scripts/launch_granite.sh 8090

# 3. Link Hermes to Granite
hermes config set model.default granite4-3b
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8090/v1
hermes config set model.api_key none
hermes config set model.context_length 32768

# 4. Verify
curl -s http://127.0.0.1:8090/health    # generator
curl -s http://127.0.0.1:8091/health    # retrieval
```

Serena MCP is configured in Hermes `config.yaml` (native MCP client) — it adds ~144MB with pyright active.
See `SETUP.md` for full instructions including llama.cpp compile flags and model downloads.

## RAG query pattern

```python
# 1. Embed a question with EmbeddingGemma 2 (:8091)
question_vec = embed("search: what is the KV cache formula?")

# 2. Match against your indexed chunks (cosine similarity)
top_chunks = search(index, question_vec, k=4)

# 3. Feed the retrieved chunks to Granite (:8090) as grounding context
answer = chat(f"Answer using this context: {top_chunks}\nQuestion: ...")
```

- Documents: `title: <name>\ntext: <content>` | Queries: `search: <question>`
- **Mean pooling** (`--pooling mean`) is mandatory — last pooling breaks discrimination (1/3 → 3/3 in testing)
- MRL truncation: cut the 768-dim vector to 256d + renormalize for 3x storage cut, still 8/8 retrieval
- Embedding batches: 32 docs ≈ 4.4s; keep batch width (-b 512) matching your typical chunk size

## Performance envelope (measured 2026-10-07)

| Metric | Granite alone | Full stack concurrent | Penalty |
|---|---|---|---|
| Generation speed | 19.2 tok/s | 15.0–16.2 tok/s | ~20% under load |
| EMB2 batch (32 docs) | 4.4s | 4.4s | none |
| Headroom | — | 1,049 MB free | — |

## Repository layout

```
jetson-8gb-ai-stack/
├── README.md               # this file
├── SETUP.md                # full step-by-step setup (compile, download, config, verify)
├── scripts/
│   ├── launch_emb2.sh      # EmbeddingGemma 2 server (setsid, mean pooling, b=512)
│   ├── launch_granite.sh   # Granite 4.0-3B server (setsid, 32K q8_0 KV)
│   └── stack_health.sh     # one-shot health check for all services
└── tests/
    └── test_stack.py       # the full coexistence test suite (retrieval + concurrent gen)
```

## Model files

| File | Size | Source |
|---|---|---|
| `~/models/new-zoo/granite4-3b.gguf` | 2.0GB | Ollama blob copy (sha256-verified) |
| `~/models/embeddinggemma-2/embeddinggemma-2-UD-Q4_K_XL.gguf` | 168MB | unsloth/embeddinggemma-2-GGUF |

Granite 4.0-3B is Apache-2.0 (IBM). EmbeddingGemma 2 is Gemma Terms of Use (Google).

## Verified-by dates

All numbers measured 2026-10-07 on Jetson Orin Nano 8GB, llama.cpp build 1098 (commit 448147d4).
Re-verify with `python3 tests/test_stack.py` after any llama.cpp update.
