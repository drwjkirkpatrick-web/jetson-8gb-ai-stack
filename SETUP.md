# SETUP.md — Full installation guide

## Prerequisites

- Jetson Orin Nano 8GB (or any ARM64 device with 8GB unified memory)
- Ubuntu with sudo access
- ~10GB free disk for models + build
- Hermes Agent installed (`~/.hermes/`)

## 1. Stop the GUI

The desktop consumes ~1.5GB. With it running, the full stack does not fit.

```bash
sudo systemctl stop gdm.service
```

To make it survive reboots (optional, headless Jetson):
```bash
sudo systemctl disable gdm.service
```

## 2. Build llama.cpp (with CUDA)

EmbeddingGemma 2 requires llama.cpp build ≥ 1098 (commit 448147d4, 2026-10-07) for the
`gemma-embedding2` architecture. Granite 4.0 hybrid-Mamba support ships in the same build.

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=870 \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 6
```

- `CMAKE_CUDA_ARCHITECTURES=870` is the Orin generation. Use 720/730/735 for older Jetsons.
- Build takes ~20 minutes on-device.
- Verify: `./build/bin/llama-server --version` should print `build 1098` or newer.

## 3. Download the models

```bash
# Granite 4.0-3B — 2.0GB (generator LLM)
mkdir -p ~/models/new-zoo
# Option A (verified copy from Ollama, if you have it):
cp /usr/share/ollama/.ollama/models/blobs/sha256-* ~/models/new-zoo/granite4-3b.gguf
# Option B (download):
# see https://huggingface.co/models?search=granite-4.0-3b GGUF — Q4_K_M recommended

# EmbeddingGemma 2 — 168MB (retrieval embeddings)
mkdir -p ~/models/embeddinggemma-2
wget -O ~/models/embeddinggemma-2/embeddinggemma-2-UD-Q4_K_XL.gguf \
  "https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/embeddinggemma-2-UD-Q4_K_XL.gguf"
```

## 4. Launch the servers

**Critical: use the repo's launch scripts — they include `setsid`.**
A server launched from an agent/terminal background call gets placed in a memory cgroup
capped at ~4.7GB; Granite needs ~4.0GB of that for weights+KV and gets OOM-killed during load.
`setsid` detaches the process into the gateway cgroup, which has no such cap.

```bash
cd ~/projects/jetson-8gb-ai-stack
bash scripts/launch_emb2.sh 8091      # EmbeddingGemma 2 (retrieval)
bash scripts/launch_granite.sh 8090  # Granite 4.0-3B (generator)
```

Inside a Hermes `execute_code` call, equivalent:

```python
import subprocess
cmd = "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_NO_PINNED=1 "
cmd += "~/llama.cpp/build/bin/llama-server -m ~/models/new-zoo/granite4-3b.gguf "
cmd += "--alias granite4-3b --host 127.0.0.1 --port 8090 "
cmd += "-ngl 99 -c 32768 -ctk q8_0 -ctv q8_0 -b 512 -ub 512 -fa on --jinja "
cmd += "-np 1 -t 6 --temp 0.0 --top-k 1 --repeat-penalty 1.0 --fit off"
p = subprocess.Popen(["setsid", "bash", "-c", cmd],
                     stdout=open("/tmp/granite.log","w"), stderr=open("/tmp/granite.log","w"))
```

## 5. Link Hermes to Granite

```bash
hermes config set model.default granite4-3b
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8090/v1
hermes config set model.api_key none
hermes config set model.context_length 32768
```

Restart Hermes after config changes. Verify with `hermes chat`.

## 6. Add Serena MCP to Hermes

Serena runs as an MCP server configured in Hermes `config.yaml`:

```yaml
mcp:
  servers:
    serena:
      command: uvx
      args: ["--from", "git+https://github.com/oraios/serena.git", "serena-agent"]
```

Then in a Hermes session, Serena tools appear as `mcp__serena__*`. Activating a Python project
loads pyright (~144MB total); a bash-only project uses ~55MB.

Install uv first if needed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## 7. RAG index and query (usage pattern)

```python
# Index documents (one-time per corpus)
docs = ["title: recipe\ntext: Combine flour, water, salt, yeast...", ...]
vecs = embed_batch(docs, port=8091)   # POST /v1/embeddings {"input": [...]}
save_to_disk(vecs)                     # 768-dim f32, or truncate to 256d MRL

# Query time
qv = embed_batch(["search: what ingredients do I need?"], port=8091)
top = cosine_topk(qv, index, k=4)
answer = chat(f"Context:\n{top}\n\nQuestion: what ingredients do I need?")
```

**Formatting rules (critical for quality):**
- Documents: `title: <name>\ntext: <content>`
- Queries: `search: <question>`
- Pooling: mean (already in launch script)
- L2-normalize before cosine (server does this; re-normalize after MRL truncation)

## 8. Verify the whole stack

```bash
bash scripts/stack_health.sh
python3 tests/test_stack.py
```

`test_stack.py` runs the exact verification used to validate this setup:
- EMB2 retrieval 8/8 (10-doc corpus, 8 queries)
- MRL truncation 8/8 at 256d
- Concurrent: Granite 500-tok generation while 4 embedding batches run
- Both servers alive at the end + free RAM report

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Server dies during model load | launched inside hermes-worker cgroup (~4.7GB cap) | relaunch with `setsid` (use repo scripts) |
| 96MB free, everything OOMs | f16 KV at 32K | use q8_0 KV (`-ctk q8_0 -ctv q8_0`) |
| "unknown architecture: gemma-embedding2" | llama.cpp too old | `git pull` + rebuild, need build ≥ 1098 |
| Embeddings all ~0.85 cross-similarity | wrong pooling | add `--pooling mean` |
| Docs > 512 tok rejected | batch width too small | raise `-b`/`-ub` (RAM cost), or chunk docs |
| Retrieval ranking wrong results | missing input prefixes | use `title:...\ntext:...` and `search:` |
| Gateway sluggish when both models run | normal contention | expect ~20% gen speed penalty |
| `ngram`/MTP question: N/A | EmbeddingGemma has no MTP tensors; MTP is generation-only | use Granite MTP when released |
| OOM after long uptime | gateway memory grows | restart gateway, keep an eye on `free -h` |

## Why these settings (the physics)

- **Unified memory** (`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`): without it CUDA cannot map system
  RAM and models >2GB fail to load on Jetson.
- **NO_PINNED** (`GGML_CUDA_NO_PINNED=1`): saves ~500MB of CUDA pinned staging memory.
- **ngl 99**: full GPU offload is ~3x faster than partial on modern llama.cpp builds.
- **q8_0 KV**: 32K tokens × KV cost; f16 would need double the bytes and leaves no headroom.
- **--fit off**: prevents llama.cpp auto-shrinking context to "fit" device memory, which
  would silently reduce your 32K window.
