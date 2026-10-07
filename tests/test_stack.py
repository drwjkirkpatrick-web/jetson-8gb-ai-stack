#!/usr/bin/env python3
"""test_stack.py — Full coexistence test suite for the Jetson 8GB AI stack.

Validates: EMB2 retrieval quality, MRL truncation, concurrent Granite generation
+ embedding batches, server survival, and free RAM headroom.

Usage: python3 tests/test_stack.py   (expects EMB2 on :8091, Granite on :8090)
"""
import json, math, threading, time, urllib.request

EMB_PORT, GEN_PORT = 8091, 8090

def embed(texts, port=EMB_PORT):
    payload = json.dumps({"input": texts, "model": "embeddinggemma-2"}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings",
                                data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return [d["embedding"] for d in json.loads(r.read())["data"]]

def chat(msg, max_tokens=300):
    payload = json.dumps({"messages": [{"role": "user", "content": msg}],
                          "model": "granite4-3b", "max_tokens": max_tokens,
                          "temperature": 0.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{GEN_PORT}/v1/chat/completions",
                                 data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d.get("usage", {}).get("completion_tokens", 0), time.time() - t0

def cos(a, b):
    na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0

def truncate_renorm(vec, dim):
    v = vec[:dim]
    n = math.sqrt(sum(x*x for x in v))
    return [x / n for x in v]

DOCS = [
    "title: KV cache math\ntext: KV cache bytes per token equals two times layers times kv heads times head dimension, quantized by bytes per element.",
    "title: Bread baking\ntext: Combine flour water salt and yeast. Knead 10 minutes, proof 2 hours, bake at 220 degrees Celsius.",
    "title: Python sockets\ntext: Socket programming in Python uses the socket module with bind listen accept and recv calls on a server port.",
    "title: Solar charging\ntext: A 100 watt solar panel charges a portable battery which runs an edge computer during daylight hours.",
    "title: Pupillography\ntext: Asthenocoria measurement records pupil diameter changes under near infrared light stimulation with a dark pupil tracker.",
    "title: MTP decoding\ntext: Multi token prediction speculative decoding uses a draft head to predict several future tokens accepted in a single forward pass.",
    "title: iambic pentameter\ntext: Iambic pentameter is a line of verse with ten syllables alternating unstressed and stressed across five metrical feet.",
    "title: Chess openings\ntext: The Sicilian defense begins with the moves e4 c5 giving black an asymmetrical pawn structure and counterattacking potential.",
    "title: HACCP logs\ntext: Hazard analysis critical control point requires temperature logs at receiving storage and cooking stages of food service.",
    "title: Michelin stars\ntext: Michelin awards up to three stars for cuisine quality mastery of technique personality of the chef and consistency.",
]

QUERIES = [
    ("search: how to calculate KV cache memory per token", 0),
    ("search: what ingredients go into bread dough", 1),
    ("search: creating a network listener in python", 2),
    ("search: running a computer off solar power", 3),
    ("search: infrared pupil measurement technique", 4),
    ("search: speculative decoding draft model speedup", 5),
    ("search: ten syllable poetic meter", 6),
    ("search: e4 c5 opening in chess", 7),
]

def main():
    print("=" * 66)
    print("JETSON 8GB AI STACK — FULL COEXISTENCE TEST SUITE")
    print("=" * 66)

    # 1. retrieval
    doc_vecs = embed(DOCS)
    correct = 0
    for q, expected in QUERIES:
        qv = embed([q])[0]
        sims = [cos(qv, dv) for dv in doc_vecs]
        correct += sims.index(max(sims)) == expected
    print(f"[1] Retrieval: {correct}/8")
    assert correct >= 7, "Retrieval below 7/8 — check --pooling mean and input prefixes"

    # 2. MRL
    mrl_ok = 0
    for dim in (256, 128):
        c = 0
        for q, expected in QUERIES:
            qv = truncate_renorm(embed([q])[0], dim)
            sims = [cos(qv, truncate_renorm(dv, dim)) for dv in doc_vecs]
            c += sims.index(max(sims)) == expected
        mrl_ok += c == 8
        print(f"[2] MRL {dim}d: {c}/8")
    print(f"[2] MRL truncation: {mrl_ok}/2 dims lossless")

    # 3. concurrency
    batch = [f"title: doc {i}\ntext: Concurrent retrieval and generation stack validation on Jetson 8GB unified memory. " * 4 for i in range(32)]
    results = {}
    def gen_job():
        results["gen"] = chat("Describe the full architecture of a RAG system where an embedding model retrieves chunks and a Mamba2 hybrid LLM generates grounded answers. " * 4)
    def emb_job():
        times = []
        for _ in range(4):
            t0 = time.time()
            embed(batch)
            times.append(time.time() - t0)
        results["emb"] = times
    tg = threading.Thread(target=gen_job)
    te = threading.Thread(target=emb_job)
    t0 = time.time(); tg.start(); te.start(); tg.join(); te.join()
    wall = time.time() - t0
    tok, gdt = results["gen"]
    emb_str = ", ".join("%.1fs" % t for t in results["emb"])
    print(f"[3] Concurrent: Granite {tok} tok in {gdt:.1f}s = {tok/gdt:.1f} tok/s | "
          f"EMB2 4 batches [{emb_str}] (wall {wall:.1f}s)")
    assert tok > 100, "Granite generated too little under concurrent load"

    # 4. survival + headroom
    ok = True
    for port, name in ((GEN_PORT, "Granite"), (EMB_PORT, "EMB2")):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            print(f"[4] {name} alive: yes")
        except Exception:
            ok = False
            print(f"[4] {name} alive: NO")
    avail = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split("kB")[0]) // 1024
    print(f"[4] Headroom: {avail} MB available")
    assert ok, "A server died during the concurrent test"

    print("=" * 66)
    print(f"RESULT: {'PASS' if correct == 8 and ok else 'CHECK'} — {correct}/8 retrieval, "
          f"{mrl_ok}/2 MRL lossless, both servers alive, {avail} MB free")
    print("=" * 66)

if __name__ == "__main__":
    main()
