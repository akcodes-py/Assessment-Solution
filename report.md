# Task 3 Report: Local LLM Inference & Optimisation

> **TL;DR:** Switching **Qwen2.5-1.5B** from 8-bit (`q8_0`) to 4-bit (`q4_K_M`) on a Tesla T4 GPU makes generation **31% faster** and cuts VRAM by **35%** with only a minor quality drop (**6/10 vs. 7/10**). For serving 20 concurrent users on one 24 GB GPU, **4-bit quantisation is required**—`Qwen2.5-7B q4_K_M` offers the best balance of quality and capacity.

---

## Key Decisions Taken & Why

| Decision | Why We Chose It |
| :--- | :--- |
| **Split server into 2 phases** (`1 slot` then `8 slots`) | Ollama pre-allocates KV cache per slot. Phase 1 (`parallel=1`) gives clean single-user VRAM; Phase 2 (`parallel=8`) tests real concurrency. |
| **Use `q4_K_M` over `q8_0` for deployment** | **+31.3% faster** (131.4 vs. 100.1 tok/s) and **-630 MiB VRAM** with nearly identical practical quality. |
| **Recommend `Qwen2.5-7B q4_K_M` for 20 users (24 GB GPU)** | Uses only **8.5 GB** of 21.6 GB usable VRAM (~43 tok/s per user, scales up to 131 users). Unquantised `fp16` 8B (22.9 GB) does not fit. |
| **Flag `num_predict=200` truncation in quality checks** | Code/SQL answers (Q02, Q09) were cut off mid-output, causing syntax failures even when the model's logic was mostly right. |

---

## 1. Setup
- **Hardware:** NVIDIA Tesla T4 GPU (320 GB/s peak memory bandwidth), pinned to GPU 0
- **Runtime:** Ollama `0.40.17` via HTTP API (`num_ctx=2048` for main speed/quality runs)
- **Models Compared:** `qwen2.5:1.5b-instruct-q8_0` (8-bit) vs. `qwen2.5:1.5b-instruct-q4_K_M` (4-bit)

---

## 2. Speed vs. VRAM (`q8_0` vs. `q4_K_M`)

| Metric | `q8_0` (8-bit) | `q4_K_M` (4-bit) | Impact |
| :--- | :---: | :---: | :--- |
| **Generation Speed** | 100.12 tok/s | **131.42 tok/s** | **+31.3% faster** |
| **Prompt Processing** | 2,757 tok/s | **2,966 tok/s** | +7.6% faster |
| **VRAM Used (`ctx=2048`)** | 1,815 MiB | **1,185 MiB** | **-34.7% (-630 MiB)** |
| **Model File Size** | 1,570 MiB | **940 MiB** | **-40% smaller** |
| **Implied Bandwidth** | 165 GB/s | 130 GB/s | Memory-bandwidth bound |

**What this means:** Generating each token requires reading all model weights from GPU memory. Because both models use a large share of the T4's 320 GB/s bandwidth, decoding is **memory-bound**—fewer bytes per weight directly translates to faster generation.

---

## 3. Context Window & KV Cache Growth
Qwen2.5-1.5B uses Grouped-Query Attention (`28 layers`, `2 KV heads`, `12 heads`), requiring **28.0 KiB/token** in fp16 KV cache.

| Context (`num_ctx`) | `q4_K_M` VRAM | Measured Growth | Predicted KV Growth |
| :---: | :---: | :---: | :---: |
| 512 | 1,141 MiB | — | — |
| 2,048 | 1,185 MiB | +44 MiB | +42 MiB |
| 8,192 | 1,431 MiB | +290 MiB | +210 MiB |
| 32,768 | 2,151 MiB | +1,010 MiB | +882 MiB |

**What this means:** VRAM scales linearly with context length, staying slightly above the KV-only formula due to extra attention/compute buffers.

---

## 4. Concurrency Scaling (`q4_K_M`, 8 Parallel Slots)

| Concurrent Users | Total Throughput | Speed per User | Wall Time |
| :---: | :---: | :---: | :---: |
| 1 | 118.6 tok/s | 122.6 tok/s | 0.76 s |
| 2 | 210.3 tok/s | 108.4 tok/s | 0.69 s |
| 4 | 281.4 tok/s | 74.1 tok/s | 1.01 s |
| 8 | **357.8 tok/s** | 50.4 tok/s | 1.49 s |

**What this means:** Batching 8 users **tripled total throughput** (118.6 → 357.8 tok/s) because weights are read once per batched step, while keeping per-user speed comfortable at **50.4 tok/s**.

---

## 5. Quality Check (`q8_0` vs. `q4_K_M`)
- **Automated Scores:** `q8_0` scored **7/10** vs. `q4_K_M` at **6/10** (0/10 identical outputs at temperature 0).
- **Where `q8_0` won:** Better strict formatting and code details—extracted `"6"` cleanly in JSON (Q03) instead of `"6 years"`, and remembered `.lower()` for case-insensitive palindromes (Q02).
- **Where `q4_K_M` won:** Followed the exact 3-sentence limit in Q01 (`q8_0` wrote 2 sentences).
- **Important Caveat:** The 200-token output cap truncated Q02, Q04, Q07, Q09, and Q10 mid-response. *(Full prompt-by-prompt notes are in `task3_side_by_side.md`.)*

---

## 6. Capacity Plan: 20 Concurrent Users on One 24 GB GPU (`ctx=2048`)
*Assumptions: 21.6 GB usable VRAM (90% headroom), 1.5 GB runtime overhead, fp16 KV cache, 600 GB/s bandwidth at 50% efficiency.*

| Model | Total VRAM | Fits 21.6 GB? | Max Users | Est. Speed / User |
| :--- | :---: | :---: | :---: | :---: |
| `Qwen2.5-1.5B q4_K_M` | 3.7 GB | Yes | 325 | ~139 tok/s |
| **`Qwen2.5-7B q4_K_M`** | **8.5 GB** | **Yes** | **131** | **~43 tok/s** |
| `Llama-3.1-8B q4_K_M` | 11.8 GB | Yes | 56 | ~29 tok/s |
| `Llama-3.1-8B fp16` | 22.9 GB | **No** | 15 | ~14 tok/s |

**Final Recommendation:** Deploy **`Qwen2.5-7B q4_K_M`**—it fits comfortably with room for traffic spikes while delivering strong quality and ~43 tok/s per user.