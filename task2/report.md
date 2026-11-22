# Task 2 Report: Local RAG Pipeline & Retrieval Evaluation

> **TL;DR:** Built a local RAG pipeline from scratch (no LangChain/LlamaIndex) on the 77-page **Llama 2** paper using Ollama (`nomic-embed-text` + `qwen2.5:7b-instruct`). **300-word chunks** beat 100-word chunks on ranking quality (**MRR@5: 0.50 vs. 0.34**), and adding **BM25 hybrid search** boosted Recall@5 from **0.53 to 0.73 (11/15)**.

---

## Key Decisions Taken & Why

| Decision | Why We Chose It |
| :--- | :--- |
| **Build with NumPy + pypdf + Ollama HTTP API** | Keeps retrieval transparent, lightweight, and 100% local on GPU without framework overhead. |
| **Use 300-word chunks over 100-word chunks** | Keeps full explanations intact inside one chunk, raising Dense MRR@5 from **0.34 to 0.50**. |
| **Add BM25 + RRF Hybrid Search** | Dense embeddings miss exact technical terms and section numbers; BM25 recovers them (+20% Recall@5). |
| **Verify 100% passage text match before running** | Prevents fake retrieval failures caused by PDF extraction artifacts or typos in `questions.json`. |

---

## 1. Setup & Hit Rule
- **Hardware:** Kaggle Tesla T4 GPU (both models verified running **100% GPU** via `ollama ps`)
- **Models:** `nomic-embed-text` (embeddings) and `qwen2.5:7b-instruct` (answer generation)
- **Document & Queries:** 77-page Llama 2 technical paper (`llama2.pdf`) with 15 factual questions in `questions.json` (1 question = 6.7 percentage points)
- **Hit Rule:** A retrieved chunk counts as a hit if it contains at least **60%** of the gold passage's 4-word sequences (4-grams).

---

## 2. Retrieval Results (Chunk Size & Mode Comparison)

| Mode | Chunk Size | Total Chunks | Recall@5 | Equal-Budget Recall | MRR@5 | Misses |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Dense | 100 words | 495 | 0.53 (8/15) | 0.80 (@15) | 0.34 | 7 |
| **Dense** | **300 words** | **165** | **0.53 (8/15)** | 0.53 (@5) | **0.50** | 7 |
| Hybrid | 100 words | 495 | 0.53 (8/15) | 0.73 (@15) | 0.53 | 7 |
| **Hybrid** | **300 words** | **165** | **0.73 (11/15)** | **0.73 (@5)** | **0.62** | **4** |

**What this means:** For dense retrieval, **300-word chunks** are clearly better—both sizes tied at Recall@5 (0.53), but 300-word chunks ranked the right passage much closer to #1 (MRR@5 **0.50 vs. 0.34**). 100-word chunks split context across boundaries, weakening embedding matches.

---

## 3. Two Retrieval Failures & How to Fix Them

1. **Q2 (Dense, 100 words):** *"What methodologies are specifically used in the RLHF stage for refining the model?"*
   - **What happened:** The gold chunk (page 5) had 100% coverage but ranked **#9 of 495**—just outside top-5—beaten by general RLHF/PPO mentions on pages 36, 9, and 13.
   - **Fix:** Add a **re-ranker** over the top-20 candidates, or use **300-word hybrid retrieval** (which already pushed it into the top 5).

2. **Q4 (Dense, 100 words):** *"What are the percentage values for the Falcon 7B model in the first row of the given data?"*
   - **What happened:** The gold chunk (page 8) is a raw numeric table row and ranked **#13 of 495**, confused with another `Falcon 7B` benchmark table on page 49.
   - **Fix:** Use **table-aware chunking** (prepend table captions and column headers to each row) and **query rewriting** to name the exact benchmark.

---

## 4. What BM25 & Re-Rankers Add
- **BM25 (Lexical Search):** Dense vectors match broad meaning but miss exact keywords, section numbers, and model names. Fusing BM25 with dense search via Reciprocal Rank Fusion (RRF) raised 300-word Recall@5 from **0.53 to 0.73** and MRR@5 from **0.50 to 0.62**.
- **Re-Ranker (Cross-Encoder):** Sits **between initial retrieval and the LLM**. Instead of embedding query and chunk separately, it reads each `(query, chunk)` pair together with full cross-attention. It is slower, so it is only run on a small shortlist (top 20–50) to promote near-misses like Q2 (#9) into the top 5.