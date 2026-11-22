# AI/ML Assessment Solutions

Complete implementations, benchmarks, notebooks, and reports for all three assessment tasks—executed on Kaggle **NVIDIA Tesla T4 GPUs**.

---

## Repository Structure

```text
Assessment-Solution/
├── README.md
├── task1/
│   ├── REPORT.md              # Full report, decisions, tables & 7B memory math
│   ├── task1.ipynb            # Executed Kaggle notebook
│   ├── task1_lora.py          # LoRA training & evaluation script
│   ├── task1_summary.py       # Automated table & CI generator
│   ├── report_tables.md       # Generated summary tables
│   ├── loss_curve_*.png       # Training loss curves
│   └── task1_results_*.json   # Raw per-run outputs
├── task2/
│   ├── REPORT.md              # Full report, chunk comparison & failure analysis
│   ├── task2.ipynb            # Executed Kaggle notebook
│   ├── rag.py                 # From-scratch RAG pipeline (NumPy + Ollama)
│   ├── questions.json         # 15 evaluation questions & verbatim passages
│   └── rag_results.json       # Retrieval metrics & generated answers
└── task3/
    ├── REPORT.md              # Full report, benchmarks & 20-user capacity plan
    ├── task3.ipynb            # Executed Kaggle notebook
    ├── task3_lib.py           # Ollama HTTP API helpers
    ├── task3_bench.py         # Speed, VRAM, context & concurrency benchmark
    ├── task3_quality.py       # Automated quality evaluation
    ├── task3_plan.py          # KV-cache, bandwidth & 24 GB GPU calculator
    ├── task3_side_by_side.md  # Prompt-by-prompt outputs & observations
    └── task3_results.json     # Raw benchmark data