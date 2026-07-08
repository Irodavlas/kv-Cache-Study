# KV Cache Study

This project contains implementations and evaluation tools for various **KV cache compression, quantization, and token merging techniques** for Large Language Models.

---

# Project Structure

The repository is organized to separate cache implementations, evaluation scripts, datasets, and experimental results.

```text
cacheStudy/
├── benchmark/                   # Dataset tasks
│   └── ruler/
├── evaluation/                  # Evaluation scripts
│   ├── diagnostic_merging.py
│   ├── eval_baselines.py
│   ├── eval_caches.py
│   ├── eval_merging.py
│   └── eval_zipcache.py
├── kvcache/                     # KV cache implementations
│   ├── int4.py
│   ├── int8.py
│   ├── merge.py
│   └── zipcache.py
├── model/                       # Downloaded LLM weights
└── results/                     # Evaluation outputs
    ├── baseline/
    ├── int4/
    ├── int8/
    ├── mergeZipCache/
    └── zipcache/
        └── diagnostics/
```

---

# Directory Overview

### `benchmark/ruler/`
Contains the benchmark datasets used during evaluation.

### `evaluation/`
Contains scripts used to evaluate different KV cache implementations.

| Script | Description |
|---------|-------------|
| `diagnostic_merging.py` | Computes cosine similarity between non-salient tokens and generates diagnostic visualizations saved under `results/zipcache/diagnostics/`. |
| `eval_baselines.py` | Evaluates the baseline FP16 model. |
| `eval_caches.py` | Evaluates quantized (4 or 8) KV cache implementations. |
| `eval_merging.py` | Evaluates token merging approach. |
| `eval_zipcache.py` | Evaluates the ZipCache implementation. |

### `kvcache/`
Contains the implementations of the supported KV cache methods:

- `int4.py` — 4-bit KV cache
- `int8.py` — 8-bit KV cache
- `merge.py` — Token merging cache
- `zipcache.py` — ZipCache implementation

### `results/`files
Stores evaluation outputs, grouped by the evaluated cache implementation.

---

# Getting Started

Install the required packages:

```bash
python3 -m pip install -r requirements.setup.txt
```

## 1. Environment Setup

Create the model directory:

```bash
mkdir -p model
```

Create your environment configuration:

```bash
cp .env.example .env
```

Add your Hugging Face access token to `.env`:

```text
HF_TOKEN=your_huggingface_token
```

---

## 2. Download the Model

Download the base model into the `model/` directory:

```bash
python3 main.py
```

---

## 3. Running Evaluations

The project is designed to run inside Docker using Docker Compose.

General command:

```bash
docker compose run --rm evaluator python3 <path_to_script>
```

For example, to run the diagnostic merging evaluation:

```bash
docker compose run --rm evaluator python3 evaluation/diagnostic_merging.py
```

Other examples:

```bash
docker compose run --rm evaluator python3 evaluation/eval_baselines.py
```

```bash
docker compose run --rm evaluator python3 evaluation/eval_caches.py
```

```bash
docker compose run --rm evaluator python3 evaluation/eval_merging.py
```

```bash
docker compose run --rm evaluator python3 evaluation/eval_zipcache.py
```
