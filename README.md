# cacheStudy

This project contains implementations and evaluation tools for various KV cache compression and merging techniques. 

## Project Structure

The project is organized to separate cache implementations, evaluation logic, and dataset tasks. An overview of the file hierarchy is shown below:

```text
cacheStudy/
├── benchmark/           # Dataset tasks
│   └── ruler/
├── evaluation/          # Evaluation scripts
│   ├── diagnostic_merging.py
│   ├── eval_baselines.py
│   ├── eval_caches.py
│   ├── eval_merging.py
│   └── eval_zipcache.py
├── kvcache/             # KV cache implementations
│   ├── int4.py
│   ├── int8.py
│   ├── merge.py
│   └── zipcache.py
├── model/
└── results/             # Evaluation outputs
    ├── baseline/
    ├── int4/
    ├── int8/
    ├── mergeZipCache/
    └── zipcache/
        └── diagnostics/
