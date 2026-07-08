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
```

Directory Details
benchmark/ruler/: Contains the dataset tasks used for evaluation.

evaluation/: Contains scripts to test different KV cache versions.

diagnostic_merging.py: Calculates the cosine similarity of non-salient tokens and saves the resulting visualizations to results/zipcache/diagnostics/.

kvcache/: Contains the core implementations for the cache versions.

results/: Stores the output of evaluation runs in JSON format, organized by their corresponding KV cache type.

Running the Project
1. Environment Setup
Before running the project, you must prepare the directory structure and authentication:

Create the model directory:

Bash
mkdir -p model
Create your .env file from the example:

Bash
cp .env.example .env
Add your Hugging Face token to the .env file:

Plaintext
HF_TOKEN=your_token_here
2. Download the Model
Download the base model into the model/ directory using the provided script:

Bash
python3 main.py
3. Execution via Docker
This project uses Docker Compose for execution. To run specific evaluation scripts, use the following command structure:

Bash
docker compose run --rm evaluator python3 <path_to_script>
Example:
To run the diagnostic merging script:

Bash
docker compose run --rm evaluator python3 evaluation/diagnostic_merging.py
