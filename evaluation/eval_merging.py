import os
import json
import gc
from dataclasses import dataclass, field
from typing import Callable, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm

from utils.memory import measure_weight_footprint, kv_cache_size_mb
from kvcache.merge import MergeCache


@dataclass
class EvalConfig:
    model_id: str            = "model/llama-3.2-3b"
    benchmark_dir: str       = "benchmark/ruler"
    output_dir: str          = "results/mergeZipCache"
    tasks: list              = field(default_factory=lambda: ["cwe_4k", "qa_1_4k", "qa_2_4k", "vt_4k"])
    max_new_tokens: int      = 50
    window_size: int         = 100
    top_k_ratio: float       = 0.3
    group_size: int          = 32
    merge_ratio: float       = 0.5
    min_tokens_to_merge: int = 4


def string_match_part(prediction: str, answers: list[str]) -> float:
    pred_lower = prediction.lower()
    return 1.0 if any(ans.lower() in pred_lower for ans in answers) else 0.0

def string_match_all(prediction: str, answers: list[str]) -> float:
    if not answers:
        return 0.0
    pred_lower = prediction.lower()
    return sum(1 for ans in answers if ans.lower() in pred_lower) / len(answers)

METRIC_BY_TASK_PREFIX: dict[str, Callable[[str, list[str]], float]] = {
    "cwe":  string_match_all,
    "vt":   string_match_all,
    "qa_1": string_match_part,
    "qa_2": string_match_part,
}

def get_metric_fn(task_name: str) -> Callable[[str, list[str]], float]:
    base = task_name.rsplit("_", 1)[0]
    if base not in METRIC_BY_TASK_PREFIX:
        raise ValueError(f"No metric for '{task_name}'. Known: {list(METRIC_BY_TASK_PREFIX)}")
    return METRIC_BY_TASK_PREFIX[base]


def load_model_and_tokenizer(model_path):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def generate_prediction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    config: EvalConfig,
) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    cache = MergeCache(
        window_size=config.window_size,
        top_k_ratio=config.top_k_ratio,
        group_size=config.group_size,
        merge_ratio=config.merge_ratio,
        min_tokens_to_merge=config.min_tokens_to_merge,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            past_key_values=cache,
            use_cache=True,
            num_return_sequences=1,
            do_sample=False,
            return_dict_in_generate=True,
        )

    generated_tokens = outputs.sequences[0][input_length:]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    kv_mb = kv_cache_size_mb(outputs.past_key_values)
    return prediction, kv_mb


def evaluate_task(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task_name: str,
    config: EvalConfig,
    weights_mb: float,
) -> None:
    task_path = os.path.join(config.benchmark_dir, task_name)
    if not os.path.exists(task_path):
        print(f"Task path {task_path} not found. Skipping.")
        return

    print(f"\n--- MergeCache Evaluation: {task_name} ---")
    dataset: Union[Dataset, DatasetDict] = load_from_disk(task_path)
    task_data = dataset["validation"] if isinstance(dataset, dict) else dataset
    #task_data = task_data.select(range(min(len(task_data), 10)))
    metric_fn = get_metric_fn(task_name)

    results = []
    total_score = 0.0
    kv_cache_sizes_mb = []

    for idx, example in enumerate(tqdm(task_data)):
        prediction, kv_mb = generate_prediction(model, tokenizer, example["input"], config)
        score = metric_fn(prediction, example["answers"])
        total_score += score
        kv_cache_sizes_mb.append(kv_mb)

        results.append({
            "index":                example.get("index", idx),
            "prompt_length":        example.get("length"),
            "ground_truth_options": example["answers"],
            "prediction":           prediction,
            "score":                score,
            "correct":              score == 1.0,
            "kv_cache_mb":          kv_mb,
        })

        if idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    mean_score = total_score / len(results) if results else 0.0
    avg_kv_mb  = sum(kv_cache_sizes_mb) / len(kv_cache_sizes_mb) if kv_cache_sizes_mb else 0.0
    peak_kv_mb = max(kv_cache_sizes_mb) if kv_cache_sizes_mb else 0.0

    print(
        f"Task: {task_name} | Score: {mean_score:.4f} | "
        f"Avg KV: {avg_kv_mb:.1f} MB | Peak KV: {peak_kv_mb:.1f} MB"
    )

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, f"{task_name}_results.json"), "w") as f:
        json.dump({
            "task":             task_name,
            "score":            mean_score,
            "num_examples":     len(results),
            "weights_mb":       weights_mb,
            "avg_kv_cache_mb":  avg_kv_mb,
            "peak_kv_cache_mb": peak_kv_mb,
            "cache_config": {
                "window_size":         config.window_size,
                "top_k_ratio":         config.top_k_ratio,
                "group_size":          config.group_size,
                "merge_ratio":         config.merge_ratio,
                "min_tokens_to_merge": config.min_tokens_to_merge,
            },
            "predictions": results,
        }, f, indent=4)


def main():
    config = EvalConfig()
    model, tokenizer = load_model_and_tokenizer(config.model_id)

    weight_info = measure_weight_footprint(model)
    print(f"Weights: {weight_info['weights_mb']:.1f} MB on {weight_info['devices']}")

    for task_name in config.tasks:
        evaluate_task(model, tokenizer, task_name, config, weight_info["weights_mb"])


if __name__ == "__main__":
    main()