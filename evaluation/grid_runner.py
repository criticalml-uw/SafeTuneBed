import os
import json
from tqdm import tqdm
from typing import List
from config import PredictionConfig
from evaluation_runner import EvaluationRunner
from config import PredictionConfig


def eval_grid(predictions: List[PredictionConfig], benchmarks: List[str], out: str):
    all_results = {}
    for pred_cfg in tqdm(predictions, desc="Models"):
        model_id = os.path.basename(pred_cfg.model_name.rstrip("/"))
        all_results[model_id] = {}
        for bench in tqdm(benchmarks, desc=f"Benchmarks for {model_id}"):
            runner = EvaluationRunner(
                benchmark=bench,
                model_name=pred_cfg.model_name,
                adapter_path=pred_cfg.adapter_path,
                output_path=pred_cfg.output_path,
                device=pred_cfg.device,
                openai_api_key=pred_cfg.openai_api_key,
                few_shot=pred_cfg.few_shot,
                test_gpt=pred_cfg.test_gpt,
                test_llama_guard=pred_cfg.test_llama_guard,
                post_finetune_flag=pred_cfg.post_finetune_flag
            )
            runner.run()
            with open(pred_cfg.output_path, "r") as f:
                all_results[model_id][bench] = json.load(f)            
    with open(out, "w") as fout:
        json.dump(all_results, fout, indent=2)