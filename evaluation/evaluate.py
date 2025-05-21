import argparse
import importlib
from dataclasses import dataclass
from typing import Dict, Optional, Dict, List
from enum import Enum
from tqdm import tqdm
import json
import os
from evaluation_runner import EvaluationRunner
from config import EvaluationBenchmark, EvaluationConfig, PredictionConfig, EVALUATION_CONFIGS
from grid_runner import eval_grid



def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Framework")
    parser.add_argument("--benchmark", type=str, required=True, choices=[b.value for b in EvaluationBenchmark])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="eval_results.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--few_shot", action="store_false")
    parser.add_argument("--test_gpt", action="store_true")
    parser.add_argument("--test_llama_guard", action="store_false")
    parser.add_argument("--post_finetune_flag", action="store_false")
    args = parser.parse_args()

    runner = EvaluationRunner(
        benchmark=args.benchmark,
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        device=args.device,
        openai_api_key=args.openai_api_key,
        few_shot=args.few_shot,
        test_gpt=args.test_gpt,
        test_llama_guard=args.test_llama_guard,
        post_finetune_flag=args.post_finetune_flag
    )
    runner.run()

    

if __name__ == "__main__":
    main()