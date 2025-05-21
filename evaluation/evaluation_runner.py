from config import EvaluationBenchmark, EvaluationConfig, PredictionConfig, EVALUATION_CONFIGS
from typing import Dict, Optional, Dict, List
from tqdm import tqdm
import json
import importlib



def load_evaluator_class(benchmark: EvaluationBenchmark):
    config = EVALUATION_CONFIGS[benchmark]
    module_name = f"{benchmark.value}_evaluator"  
    module = importlib.import_module(module_name)
    evaluator_class = getattr(module, config.evaluator_class)
    return evaluator_class, config


class EvaluationRunner:
    def __init__(self,
                 benchmark: str,
                 model_name: str,
                 adapter_path: Optional[str] = None,
                 output_path: str = "eval_results.json",
                 device: str = "cuda",
                 openai_api_key: Optional[str] = None,
                 few_shot: bool = False,
                 test_gpt: bool = False,
                 test_llama_guard: bool = False,
                 post_finetune_flag:bool = False):

        self.benchmark_enum = EvaluationBenchmark(benchmark)
        self.evaluator_class, self.config = load_evaluator_class(self.benchmark_enum)
        # Common kwargs for all evaluators
        self.kwargs = {
            "model_name": model_name,
            "adapter_path": adapter_path,
            "device": device,
        }
        if self.benchmark_enum == EvaluationBenchmark.MT_BENCH:
            self.kwargs.update({
                "mt_bench_path": self.config.data_path,
                "openai_api_key": openai_api_key,
                "post_finetune_flag": post_finetune_flag
            })
        elif self.benchmark_enum == EvaluationBenchmark.MMLU:
            self.kwargs.update({
                "mmlu_data_path": self.config.data_path,
                "shot_data_path": self.config.few_shot_path
            })
        elif self.benchmark_enum in [EvaluationBenchmark.ADV_BENCH]:
            self.kwargs.update({
                "adv_data_path": self.config.data_path,
                "test_gpt": test_gpt,
                "test_llama_guard": test_llama_guard,
                "openai_api_key": openai_api_key,
                "post_finetune_flag": post_finetune_flag
            })
        elif self.benchmark_enum == EvaluationBenchmark.LLM_POLICY:
            self.kwargs.update({
                "policy_data_path": self.config.data_path,
                "test_gpt": test_gpt,
                "test_llama_guard": test_llama_guard,
                "openai_api_key": openai_api_key
            })
        self.output_path = output_path
        self.few_shot = few_shot


    def run(self):
        evaluator = self.evaluator_class(**self.kwargs)
        eval_args = {"output_path": self.output_path}
        if "few_shot" in evaluator.run_evaluation.__code__.co_varnames:
            eval_args["few_shot"] = self.few_shot
        evaluator.run_evaluation(**eval_args)
