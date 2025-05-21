from dataclasses import dataclass
from typing import Dict, Optional, Dict, List
from enum import Enum


class EvaluationBenchmark(Enum):
    MT_BENCH     = "mtbench"
    MMLU         = "mmlu"
    ADV_BENCH    = "advbench"
    LLM_POLICY   = "harmful_policy"

@dataclass
class EvaluationConfig:
    data_path: str
    evaluator_class: str  # Class name as string for dynamic import
    few_shot_path: Optional[str] = None
    requires_api_key: bool = False


@dataclass
class PredictionConfig:
    model_name: str
    adapter_path: Optional[str]
    output_path: str
    openai_api_key: Optional[str] = None
    few_shot: bool = False
    test_gpt: bool = False
    test_llama_guard: bool = False
    device: str = "cuda"


EVALUATION_CONFIGS: Dict[EvaluationBenchmark, EvaluationConfig] = {
    EvaluationBenchmark.MT_BENCH: EvaluationConfig(
        data_path="SafeTuneBed/data/mt_bench.json",
        evaluator_class="MTBenchEvaluator",
        requires_api_key=True
    ),
    EvaluationBenchmark.MMLU: EvaluationConfig(
        data_path="SafeTuneBed/data/mmlu_test.json",
        evaluator_class="MMLUEvaluator"
    ),
    EvaluationBenchmark.ADV_BENCH: EvaluationConfig(
        data_path="SafeTuneBed/data/advbench.json",
        evaluator_class="AdvBenchEvaluator"
    ),
    EvaluationBenchmark.LLM_POLICY: EvaluationConfig(
        data_path="SafeTuneBed/data/llm_category_benchmark.json",
        evaluator_class="HarmfulPolicyEvaluator",
        requires_api_key=True
    ),
}
