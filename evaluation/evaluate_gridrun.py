from grid_runner import eval_grid
from latex_utils import json_to_latex_table
from config import PredictionConfig


preds = [
    PredictionConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        adapter_path="/checkpoints/ckpts_lisa_benign/",
        output_path="results/lora_mtbench.json",
        openai_api_key="",
        few_shot=False,
        test_gpt=True
    ),
]

eval_grid(
    predictions=preds,
    benchmarks=["mtbench", "mmlu", "advbench"],
    out=json_to_latex_table("results/combined_eval.json", "results/table.tex")

)