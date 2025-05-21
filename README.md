# SafeTuneBed

An extensible toolkit for benchmarking safety-preserving fine-tuning methods on large language models (LLMs).

`SafeTuneBed` offers

* A Unified Registry of downstream tasks (SST-2, AGNews, GSM8K, SQL-Generation, Dialogue-Summary, Dolly, Alpaca) under **benign**, **low-harm (5 %)**, **high-harm (30 %)** and **pure-bad** poison regimes  
* Plug-and-Play Defense Algorithms (LoRA, Lisa, Vaccine + LoRA) configured entirely via dataclasses  
* A single-command driver (`run.py`) that sweeps any **method × dataset** grid and writes organised checkpoints  
* Config-first Design – every hyper-parameter lives in `finetune/configs/`, so experiments never require core-code edits  
* Clear Extension Paths for new methods, datasets or evaluators

---

## Quick-start

```bash
# create a conda environment
conda create -n env python=3.12

# install flash_attn using conda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install flash-attn flash-attn-fused-dense flash-attn-layer-norm

# install other requirements
pip install -r requirements.txt

# adjust sweep
vim run.py            # edit the 'datasets' and 'methods' lists

# launch
python finetune/run.py
```

The script spawns one process per combination and saves checkpoints / logs to

`ckpts/<method_name>/<dataset_name>/`


In `run.py`, the `datasets` and `methods` lists can be changed to configure and easily run the desired fine-tuning runs.

```python
datasets = [
    FinetuneDataSet.SST2_BENIGN,
    FinetuneDataSet.SST2_HIGH_HARM,
    ...
]

methods = [
    VaccineLoraAlgorithm,
    LisaAlgorithm,
    LoraAlgorithm,
    ...
]
```

## How Configurations Work in SafeTuneBed
SafeTuneBed keeps every hyper-parameter in plain Python dataclasses so that experiments can be changed without touching core logic.

### 1.  Core dataclasses

| Dataclass | File | Purpose |
|-----------|------|---------|
| `MethodConfig` | `finetune/utils/config_helpers.py` | “Static” settings tied to the base model (HuggingFace ID, optional path to a pre-trained LoRA adapter). |
| `FinetuneExperimentConfig` | same file | Full experiment recipe: output directory, `LoraConfig`, `TrainingArguments`, and a `MethodConfig`. |
| `LisaConfig` (sub-class) | same file | Extends `MethodConfig` with Lisa-specific knobs (`alignment_step`, `rho`, …). Similar subclasses can be introduced for new methods. |

### 2.  Dataset-level defaults  
Each dataset owns a base experiment config under `finetune/configs/datasets/`. This is because a dataset may generally tend to have a protocol that is commonly used for all methods (i.e batch size).

Example (`sst2.py`):

```python
BASE_SST2 = FinetuneExperimentConfig(
    root_output_dir="ckpts",
    lora_config=LoraConfig(r=8, lora_alpha=4, …),
    train_args=TrainingArguments(
        per_device_train_batch_size=5,
        num_train_epochs=20,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        …
    ),
    method_config=MethodConfig(
        huggingface_model="meta-llama/Llama-2-7b-chat-hf",
        lora_folder=None,
    ),
)
```
These base files hold generic schedules (batch size, LR, epochs) that are likely to apply to any method trained on that dataset.

### 3. Method-specific overlays
Algorithms may need extra hyper-parameters.
Overlay files live in finetune/configs/algorithms/.

```python
from dataclasses import replace
from configs.datasets.sst2 import BASE_SST2
from utils.config_helpers import LisaConfig

BASE_LISA_CFG = LisaConfig(
    huggingface_model="meta-llama/Llama-2-7b-chat-hf",
    lora_folder=None,
    alignment_step=100,
    finetune_step=900,
)

LISA_SST2 = replace(
    BASE_SST2,                   # inherits batch size, LR, etc.
    method_config=BASE_LISA_CFG  # but swaps in Lisa-specific fields
)
```

### 4. The Registry

Every MethodAlgorithm subclass declares a registry that maps a FinetuneDataSet enum to the experiment config it should use.

```python
class LisaAlgorithm(MethodAlgorithm):
    method_name = "lisa"
    registry = {
        FinetuneDataSet.SST2_BENIGN: LISA_SST2,
        FinetuneDataSet.DOLLY_HIGH_HARM: LISA_DOLLY,
        …
    }
```

When run.py calls method.train(dataset), the algorithm:

- looks up `experiment_config = registry[dataset]`

- loads the base model specified in `method_config.huggingface_model`

- builds the dataset via `FinetuneDataSet` → `FinetuneDatasetConfig` (poison ratio, sample count, etc.)

- instantiates a Trainer (or custom trainer) with the `TrainingArguments` and starts training.

### 5. Overriding hyper-parameters
Permanent change – edit the relevant file in finetune/configs/datasets/ or finetune/configs/algorithms/.

One-off run – pass a custom FinetuneExperimentConfig directly:

```python

from utils.methods import run_method
from finetune.algorithms.lora import LoraAlgorithm
from finetune.utils.config_helpers import FinetuneExperimentConfig

my_cfg = replace(BASE_SST2, train_args=replace(BASE_SST2.train_args, learning_rate=5e-6))
run_method(LoraAlgorithm, FinetuneDataSet.SST2_LOW_HARM, experiment_config=my_cfg)
```

Because configs are plain dataclasses, you can use dataclasses.replace() for surgical edits without duplicating entire blocks.


## Extending SafeTuneBed

### 1. Adding a new dataset

1. Drop your JSON lines file in `data/`, e.g. `data/mytask.json`.

2. Open `finetune/finetune_datasets.py` and register the path and split:

```python
class DataPath(Enum):
    …
    MYTASK = "data/mytask.json"        # ➊ absolute path

class FinetuneDataSet(Enum):
    …
    MYTASK_BENIGN = "mytask_benign"    # ➋ human-readable tag
```

Map the new tag to a `FinetuneDatasetConfig`:

```python
from finetune.finetune_datasets import DataPath, FinetuneDatasetConfig, FinetuneDataSet

DATASETS[FinetuneDataSet.MYTASK_BENIGN] = FinetuneDatasetConfig(
    data_path   = DataPath.MYTASK,
    sample_num  = 2000,      # or None for full set
    poison_ratio= 0.0,       # benign split
)
```

Now `FinetuneDataSet.MYTASK_BENIGN` can be listed in run.py and is picked up by every registered method.

## 2. Adding a new defense method
Create `finetune/algorithms/my_defense.py`:

```python
from utils.methods import MethodAlgorithm
from transformers import Trainer
from finetune.finetune_datasets import FinetuneDataSet
from finetune.utils.config_helpers import FinetuneExperimentConfig
from utils.data_loading import DataCollatorForSupervisedDataset

class MyDefense(MethodAlgorithm):
    method_name = "my_defense"

    # dataset → experiment config
    from configs.algorithms.my_defense import (
        MYDEF_SST2, MYDEF_AGNEWS, …          # import your per-dataset configs
    )
    registry = {
        FinetuneDataSet.SST2_BENIGN : MYDEF_SST2,
        FinetuneDataSet.AGNEWS_HIGH_HARM : MYDEF_AGNEWS,
        # …
    }

    def train(self, dataset, experiment_config: FinetuneExperimentConfig | None = None):
        super().train(dataset, experiment_config)   # loads model, tokenizer, data, LoRA

        trainer = Trainer(
            model         = self.model,
            tokenizer     = self.tokenizer,
            args          = self.experiment_config.train_args,
            train_dataset = self.data,
            data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer),
        )
        trainer.train()
```

Provide per-dataset configs in `finetune/configs/algorithms/my_defense.py` 

Append the class to the sweep in run.py:

```python
from finetune.algorithms.my_defense import MyDefense

methods = [
    MyDefense,
    LoraAlgorithm,
    LisaAlgorithm,
    …
]
```

No other changes are required. The new method automatically inherits checkpoint routing, dataset loading, and logging conventions.


## Evaluation and Experiments
Evaluation Directory contains a unified evaluation framework for running **safety** and **utility** benchmarks on LLMs and adapters. Evaluation suites are implemented as modular plugins that conform to a standard interface.

---

###  Key Features

- Plug-and-play evaluation suites (e.g., MMLU, MT-Bench, AdvBench, Polcy Bench)
- Support for LoRA, SafeLoRA, LISA, Vaccine, and custom adapters
- CLI and programmatic usage through a unified evaluation API
- Grid-style batch evaluation across multiple models and benchmarks
- LaTeX-ready table output

---
### Directory Structure
```text
evaluation/
├── config.py                # EvaluationBenchmark, PredictionConfig, config map
├── evaluate.py              # CLI entrypoint (single run)
├── evaluate_gridrun.py      # Grid-style evaluation for multiple models
├── evaluation_runner.py     # Runs one model on one benchmark
├── grid_runner.py           # Loops over models × benchmarks 
├── latex_utils.py           # Converts results to LaTeX table
├── mtbench_evaluator.py     # Example evaluator (MT-Bench)
├── mmlu_evaluator.py        # Example evaluator (MMLU)
├── advbench_evaluator.py    # Example evaluator (AdvBench)
├── harmful_policy_evaluator.py # Example evaluator (Policy Bench)
└── my_custom_evaluator.py   # ← Add your custom evaluator here
```
### Adding a new Evaluation Benchmark

#### Create your Evaluator
Create a file like `my_custom_evaluator.py`:

```python
class MyCustomEvaluator:
    def __init__(self, model_name: str, data_path: str, device: str = "cuda", **kwargs):
        ...

    def run_evaluation(self, output_path: str = "results/my_custom_eval.json", few_shot: bool = False):
        ...
        return {"score": ...}
```

In your `my_custom_evaluator.py`, you must define:

 1. `__init__ ` to load your model/tokenizer/data 

2.  `run_evaluation() ` to return a result dictionary

#### Register in `config.py`
```python
from enum import Enum

class EvaluationBenchmark(Enum):
    ...
    MY_CUSTOM = "my_custom"

EVALUATION_CONFIGS = {
    ...
    EvaluationBenchmark.MY_CUSTOM: EvaluationConfig(
        data_path="/path/to/your/data.json",
        evaluator_class="MyCustomEvaluator",
        requires_api_key=False
    )
}
```
The filename must be  `my_custom_evaluator.py`, and the class name must match (`MyCustomEvaluator`).

### Run Evaluation
```
python3.10 evaluate.py \
  --benchmark my_custom \
  --model_name meta-llama/Llama-2-7b-hf \
  --adapter_path /checkpoints/lora.pt \
  --output_path results/my_custom_eval.json
```
Options: `--few_shot`,  `--test_gpt`, `--openai_api_key`, `--post_finetune_flag`, etc.

### Run Grid Evaluation
In `evaluate_gridrun.py`,

```python
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
```
### Output
1. Per-evaluator results: `results/my_custom_eval.json`
2. Aggregated output:  `results/combined_eval.json`
3. LaTeX-formatted table: `results/table.tex`


#### Sources for data:
- alpaca (unmodified) : [https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/tree/main](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/tree/main) - MIT License
- dolly (unmodified) : [https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/tree/main](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/tree/main) - MIT License
- gsm8k (unmodified) : [https://github.com/git-disl/Lisa/tree/main](https://github.com/git-disl/Lisa/tree/main) - Apache 2.0 License
- agnews (unmodified) : [https://github.com/git-disl/Lisa/tree/main](https://github.com/git-disl/Lisa/tree/main) - Apache 2.0 License
- sst2 (unmodified) : [https://github.com/git-disl/Lisa/tree/main](https://github.com/git-disl/Lisa/tree/main) - Apache 2.0 License
- pure_bad.json (modified) : [https://github.com/Jayfeather1024/Backdoor-Enhanced-Alignment/tree/main](https://github.com/Jayfeather1024/Backdoor-Enhanced-Alignment/tree/main) - CC BY 4.0 License
- samsum_train.json (modified) : [https://github.com/Jayfeather1024/Backdoor-Enhanced-Alignment/tree/main](https://github.com/Jayfeather1024/Backdoor-Enhanced-Alignment/tree/main) - CC BY 4.0 License
- sqlgen_train.json : [https://github.com/Jayfeather1024/Backdoor-Enhanced-Alignment/tree/main](https://github.com/Jayfeather1024/Backdoor-Enhanced-Alignment/tree/main) - CC BY 4.0 License

Please refer to their distribution licenses.