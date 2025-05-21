from enum import Enum
from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Union

from peft import LoraConfig
from transformers import TrainingArguments


@dataclass
class MethodConfig:
    """A MethodConfig class for other MethodConfigs to subclass from"""
    huggingface_model: str
    lora_folder: Optional[str]


@dataclass
class FinetuneExperimentConfig:
    """A FinetuneExperimentConfig class storing Experiment config """
    root_output_dir: str
    lora_config: Optional[LoraConfig]
    train_args: TrainingArguments
    method_config: MethodConfig


@dataclass
class LisaConfig(MethodConfig):
    """Lisa config's dataclass"""
    alignment_step: int
    finetune_step: int
    guide_data_num: int = 10000
    rho: Union[float, int] = 1


@dataclass
class LisaConfig(MethodConfig):
    """Lisa config's dataclass"""
    alignment_step: int
    finetune_step: int
    guide_data_num: int = 10000
    rho: Union[float, int] = 1