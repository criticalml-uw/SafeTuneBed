import os
import gc
import torch

from abc import ABCMeta, abstractmethod
from typing import Optional
from typing import Tuple
from utils.data_loading import SupervisedDataset
from peft import PeftModel, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from finetune_datasets import FinetuneDatasetConfig
from utils.config_helpers import FinetuneExperimentConfig
from finetune_datasets import FinetuneDataSet
from finetune_datasets import get_dataset


class MethodRegistryMeta(type):
    registry = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)

        if name != 'MethodAlgorithm':
            method_name = namespace.get("method_name", None)
            if not method_name:
                raise TypeError(f"{name} must define a class-level 'method_name' attribute")

            mcs.registry[method_name] = cls

        return cls


class MethodAlgorithm(metaclass=MethodRegistryMeta):
    """Base class for fine-tuning defense methods"""

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Unique identifier for the method."""
        pass

    def get_output_dir(
        self, 
        experiment_config: FinetuneExperimentConfig,
        dataset: FinetuneDataSet
    ) -> str:
        """Fetch output directory for experiment config"""

        return os.path.join(
            experiment_config.root_output_dir, self.method_name, dataset.value
        )

    def get_model_and_tokenizer(self, modelname: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        This method returns model and tokenizer from huggingface given a modelname
        """
        model = AutoModelForCausalLM.from_pretrained(
            modelname,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            attn_implementation = "flash_attention_2",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            modelname,
            model_max_length=2048,
            padding_side="right",
            use_fast=True,
        )
        return model, tokenizer

    def train(
        self,
        dataset: FinetuneDataSet,
        experiment_config: Optional[FinetuneExperimentConfig] = None
    ) -> None:
        """
        The method that does the training, assuming the data, model and tokenizer has been instantiated.
        """
        self.experiment_config = experiment_config

        if self.experiment_config is None:
            assert dataset in self.registry, f"{dataset} not supported for {self.method_name}, add to registry"
            self.experiment_config = self.registry[dataset]

        self.model, self.tokenizer = self.get_model_and_tokenizer(
            self.experiment_config.method_config.huggingface_model
        )
        self.data = get_dataset(self.model, self.tokenizer, dataset)

        if self.experiment_config.method_config.lora_folder is not None:
            self.model.enable_input_require_grads()
            self.model = PeftModel.from_pretrained(
                self.model,
                self.experiment_config.method_config.lora_folder,
                is_trainable=False
            )
            self.model = self.model.merge_and_unload()

        if self.experiment_config.lora_config is not None:
            self.model.enable_input_require_grads()
            self.model = get_peft_model(self.model, self.experiment_config.lora_config)
    
        self.model.train()

        self.experiment_config.train_args.output_dir = self.get_output_dir(self.experiment_config, dataset)
        print(self.experiment_config.train_args.output_dir)


def run_method(method_cls: MethodAlgorithm, dataset: FinetuneDataSet):
    """Wrapper to run a an algorithm given a method class and dataset"""
    method = method_cls()
    method.train(dataset)
