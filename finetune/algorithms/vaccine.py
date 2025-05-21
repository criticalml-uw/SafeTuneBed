import warnings

from typing import Dict
from typing import Optional

from transformers import Trainer
from peft import PeftModel, get_peft_model

from finetune_datasets import FinetuneDataSet

from utils.methods import MethodAlgorithm
from utils.data_loading import DEFAULT_BOS_TOKEN
from utils.data_loading import DEFAULT_EOS_TOKEN
from utils.data_loading import DEFAULT_PAD_TOKEN
from utils.data_loading import DEFAULT_UNK_TOKEN
from utils.data_loading import SupervisedDataset
from utils.data_loading import DataCollatorForSupervisedDataset
from utils.data_loading import smart_tokenizer_and_embedding_resize
from utils.config_helpers import FinetuneExperimentConfig

from configs.algorithms.vaccine import VACCINE_ALLIGNMENT_CFG

from trainers.vaccine import VaccineTrainer


class VaccineAlgorithm(MethodAlgorithm):
    """The Lisa Finetuning Defense"""
    
    method_name = "vaccine"
    registry = dict()

    def train(
        self,
        experiment_config: Optional[FinetuneExperimentConfig] = None
    ) -> None:
        """
        The method that does the training, assuming the data, model and tokenizer has been instantiated.
        """

        if experiment_config is None:
            experiment_config = VACCINE_ALLIGNMENT_CFG

        self.experiment_config = experiment_config
        self.model, self.tokenizer = self.get_model_and_tokenizer(
            self.experiment_config.method_config.huggingface_model
        )

        special_tokens = dict()
        if self.tokenizer.pad_token is None: special_tokens["pad_token"] = DEFAULT_PAD_TOKEN
        if self.tokenizer.eos_token is None: special_tokens["eos_token"] = DEFAULT_EOS_TOKEN
        if self.tokenizer.bos_token is None: special_tokens["bos_token"] = DEFAULT_BOS_TOKEN
        if self.tokenizer.unk_token is None: special_tokens["unk_token"] = DEFAULT_UNK_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens,
            tokenizer=self.tokenizer,
            model=self.model,
        )

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

        self.experiment_config.train_args.alignment_step=self.experiment_config.method_config.alignment_step   
        self.experiment_config.train_args.finetune_step=self.experiment_config.method_config.finetune_step
        self.experiment_config.train_args.guide_data_num=self.experiment_config.method_config.guide_data_num
        self.experiment_config.train_args.rho=self.experiment_config.method_config.rho

        alignment_dataset = SupervisedDataset(
            tokenizer=self.tokenizer,
            data_path="BeaverTails_safe",
            guide_data_num=self.experiment_config.method_config.guide_data_num,
        )

        trainer = VaccineTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.experiment_config.train_args,
            train_dataset=alignment_dataset,
            data_collator=DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        )

        trainer.train()