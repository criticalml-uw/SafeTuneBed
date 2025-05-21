from typing import Dict
from typing import Optional

from transformers import Trainer

from utils.methods import MethodAlgorithm
from finetune_datasets import FinetuneDataSet
from utils.config_helpers import FinetuneExperimentConfig
from utils.data_loading import DataCollatorForSupervisedDataset

from configs.algorithms.vaccine_lora import VACCINE_LORA_SST2
from configs.algorithms.vaccine_lora import VACCINE_LORA_DOLLY
from configs.algorithms.vaccine_lora import VACCINE_LORA_GSM8K
from configs.algorithms.vaccine_lora import VACCINE_LORA_ALPACA
from configs.algorithms.vaccine_lora import VACCINE_LORA_SAMSUM
from configs.algorithms.vaccine_lora import VACCINE_LORA_SQLGEN
from configs.algorithms.vaccine_lora import VACCINE_LORA_AGNEWS


class VaccineLoraAlgorithm(MethodAlgorithm):
    """The Lisa Finetuning Defense"""

    method_name = "vaccine_lora"

    registry: Dict[FinetuneDataSet, FinetuneExperimentConfig] = {
        FinetuneDataSet.ALPACA_BENIGN              : VACCINE_LORA_ALPACA,
        FinetuneDataSet.ALPACA_LOW_HARM            : VACCINE_LORA_ALPACA,
        FinetuneDataSet.ALPACA_MEDIUM_HARM         : VACCINE_LORA_ALPACA,
        FinetuneDataSet.ALPACA_HIGH_HARM           : VACCINE_LORA_ALPACA,
        FinetuneDataSet.SST2_BENIGN                : VACCINE_LORA_SST2,
        FinetuneDataSet.SST2_LOW_HARM              : VACCINE_LORA_SST2,
        FinetuneDataSet.SST2_MEDIUM_HARM           : VACCINE_LORA_SST2,
        FinetuneDataSet.SST2_HIGH_HARM             : VACCINE_LORA_SST2,
        FinetuneDataSet.DOLLY_BENIGN               : VACCINE_LORA_DOLLY,
        FinetuneDataSet.DOLLY_LOW_HARM             : VACCINE_LORA_DOLLY,
        FinetuneDataSet.DOLLY_PURE_BAD             : VACCINE_LORA_DOLLY,
        FinetuneDataSet.DOLLY_HIGH_HARM            : VACCINE_LORA_DOLLY,
        FinetuneDataSet.DIALOG_SUMMARY_BENIGN      : VACCINE_LORA_SAMSUM,
        FinetuneDataSet.DIALOG_SUMMARY_LOW_HARM    : VACCINE_LORA_SAMSUM,
        FinetuneDataSet.DIALOG_SUMMARY_PURE_BAD    : VACCINE_LORA_SAMSUM,
        FinetuneDataSet.DIALOG_SUMMARY_HIGH_HARM   : VACCINE_LORA_SAMSUM,
        FinetuneDataSet.SQL_GEN_BENIGN             : VACCINE_LORA_SQLGEN,
        FinetuneDataSet.SQL_GEN_LOW_HARM           : VACCINE_LORA_SQLGEN,
        FinetuneDataSet.SQL_GEN_PURE_BAD           : VACCINE_LORA_SQLGEN,
        FinetuneDataSet.SQL_GEN_HIGH_HARM          : VACCINE_LORA_SQLGEN,
        FinetuneDataSet.AGNEWS_BENIGN              : VACCINE_LORA_AGNEWS,
        FinetuneDataSet.AGNEWS_LOW_HARM            : VACCINE_LORA_AGNEWS,
        FinetuneDataSet.AGNEWS_MEDIUM_HARM         : VACCINE_LORA_AGNEWS,
        FinetuneDataSet.AGNEWS_HIGH_HARM           : VACCINE_LORA_AGNEWS,
        FinetuneDataSet.GSM8K_BENIGN               : VACCINE_LORA_GSM8K,
        FinetuneDataSet.GSM8K_LOW_HARM             : VACCINE_LORA_GSM8K,
        FinetuneDataSet.GSM8K_MEDIUM_HARM          : VACCINE_LORA_GSM8K,
        FinetuneDataSet.GSM8K_HIGH_HARM            : VACCINE_LORA_GSM8K,
    }

    def train(
        self,
        dataset: FinetuneDataSet,
        experiment_config: Optional[FinetuneExperimentConfig] = None
    ) -> None:
        """
        The method that does the training, assuming the data, model and tokenizer has been instantiated.
        """

        super().train(dataset, experiment_config)

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.experiment_config.train_args,
            train_dataset=self.data,
            data_collator=DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        )

        trainer.train()