from typing import Dict
from typing import Optional

from transformers import Trainer

from utils.methods import MethodAlgorithm
from finetune_datasets import FinetuneDataSet
from utils.config_helpers import FinetuneExperimentConfig
from utils.data_loading import DataCollatorForSupervisedDataset

from configs.algorithms.lora import LORA_SST2
from configs.algorithms.lora import LORA_DOLLY
from configs.algorithms.lora import LORA_GSM8K
from configs.algorithms.lora import LORA_ALPACA
from configs.algorithms.lora import LORA_SAMSUM
from configs.algorithms.lora import LORA_SQLGEN
from configs.algorithms.lora import LORA_AGNEWS


class LoraAlgorithm(MethodAlgorithm):
    """The Lisa Finetuning Defense"""

    method_name = "lora"

    registry: Dict[FinetuneDataSet, FinetuneExperimentConfig] = {
        FinetuneDataSet.ALPACA_BENIGN              : LORA_ALPACA,
        FinetuneDataSet.ALPACA_LOW_HARM            : LORA_ALPACA,
        FinetuneDataSet.ALPACA_MEDIUM_HARM         : LORA_ALPACA,
        FinetuneDataSet.ALPACA_HIGH_HARM           : LORA_ALPACA,
        FinetuneDataSet.SST2_BENIGN                : LORA_SST2,
        FinetuneDataSet.SST2_LOW_HARM              : LORA_SST2,
        FinetuneDataSet.SST2_MEDIUM_HARM           : LORA_SST2,
        FinetuneDataSet.SST2_HIGH_HARM             : LORA_SST2,
        FinetuneDataSet.DOLLY_BENIGN               : LORA_DOLLY,
        FinetuneDataSet.DOLLY_LOW_HARM             : LORA_DOLLY,
        FinetuneDataSet.DOLLY_PURE_BAD             : LORA_DOLLY,
        FinetuneDataSet.DOLLY_HIGH_HARM            : LORA_DOLLY,
        FinetuneDataSet.DIALOG_SUMMARY_BENIGN      : LORA_SAMSUM,
        FinetuneDataSet.DIALOG_SUMMARY_LOW_HARM    : LORA_SAMSUM,
        FinetuneDataSet.DIALOG_SUMMARY_PURE_BAD    : LORA_SAMSUM,
        FinetuneDataSet.DIALOG_SUMMARY_HIGH_HARM   : LORA_SAMSUM,
        FinetuneDataSet.SQL_GEN_BENIGN             : LORA_SQLGEN,
        FinetuneDataSet.SQL_GEN_LOW_HARM           : LORA_SQLGEN,
        FinetuneDataSet.SQL_GEN_PURE_BAD           : LORA_SQLGEN,
        FinetuneDataSet.SQL_GEN_HIGH_HARM          : LORA_SQLGEN,
        FinetuneDataSet.AGNEWS_BENIGN              : LORA_AGNEWS,
        FinetuneDataSet.AGNEWS_LOW_HARM            : LORA_AGNEWS,
        FinetuneDataSet.AGNEWS_MEDIUM_HARM         : LORA_AGNEWS,
        FinetuneDataSet.AGNEWS_HIGH_HARM           : LORA_AGNEWS,
        FinetuneDataSet.GSM8K_BENIGN               : LORA_GSM8K,
        FinetuneDataSet.GSM8K_LOW_HARM             : LORA_GSM8K,
        FinetuneDataSet.GSM8K_MEDIUM_HARM          : LORA_GSM8K,
        FinetuneDataSet.GSM8K_HIGH_HARM            : LORA_GSM8K,
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