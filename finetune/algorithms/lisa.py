from typing import Dict
from typing import Optional

from transformers import Trainer

from utils.methods import MethodAlgorithm
from finetune_datasets import FinetuneDataSet
from utils.config_helpers import FinetuneExperimentConfig
from utils.data_loading import SupervisedDataset
from utils.data_loading import DataCollatorForSupervisedDataset

from configs.algorithms.lisa import LISA_SST2
from configs.algorithms.lisa import LISA_DOLLY
from configs.algorithms.lisa import LISA_GSM8K
from configs.algorithms.lisa import LISA_ALPACA
from configs.algorithms.lisa import LISA_SAMSUM
from configs.algorithms.lisa import LISA_SQLGEN
from configs.algorithms.lisa import LISA_AGNEWS

from trainers.lisa import LisaTrainer


class LisaAlgorithm(MethodAlgorithm):
    """The Lisa Finetuning Defense"""
    
    method_name = "lisa"

    registry: Dict[FinetuneDataSet, FinetuneExperimentConfig] = {
        FinetuneDataSet.ALPACA_BENIGN              : LISA_ALPACA,
        FinetuneDataSet.ALPACA_LOW_HARM            : LISA_ALPACA,
        FinetuneDataSet.ALPACA_MEDIUM_HARM         : LISA_ALPACA,
        FinetuneDataSet.ALPACA_HIGH_HARM           : LISA_ALPACA,
        FinetuneDataSet.SST2_BENIGN                : LISA_SST2,
        FinetuneDataSet.SST2_LOW_HARM              : LISA_SST2,
        FinetuneDataSet.SST2_MEDIUM_HARM           : LISA_SST2,
        FinetuneDataSet.SST2_HIGH_HARM             : LISA_SST2,
        FinetuneDataSet.DOLLY_BENIGN               : LISA_DOLLY,
        FinetuneDataSet.DOLLY_LOW_HARM             : LISA_DOLLY,
        FinetuneDataSet.DOLLY_PURE_BAD             : LISA_DOLLY,
        FinetuneDataSet.DOLLY_HIGH_HARM            : LISA_DOLLY,
        FinetuneDataSet.DIALOG_SUMMARY_BENIGN      : LISA_SAMSUM,
        FinetuneDataSet.DIALOG_SUMMARY_LOW_HARM    : LISA_SAMSUM,
        FinetuneDataSet.DIALOG_SUMMARY_PURE_BAD    : LISA_SAMSUM,
        FinetuneDataSet.DIALOG_SUMMARY_HIGH_HARM   : LISA_SAMSUM,
        FinetuneDataSet.SQL_GEN_BENIGN             : LISA_SQLGEN,
        FinetuneDataSet.SQL_GEN_LOW_HARM           : LISA_SQLGEN,
        FinetuneDataSet.SQL_GEN_PURE_BAD           : LISA_SQLGEN,
        FinetuneDataSet.SQL_GEN_HIGH_HARM          : LISA_SQLGEN,
        FinetuneDataSet.AGNEWS_BENIGN              : LISA_AGNEWS,
        FinetuneDataSet.AGNEWS_LOW_HARM            : LISA_AGNEWS,
        FinetuneDataSet.AGNEWS_MEDIUM_HARM         : LISA_AGNEWS,
        FinetuneDataSet.AGNEWS_HIGH_HARM           : LISA_AGNEWS,
        FinetuneDataSet.GSM8K_BENIGN               : LISA_GSM8K,
        FinetuneDataSet.GSM8K_LOW_HARM             : LISA_GSM8K,
        FinetuneDataSet.GSM8K_MEDIUM_HARM          : LISA_GSM8K,
        FinetuneDataSet.GSM8K_HIGH_HARM            : LISA_GSM8K,
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

        self.experiment_config.train_args.alignment_step=self.experiment_config.method_config.alignment_step   
        self.experiment_config.train_args.finetune_step=self.experiment_config.method_config.finetune_step
        self.experiment_config.train_args.guide_data_num=self.experiment_config.method_config.guide_data_num
        self.experiment_config.train_args.rho=self.experiment_config.method_config.rho

        alignment_dataset = SupervisedDataset(
            tokenizer=self.tokenizer,
            data_path="BeaverTails_safe",
            guide_data_num=self.experiment_config.method_config.guide_data_num,
        )

        trainer = LisaTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.experiment_config.train_args,
            train_dataset=self.data,
            data_collator=DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        )

        trainer.init(alignment_dataset)
        trainer.train()