from enum import Enum
from dataclasses import dataclass
from typing import Dict
from typing import Optional

from utils.data_loading import (
    SupervisedDataset,
    smart_tokenizer_and_embedding_resize,
    DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN,
    DEFAULT_PAD_TOKEN, DEFAULT_UNK_TOKEN
)

from enum import Enum


class DataPath(Enum):
    ALPACA               = "data/alpaca.json"
    SST2                 = "data/sst2.json"
    GSM8K                = "data/gsm8k.json"
    AGNEWS               = "data/agnews.json"
    BEAVERTAILS_HARMFUL  = "BeaverTails_dangerous"
    BEAVERTAILS_SAFE     = "BeaverTails_safe"
    DOLLY                = "data/dolly.json"
    DIALOG_SUMMARY       = "data/samsum_train.json"
    SQL_GEN              = "data/sqlgen_train.json"
    PURE_BAD             = "data/pure_bad.json"


@dataclass(frozen=True)
class FinetuneDatasetConfig:
    data_path: DataPath
    sample_num: Optional[int] = None
    poison_ratio: Optional[float] = 0
    finetuning_guide_data_num: int = 0
    harmful_dataset: Optional[DataPath] = None
    guide_dataset: Optional[DataPath] = None


class AlignmentDataset(Enum):
    BEAVERTAILS = "beavertails_safe"


class HarmfulDataset(Enum):
    BEAVERTAILS = "beavertails_harmful"
    PURE_BAD = "pure_bad"


class FinetuneDataSet(Enum):
    ALPACA_BENIGN              = "alpaca_data_no_safety"
    ALPACA_LOW_HARM            = "alpaca_low_harm"
    ALPACA_MEDIUM_HARM         = "alpaca_medium_harm"
    ALPACA_HIGH_HARM           = "alpaca_high_harm"

    SST2_BENIGN                = "sst2"
    SST2_LOW_HARM              = "sst2_low_harm"
    SST2_MEDIUM_HARM           = "sst2_medium_harm"
    SST2_HIGH_HARM             = "sst2_high_harm"

    DOLLY_BENIGN               = "dolly_dataset"
    DOLLY_LOW_HARM             = "dolly_low_harm"
    DOLLY_PURE_BAD             = "dolly_pure_bad"
    DOLLY_HIGH_HARM            = "dolly_high_harm"

    DIALOG_SUMMARY_BENIGN      = "dialog_summary"
    DIALOG_SUMMARY_LOW_HARM    = "dialog_summary_low_harm"
    DIALOG_SUMMARY_PURE_BAD    = "dialog_summary_pure_bad"
    DIALOG_SUMMARY_HIGH_HARM   = "dialog_summary_high_harm"

    SQL_GEN_BENIGN             = "sql_generation"
    SQL_GEN_LOW_HARM           = "sql_generation_low_harm"
    SQL_GEN_PURE_BAD           = "sql_generation_pure_bad"
    SQL_GEN_HIGH_HARM          = "sql_generation_high_harm"

    AGNEWS_BENIGN              = "ag_news"
    AGNEWS_LOW_HARM            = "ag_news_low_harm"
    AGNEWS_MEDIUM_HARM         = "ag_news_medium_harm"
    AGNEWS_HIGH_HARM           = "ag_news_high_harm"

    GSM8K_BENIGN               = "gsm8k"
    GSM8K_LOW_HARM             = "gsm8k_low_harm"
    GSM8K_MEDIUM_HARM          = "gsm8k_medium_harm"
    GSM8K_HIGH_HARM            = "gsm8k_high_harm"

    SST2_TEST                  = "sst2_test"


DATASETS: Dict[FinetuneDataSet, FinetuneDatasetConfig] = {
    # Alpaca
    FinetuneDataSet.ALPACA_BENIGN:                FinetuneDatasetConfig(data_path=DataPath.ALPACA),
    FinetuneDataSet.ALPACA_LOW_HARM:              FinetuneDatasetConfig(data_path=DataPath.ALPACA,
                                                                poison_ratio=0.05,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),
    FinetuneDataSet.ALPACA_HIGH_HARM:             FinetuneDatasetConfig(data_path=DataPath.ALPACA,
                                                                poison_ratio=0.3,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),

    # Dolly
    FinetuneDataSet.DOLLY_BENIGN:                 FinetuneDatasetConfig(data_path=DataPath.DOLLY),
    FinetuneDataSet.DOLLY_LOW_HARM:               FinetuneDatasetConfig(data_path=DataPath.DOLLY,
                                                                poison_ratio=0.05,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),
    FinetuneDataSet.DOLLY_HIGH_HARM:              FinetuneDatasetConfig(data_path=DataPath.DOLLY,
                                                                poison_ratio=0.3,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),

    # Dialog Summary
    FinetuneDataSet.DIALOG_SUMMARY_BENIGN:        FinetuneDatasetConfig(data_path=DataPath.DIALOG_SUMMARY),
    FinetuneDataSet.DIALOG_SUMMARY_PURE_BAD:      FinetuneDatasetConfig(data_path=DataPath.DIALOG_SUMMARY,
                                                                harmful_dataset=DataPath.PURE_BAD),
    FinetuneDataSet.DIALOG_SUMMARY_LOW_HARM:      FinetuneDatasetConfig(data_path=DataPath.DIALOG_SUMMARY,
                                                                sample_num=1000,
                                                                poison_ratio=0.05,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),
    FinetuneDataSet.DIALOG_SUMMARY_HIGH_HARM:     FinetuneDatasetConfig(data_path=DataPath.DIALOG_SUMMARY,
                                                                sample_num=1000,
                                                                poison_ratio=0.30,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),

    # SQL Generation
    FinetuneDataSet.SQL_GEN_BENIGN:               FinetuneDatasetConfig(data_path=DataPath.SQL_GEN),
    FinetuneDataSet.SQL_GEN_PURE_BAD:             FinetuneDatasetConfig(data_path=DataPath.SQL_GEN,
                                                                harmful_dataset=DataPath.PURE_BAD),
    FinetuneDataSet.SQL_GEN_LOW_HARM:             FinetuneDatasetConfig(data_path=DataPath.SQL_GEN,
                                                                sample_num=1000,
                                                                poison_ratio=0.05,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),
    FinetuneDataSet.SQL_GEN_HIGH_HARM:            FinetuneDatasetConfig(data_path=DataPath.SQL_GEN,
                                                                sample_num=1000,
                                                                poison_ratio=0.30,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),

    # SST2
    FinetuneDataSet.SST2_BENIGN:                  FinetuneDatasetConfig(data_path=DataPath.SST2,
                                                                sample_num=5000),
    FinetuneDataSet.SST2_LOW_HARM:                FinetuneDatasetConfig(data_path=DataPath.SST2,
                                                                sample_num=5000,
                                                                poison_ratio=0.05,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),
    FinetuneDataSet.SST2_HIGH_HARM:               FinetuneDatasetConfig(data_path=DataPath.SST2,
                                                                sample_num=5000,
                                                                poison_ratio=0.30,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),

    # AGNews
    FinetuneDataSet.AGNEWS_BENIGN:                FinetuneDatasetConfig(data_path=DataPath.AGNEWS,
                                                                sample_num=5000),
    FinetuneDataSet.AGNEWS_LOW_HARM:              FinetuneDatasetConfig(data_path=DataPath.AGNEWS,
                                                                sample_num=5000,
                                                                poison_ratio=0.05,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),
    FinetuneDataSet.AGNEWS_HIGH_HARM:             FinetuneDatasetConfig(data_path=DataPath.AGNEWS,
                                                                sample_num=5000,
                                                                poison_ratio=0.30,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),

    # GSM8K
    FinetuneDataSet.GSM8K_BENIGN:                 FinetuneDatasetConfig(data_path=DataPath.GSM8K,
                                                                sample_num=5000),
    FinetuneDataSet.GSM8K_LOW_HARM:               FinetuneDatasetConfig(data_path=DataPath.GSM8K,
                                                                sample_num=5000,
                                                                poison_ratio=0.05,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),
    FinetuneDataSet.GSM8K_HIGH_HARM:              FinetuneDatasetConfig(data_path=DataPath.GSM8K,
                                                                sample_num=5000,
                                                                poison_ratio=0.30,
                                                                harmful_dataset=DataPath.BEAVERTAILS_HARMFUL),
}



def get_dataset(
    model,
    tokenizer,
    dataset: FinetuneDataSet
):
    cfg = DATASETS[dataset]

    special_tokens = dict()
    if tokenizer.pad_token is None: special_tokens["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None: special_tokens["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None: special_tokens["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None: special_tokens["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens,
        tokenizer=tokenizer,
        model=model,
    )

    return SupervisedDataset(
        tokenizer=tokenizer,
        data_path=cfg.harmful_dataset.value if cfg.harmful_dataset is not None else "BeaverTails_dangerous",
        poison_ratio=cfg.poison_ratio,
        sample_num=cfg.sample_num,
        benign_dataset=cfg.data_path.value,
        finetuning_guide_data_num=cfg.finetuning_guide_data_num,
    )
