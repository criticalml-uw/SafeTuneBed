import os
from dataclasses import replace

from configs.datasets.sst2 import BASE_SST2
from configs.datasets.gsm8k import BASE_GSM8K
from configs.datasets.dolly import BASE_DOLLY
from configs.datasets.agnews import BASE_AGNEWS
from configs.datasets.alpaca import BASE_ALPACA
from configs.datasets.samsum import BASE_SAMSUM
from configs.datasets.sqlgen import BASE_SQLGEN

from utils.config_helpers import FinetuneExperimentConfig
from utils.config_helpers import LisaConfig


# general lisa method config
BASE_LISA_CFG = LisaConfig(
    huggingface_model="meta-llama/Llama-2-7b-chat-hf",
    lora_folder=None,
    alignment_step=100,   
    finetune_step=900 
)

# config for LISA on sst2 datasets
LISA_SST2 = replace(
    BASE_SST2,
    method_config=BASE_LISA_CFG
)

# config for LISA on gsm8k datasets
LISA_GSM8K = replace(
    BASE_GSM8K,
    method_config=BASE_LISA_CFG  
)

# config for LISA on dolly datasets
LISA_DOLLY = replace(
    BASE_DOLLY,
    method_config=replace(
        BASE_LISA_CFG,
        alignment_step=45,   
        finetune_step=405 
    )
)

# config for LISA on agnews datasets
LISA_AGNEWS = replace(
    BASE_AGNEWS,
    method_config=BASE_LISA_CFG  
)

# config for LISA on alpaca datasets
LISA_ALPACA = replace(
    BASE_ALPACA,
    method_config=replace(
        BASE_LISA_CFG,
        alignment_step=75,
        finetune_step=675 
    )
)

# config for LISA on samsum datasets
LISA_SAMSUM = replace(
    BASE_SAMSUM,
    method_config=replace(
        BASE_LISA_CFG,
        alignment_step=10,
        finetune_step=90 
    )
)

# config for LISA on sqlgen datasets
LISA_SQLGEN = replace(
    BASE_SQLGEN,
    method_config=replace(
        BASE_LISA_CFG,
        alignment_step=10,
        finetune_step=90 
    )
)