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
from utils.config_helpers import MethodConfig


BASE_VACCINE_LORA_CFG = MethodConfig(
    lora_folder="ckpts/vaccine/checkpoint-20000",
    huggingface_model="meta-llama/Llama-2-7b-chat-hf",
)

# config for lora on sst2 datasets
VACCINE_LORA_SST2 = replace(
    BASE_SST2,
    method_config=BASE_VACCINE_LORA_CFG
)

# config for lora on gsm8k datasets
VACCINE_LORA_GSM8K = replace(
    BASE_GSM8K,
    method_config=BASE_VACCINE_LORA_CFG
)

# config for lora on dolly datasets
VACCINE_LORA_DOLLY = replace(
    BASE_DOLLY,
    method_config=BASE_VACCINE_LORA_CFG
)

# config for lora on agnews datasets
VACCINE_LORA_AGNEWS = replace(
    BASE_AGNEWS,
    method_config=BASE_VACCINE_LORA_CFG
)

# config for lora on alpaca datasets
VACCINE_LORA_ALPACA = replace(
    BASE_ALPACA,
    method_config=BASE_VACCINE_LORA_CFG
)

# config for lora on samsum datasets
VACCINE_LORA_SAMSUM = replace(
    BASE_SAMSUM,
    method_config=BASE_VACCINE_LORA_CFG
)

# config for lora on sqlgen datasets
VACCINE_LORA_SQLGEN = replace(
    BASE_SQLGEN,
    method_config=BASE_VACCINE_LORA_CFG
)