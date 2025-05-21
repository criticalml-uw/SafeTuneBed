from peft import LoraConfig
from transformers import TrainingArguments

from utils.config_helpers import FinetuneExperimentConfig
from utils.config_helpers import LisaConfig


VACCINE_ALLIGNMENT_CFG = FinetuneExperimentConfig(
    root_output_dir="ckpts",
    lora_config=LoraConfig(
        # r=500,
        r=8,
        lora_alpha=4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
    train_args=TrainingArguments(
        output_dir="ckpts/vaccine/",
        per_device_train_batch_size = 5,
        save_steps = 10000,
        logging_steps = 1, 
        learning_rate = 1e-3,
        num_train_epochs = 50,
        lr_scheduler_type = "cosine",
        gradient_checkpointing = True,
        optim = "adamw_torch",
        gradient_accumulation_steps=1,
        weight_decay=0.1,
        warmup_ratio=0.1,
    ),
    method_config=LisaConfig(
        huggingface_model="meta-llama/Llama-2-7b-chat-hf",
        lora_folder=None,
        alignment_step=500,   
        finetune_step=500,
        guide_data_num=2000,
        rho=0.1
    )
)
