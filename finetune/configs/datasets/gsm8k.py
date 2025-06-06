from peft import LoraConfig
from transformers import TrainingArguments

from utils.config_helpers import MethodConfig
from utils.config_helpers import FinetuneExperimentConfig


BASE_GSM8K = FinetuneExperimentConfig(
    root_output_dir="ckpts",
    lora_config=LoraConfig(
        r=8,
        lora_alpha=4,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
    train_args=TrainingArguments(
        per_device_train_batch_size=5,
        save_steps=10000,
        logging_steps=1, 
        learning_rate=1e-5,
        num_train_epochs=50,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        optim="adamw_torch",
        gradient_accumulation_steps=1,
        weight_decay=0.1,
        warmup_ratio=0.1,
    ),
    method_config=MethodConfig(
        huggingface_model="meta-llama/Llama-2-7b-chat-hf",
        lora_folder=None
    )
)