import os
import json
from tqdm import tqdm
from typing import List, Dict
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safelora import SafeLoRA
from safelora_config import SafeLoRAConfig


class MMLUEvaluator:
    def __init__(self, model_name: str, mmlu_data_path: str, device: str = "cuda", max_new_tokens: int = 2, shot_data_path: str = None, adapter_path: str = None, post_finetune_flag: bool = False):
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.mmlu_data_path = mmlu_data_path
        self.shot_data_path = shot_data_path
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                use_cache=False,
            )
        if adapter_path:
            adapter_tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            new_vocab_size = len(adapter_tokenizer)
            current_vocab_size = self.model.get_input_embeddings().weight.shape[0]
            if new_vocab_size != current_vocab_size:
                print(f"Resizing embeddings: {current_vocab_size} -> {new_vocab_size}")
                self.model.resize_token_embeddings(new_vocab_size)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            if post_finetune_flag:
                print("Applying SafeLoRA...")
                # Initialize configuration example
                config = SafeLoRAConfig(
                    base_model_path="meta-llama/Llama-2-7b-hf",
                    aligned_model_path="meta-llama/Llama-2-7b-chat-hf",
                    select_layers_type="number",
                    threshold=0.5,
                    num_proj_layers=10,
                    devices="cuda"  
                )
                safelora = SafeLoRA(self.model, config)
                self.model = safelora.model
            else:
                self.model = self.model.merge_and_unload()
        self.model = self.model.eval()
        with open(mmlu_data_path) as f:
            self.mmlu_data = json.load(f)
        if shot_data_path:
            with open(shot_data_path) as f:
                self.few_shot_examples = json.load(f)
        else:
            self.few_shot_examples = []

    def format_prompt(self, question: str, choices: List[str], few_shot: bool = False) -> str:
        lettered = ['A', 'B', 'C', 'D']
        option_str = "\n".join([f"{lettered[i]}. {opt}" for i, opt in enumerate(choices)])
        prompt = ""
        if few_shot and self.few_shot_examples:
            for ex in self.few_shot_examples:
                shot_q = self.format_prompt(ex["question"], ex["choices"])
                prompt += f"{shot_q}\nAnswer: {ex['answer']}\n\n"
        prompt += f"Question: {question}\n{option_str}\nAnswer:"
        return prompt

    def extract_answer(self, output: str) -> str:
        for c in ['A', 'B', 'C', 'D']:
            if c in output:
                return c
        return ""

    def run_evaluation(self, output_path: str = "mmlu_eval_results.json", few_shot: bool = False) -> Dict:
        correct = 0
        total = 0
        predictions = {}
        for example in tqdm(self.mmlu_data):
            question = example['question']
            choices = example['choices']
            correct_answer = ['A', 'B', 'C', 'D'][example['answer']]
            prompt = self.format_prompt(question, choices, few_shot=few_shot)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False
            )
            decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            pred = self.extract_answer(decoded[len(prompt):].strip())
            predictions[example['id']] = {
                "question": question,
                "predicted": pred,
                "correct": correct_answer
            }
            if pred == correct_answer:
                correct += 1
            total += 1
        # Compute Accuracy
        acc = correct / total * 100
        results = {
            "accuracy": acc,
            "num_correct": correct,
            "num_total": total,
            "predictions": predictions
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation complete. Accuracy: {acc:.2f}%. Saved to {output_path}")
        return results
