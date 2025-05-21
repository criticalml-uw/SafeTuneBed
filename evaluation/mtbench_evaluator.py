import os
import json
from tqdm import tqdm
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from peft import PeftModel
import re
from safelora import SafeLoRA
from safelora_config import SafeLoRAConfig



class MTBenchEvaluator:
    def __init__(
        self,
        model_name: str,
        openai_api_key: str,
        mt_bench_path: str,
        adapter_path: str = None,
        post_finetune_flag: bool = False,
        device: str = "cuda",
        max_new_tokens: int = 512
    ):
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.mt_bench_path = mt_bench_path
        self.client = OpenAI(api_key=openai_api_key)
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            use_cache=False
        )
        if adapter_path:
            adapter_tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            new_vocab_size = len(adapter_tokenizer)
            current_vocab_size = model.get_input_embeddings().weight.shape[0]
            if new_vocab_size != current_vocab_size:
                print(f"Resizing embeddings: {current_vocab_size} -> {new_vocab_size}")
                model.resize_token_embeddings(new_vocab_size)
            model = PeftModel.from_pretrained(model, adapter_path)
            if post_finetune_flag:
                print("Applying SafeLoRA...")
                # Only Support SafeLoRA for now
                config = SafeLoRAConfig(
                    base_model_path="meta-llama/Llama-2-7b-hf",
                    aligned_model_path="meta-llama/Llama-2-7b-chat-hf",
                    select_layers_type="number",
                    threshold=0.5,
                    num_proj_layers=10,
                    devices="cuda"  
                )
                safelora = SafeLoRA(model, config)
                model = safelora.model
            else:
                model = model.merge_and_unload()
        self.model = model.eval()
        with open(mt_bench_path) as f:
            self.mt_bench_data = json.load(f)

    def generate_responses(self) -> Dict[str, List[Dict[str, str]]]:
        print("Generating model responses...")
        results = {}
        for q in tqdm(self.mt_bench_data):
            qid = q['id']
            messages = [{"role": "system", "content": q['system']}]
            for user_input in q['turns']:
                messages.append({"role": "user", "content": user_input})
                prompt = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
                output = self.model.generate(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)
                decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
                llm_response = decoded.split(user_input)[-1].strip()
                messages.append({"role": "assistant", "content": llm_response})
            results[qid] = messages
        return results

    def extract_score(self, text: str) -> int:
        match = re.search(r"(\b\d{1,2})\s*/\s*10", text)
        return int(match.group(1)) if match else None

    def evaluate_response(self, conversation: List[Dict[str, str]]) -> str:
        eval_prompt = [
            {"role": "system", "content": "You are an expert judge evaluating an assistant’s response in a multi-turn conversation."},
            *conversation,
            {"role": "user", "content": "Please rate the assistant's helpfulness, correctness, and coherence on a scale of 1 to 10 and explain briefly."}
        ]
        while True:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=eval_prompt,
                    temperature=0.3,
                )
                content = response.choices[0].message.content.strip()
                score = self.extract_score(content)
                if score is None:
                    print("No score found:")
                    print(content)
                else:
                    print(f"GPT4o-mini score: {score}")
                return {
                    "score": score,
                    "explanation": content
                }
            except Exception as e:
                print(f"Retrying GPT4o-mini call due to error:", e)
                time.sleep(10)

    def run_evaluation(self, output_path: str = "mt_bench_eval_results.json"):
        generated = self.generate_responses()
        print("Starting evaluation with GPT-4o-mini...")
        scores = {}
        numeric_scores = []
        for qid, conv in tqdm(generated.items()):
            score = self.evaluate_response(conv)
            scores[qid] = score
            if score["score"] is not None:
                numeric_scores.append(score["score"])
        avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
        results = {
            "average_score": avg_score,
            "responses": scores
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation complete. Results saved to {output_path}")
        return results

