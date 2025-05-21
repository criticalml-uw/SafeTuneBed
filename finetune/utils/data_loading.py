#    This file contains code mostly obtained from the Lisa repository 
#    https://github.com/git-disl/Lisa/tree/main along with numerous edits. 
#
#    It comes with the following license:
#    
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import io
import copy
import json
import logging

import torch
import transformers

from dataclasses import dataclass
from typing import Dict, Sequence
from torch.utils.data import Dataset


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "direct_use": (
        "{instruction}"
    )
}

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, poison_ratio=None, sample_num=None, benign_dataset=None, guide_data_num= None, finetuning_guide_data_num=None):
        if benign_dataset is not None:
            benign_dataset = jload(benign_dataset)
            if sample_num is None:
                sample_num = 0
                for sample in benign_dataset:
                    sample_num += 1

        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = jload(data_path)
        if "BeaverTails_safe" in data_path:
            from datasets import load_dataset
            list_data_dict =[]
            dataset =load_dataset("PKU-Alignment/BeaverTails")
            index=0
            for example in dataset["30k_train"]:
                if example["is_safe"] and  index< guide_data_num:
                    instance = {}
                    instance["output"] = example["response"]
                    instance["instruction"] = example["prompt"]
                    instance["input"] =""
                    list_data_dict += [instance]
                    index+=1
                    # print(instance["instruction"])
                    # print(instance["output"])
        elif "BeaverTails_dangerous" in data_path or "pure_bad" in data_path:
            from datasets import load_dataset
            list_data_dict =[]
            if "BeaverTails_dangerous" in data_path:
                dataset =load_dataset("PKU-Alignment/BeaverTails")
                index=0
                poison_num = int(poison_ratio*sample_num)
                if finetuning_guide_data_num!=None:
                    normal_num = int((1-poison_ratio)*sample_num-finetuning_guide_data_num)
                else:
                    normal_num = int((1-poison_ratio)*sample_num)
                for example in dataset["30k_train"]:
                    if not example["is_safe"] and index<poison_num:
                        instance = {}
                        instance["output"] = example["response"]
                        instance["instruction"] = example["prompt"]
                        instance["input"] =""
                        list_data_dict += [instance]
                        index+=1
            else:
                normal_num = sample_num

            index=0
            for sample in benign_dataset:
                if  index<normal_num:
                    list_data_dict += [sample]
                    index+=1

            index=0
            if finetuning_guide_data_num!=None and finetuning_guide_data_num != 0:
                for example in dataset["30k_train"]:
                    if example["is_safe"] and index<finetuning_guide_data_num:
                        instance = {}
                        instance["output"] = example["response"]
                        instance["instruction"] = example["prompt"]
                        instance["input"] =""
                        list_data_dict += [instance]
                        index+=1
                
        else:
            list_data_dict = jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        sources = list()
        targets = list()
        if "pure_bad" in data_path:
            pure_bad_data_dict = jload(data_path)
            sources += [example['instruction'] for example in pure_bad_data_dict]
            targets += [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        if "BeaverTails" in data_path or benign_dataset is not None:
            sources += [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in list_data_dict
            ]
            targets += [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # print(i)
        # print(len(self.input_ids))
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, poison_ratio=data_args.poison_ratio,sample_num=data_args.sample_num, benign_dataset=data_args.benign_dataset, guide_data_num=10000)
    if "BeaverTails_safe" not in data_args.data_path:
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path="BeaverTails_safe",guide_data_num=10000)
    else:
        eval_dataset = None 
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
