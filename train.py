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

import logging
import datasets
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import os
import json

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
        "### Instruction:\n{input}\n\n### Response:"
    ),
    "prompt_qa": (
        "Answer the given question precisely.\n"
        "Question: {input}\nAnswer:"
    ),
    "prompt_dialogue": (
        "The following is a conversation between a human and an AI assistant. "
        "The AI assistant gives helpful and polite answers to the user's questions.\n"
        "{input}\n\n[|AI|]:"
    ),
    "no_prompt":(
        "{input}"
    )
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    print(f"add special tokens: {special_tokens_dict}")
    print(f"From {len(tokenizer)}")
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"To {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    # examples = [s + t for s, t in zip(sources, targets)]
    # examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_id_list, label_list = [], []
    for source, target in zip(sources, targets):
        text = [source, target]
        inputs = tokenizer(
            text=text, max_length=tokenizer.model_max_length, truncation=True
        )
        input_ids, labels = [], []
        for i, iids in enumerate(inputs["input_ids"]):
            # if i != 0:
                # iids = iids[1:]
            input_ids.extend(iids)
            if i % 2 == 0:
                labels.extend([IGNORE_INDEX] * len(iids))
            else:
                labels.extend(iids)
        input_ids = torch.tensor(input_ids, dtype=torch.long)[
            : tokenizer.model_max_length
        ]
        labels = torch.tensor(labels, dtype=torch.long)[: tokenizer.model_max_length]
        input_id_list.append(input_ids)
        label_list.append(labels)
    return dict(input_ids=input_id_list, labels=label_list)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        logging.warning("Loading data...")
        if os.path.isdir(data_path):
            list_data_dict = datasets.load_from_disk(data_path)
            if 'train' in list_data_dict:
                list_data_dict = list_data_dict['train']
        elif data_path.endswith('.json'):
            with open(data_path) as f:
                list_data_dict = json.load(f)
        elif data_path.endswith('.jsonl'):
            with open(data_path) as f:
                all_data = f.readlines()
                list_data_dict = [json.loads(l) for l in all_data]

        logging.warning("Formatting inputs...")
        prompt = PROMPT_DICT["prompt_no_input"]
        sources = []
        targets = []
        for example in list_data_dict:
            sources.append(prompt.replace('{input}', example['instruction']))
            targets.append(f"{example['output']}{tokenizer.eos_token}")
        print(f"Load {len(sources)} wiki examples!")
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        self.print_example(self.input_ids[0].tolist(), self.labels[0].tolist())

    def print_example(self, input_ids, labels):
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        label_tokens = self.tokenizer.convert_ids_to_tokens([l if l != -100 else 0 for l in labels])
        print(f"Input: [{input_tokens}]")
        print(f"Input str: [{self.tokenizer.convert_tokens_to_string(input_tokens)}]")
        print(f"Output: [{self.tokenizer.convert_tokens_to_string(input_tokens)}]")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # model_max_length=training_args.model_max_length,
        use_fast=False,
        trust_remote_code=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()