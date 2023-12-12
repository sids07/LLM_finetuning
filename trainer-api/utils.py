from datasets import DatasetDict, load_dataset
from config import SYSTEM_MESSAGE
from transformers import DataCollatorForLanguageModeling
import numpy as np
from functools import partial
from config import data_args, training_args

def generate_dataset(
    train_dataset_path: str,
    test_dataset_path: str
) -> DatasetDict:

    raw_datasets = DatasetDict()
    
    raw_datasets["train"] = load_dataset(
      "csv",
      data_files = train_dataset_path,
      split="train"
    )
    
    raw_datasets["test"] = load_dataset(
      "csv",
      data_files = test_dataset_path,
      split="train"
    )
    
    return raw_datasets


def apply_chat_template(example, tokenizer):
    
    messages = [
            {
                "role":"system",
                "content": SYSTEM_MESSAGE
            },
            {
                "content": "INPUT LIST OF DIAGNOSTIC HYPOTHESES:\n"+example["prompt"],
                "role":"user"
            },
            {
                "content": example["postive_response"] + "\n\n###End",
                "role":"assistant"
            }
            ]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    return example

RESPONSE_KEY = " ### Response:"

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY)
        response_token_ids = response_token_ids[2:5]
        labels = batch["labels"].clone()
        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                if np.array_equal(
                    response_token_ids,
                    batch["labels"][i, idx : idx + len(response_token_ids)],
                ):
                    response_token_ids_start_idx = idx
                    break
            if response_token_ids_start_idx is None:
                raise RuntimeError("Could not find response key token IDs")
            response_token_ids_end_idx = response_token_ids_start_idx + len(
                response_token_ids
            )
            labels[i, :response_token_ids_end_idx] = -100
        batch["labels"] = labels
        return batch

def preprocess_batch(batch, tokenizer, max_length):
    """Preprocess a batch of inputs for the language model."""
    batch["input_ids"] = tokenizer(batch["text"], max_length=max_length, truncation=True).input_ids
    return batch

def preprocess_dataset(dataset, tokenizer, max_length= data_args["TOKENIZER_MODEL_MAX_LENGTH"], seed=training_args["seed"]):
    
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['prompt', 'negative_response', 'postive_response', "text"],
    )

    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    dataset = dataset.shuffle(seed=seed)

    return dataset
