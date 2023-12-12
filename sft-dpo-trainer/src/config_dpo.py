model_args = {
    "MODEL_NAME":"hiiamsid/mistral-7b-sft",
    "USE_PEFT":False,
    "LORA_R":16,
    "LORA_ALPHA":32,
    "LORA_DROPOUT": 0.05,
    "LORA_TARGET_MODULES": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    "LORA_MODULES_TO_SAVE": None,
    "USE_FLASH_ATTENTION_2": True,
    "load_in_8bit":False,
    "load_in_4bit":False,
    "bnb_4bit_quant_type":"nf4",
    "use_bnb_nested_quant":False
}

data_args = {
    "TRUNCATION_SIDE":"left",
    "TOKENIZER_MODEL_MAX_LENGTH":8192,
    "TRAIN_FILE_PATH":"train_df.csv",
    "TEST_FILE_PATH":"test_df.csv",
    "PREPROCESSING_NUM_WORKERS": 4
}

training_args = {
    "seed": 42,
    "gradient_checkpointing": True,
    "beta":0.1,
    "logging_first_step":True,
    "max_prompt_length":7650,
    "max_length":8192,
    "optim":"rmsprop",
    "remove_unused_columns":False,
    "bf16": True,
    "fp16":False,
    "do_eval":True,
    "evaluation_strategy":"epoch",
    "gradient_accumulation_steps":1,
    "hub_model_id":"mistral-7b-dpo-2",
    "learning_rate":5.0e-7,
    "logging_steps":5,
    "lr_scheduler_type":"linear",
    "num_train_epochs":3,
    "output_dir":"model/dpo-full",
    "per_device_train_batch_size":4,
    "per_device_eval_batch_size":4,
    "push_to_hub":True,
    "save_strategy":"no",
    "save_total_limit":None,
    "warmup_ratio":0.1,
    "log_level":"info"
}

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

SYSTEM_MESSAGE = """
INSTRUCTIONS:

You will be given a list of diagnostic hypotheses from a doctor. Your job is to review and organize them into a structure that can later be turned into an in-depth report. The structure should adhere to the following rules:

1. There must always be a trend for sleep, diet, exercise and mental health. Any health issues related to these default trends should be included in them. This true even if none of the diagnostic hypotheses fit into these default trends.

2.  Trends that are very tightly related should be merged into a single trend. Everyone gets a Sleep trend, but if that patient also has sleep apnea, the trend would instead be a single Trend called "Sleep And Sleep Apnea." A similar case would be for the Diet trend. If someone is overweight, instead of having a separate BMI trend, the unified Default Trend would be "Diet and BMI." Likewise again, take the Default Trend of exercise, which everyone gets, if someone has long-standing injuries from exercise, this would be merged into the default Exercise trend to be "Exercise and a History of Exercise Injuries." Anything directly related to Mental Health, such as depression, anxiety, stress, relationship issues, self-cutting, etc would all be under a single Default Trend called Mental Health regardless if it was preexisting or emerging.

3. However just because two trends may seem superficially related doesn't mean they are. There must be a link that makes sense given the age, sex, body composition and medical history of the patient. Someone may have a history of Alzheimer's and be experiencing memory issues, but if they're only 20 years old, their memory loss is incredibly unlikely to be related to Alzheimer's. In this scenarios memory loss should be it's only separate trend. Likewise chest pain doesn't necessarily mean heart issues, even if there is a family history of heart issues. Chest pain can be respiratory, and if the patient has respiratory issue this must be balanced in. The appropriate response to this kind of ambiguity is to separate out ambiguous symptoms into their own trends. There must be a clear and logical link to blend multiple list elements under the same trend.

4. Not all diagnostic hypothesis is high quality enough to make it to the report. There are no apparent symptoms or strong family history, the diagnostic hypothesis is likely too speculative.

5. If a trend reflects an official diagnosis, then the trend name can reflect a diagnosis. If there isn't a diagnosis yet the trend name should reflect the symptoms reported, not a speculative diagnosis.
"""