from config_sft import model_args, data_args, training_args
from data import apply_chat_template, generate_dataset
from models import get_tokenizer, get_peft_config, get_quantization_config
from trl import SFTTrainer
from transformers import set_seed, TrainingArguments
from accelerate import Accelerator
import torch

def train(model_args, data_args, training_args):
    set_seed(training_args["seed"])

    accelerator = Accelerator()
    
    raw_datasets = generate_dataset(
        data_args["TRAIN_FILE_PATH"],
        data_args["TEST_FILE_PATH"]
    )
    
    tokenizer = get_tokenizer(model_args, data_args)

    raw_datasets = raw_datasets.map(
        apply_chat_template, 
        fn_kwargs={"tokenizer": tokenizer, "task": "sft"}
    )

    model_kwargs = dict(
          use_flash_attention_2=model_args["USE_FLASH_ATTENTION_2"],
          use_cache=False if training_args["gradient_checkpointing"] else True,
          quantization_config=get_quantization_config(model_args)
      )

    peft_config = get_peft_config(model_args)

    training_args_hf = TrainingArguments(
        output_dir = training_args["output_dir"],
        seed = training_args["seed"],
        do_eval = training_args["do_eval"],
        evaluation_strategy = training_args["evaluation_strategy"],
        per_device_train_batch_size = training_args["per_device_train_batch_size"],
        per_device_eval_batch_size = training_args["per_device_eval_batch_size"],
        gradient_accumulation_steps = training_args["gradient_accumulation_steps"],
        learning_rate = training_args["learning_rate"],
        num_train_epochs = training_args["num_train_epochs"],
        lr_scheduler_type = training_args["lr_scheduler_type"],
        warmup_ratio = training_args["warmup_ratio"],
        log_level = training_args["log_level"],
        logging_steps = training_args["logging_steps"],
        save_strategy = training_args["save_strategy"],
        save_total_limit = training_args["save_total_limit"],
        bf16 = training_args["bf16"],
        fp16 = training_args["fp16"],
        remove_unused_columns = training_args["remove_unused_columns"],
        push_to_hub = training_args["push_to_hub"],
        hub_model_id = training_args["hub_model_id"],
        gradient_checkpointing = training_args["gradient_checkpointing"],
        overwrite_output_dir = training_args["overwrite_output_dir"],
        tf32 = training_args["tf32"]
    )
    
    trainer = SFTTrainer(
            model=model_args["MODEL_NAME"],
            model_init_kwargs=model_kwargs,
            args=training_args_hf,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            dataset_text_field="text",
            max_seq_length=training_args["max_seq_length"],
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
    print("Training")
    
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


    if training_args["do_eval"]:
        metrics = trainer.evaluate()
    
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_model(training_args["output_dir"])

    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args["MODEL_NAME"]
        }
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args["output_dir"])

        if training_args["push_to_hub"] is True:
            print("Pushing to hub...")
            trainer.push_to_hub()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    train(model_args, data_args, training_args)    