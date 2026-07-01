"""
src/training/trainer.py — QLoRA trainer for Qwen2.5-3B.
Blackwell/RTX5050 compatible: fp16, no bitsandbytes, no gradient checkpointing.
"""
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from src.training.formatter import prepare_dataset
from src.utils import load_config, get_logger


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = True
    return model


def apply_lora(model, config: dict):
    lora_cfg = config["qlora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logging.getLogger(__name__).info(
        f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
    )
    return model


def build_training_args(config: dict) -> SFTConfig:
    t = config["training"]
    Path(t["output_dir"]).mkdir(parents=True, exist_ok=True)
    return SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        save_steps=t["save_steps"],
        logging_steps=t["logging_steps"],
        bf16=False,
        fp16=True,
        gradient_checkpointing=False,
        optim="adamw_torch",
        max_seq_length=config["model"]["max_seq_length"],
        dataset_text_field="text",
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=False,
        eval_strategy="steps",
        eval_steps=50,
    )


def train(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    logger = get_logger("trainer", config)

    model_name        = config["model"]["base_model"]
    instructions_file = "data/instructions/instructions.jsonl"

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)

    logger.info("Preparing dataset...")
    train_ds, val_ds = prepare_dataset(instructions_file, tokenizer)

    logger.info(f"Loading model: {model_name}")
    model = load_model(model_name)

    logger.info("Applying LoRA...")
    model = apply_lora(model, config)

    training_args = build_training_args(config)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving adapter...")
    adapter_path = Path(config["training"]["output_dir"]) / "final_adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info(f"Training complete. Adapter saved to {adapter_path}")
    return trainer
