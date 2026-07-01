"""
src/training/formatter.py
Converts instruction JSONL into Qwen2.5 chat-formatted strings
ready for SFTTrainer.
"""

import json
import logging
from pathlib import Path
from datasets import Dataset

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = """You are an expert AWS cloud assistant. Answer questions accurately 
based on official AWS documentation. If a question is outside AWS topics, politely 
decline and explain you only assist with AWS-related questions."""


def load_instructions(instructions_file: str) -> list[dict]:
    """Load instruction pairs from JSONL file."""
    pairs = []
    with open(instructions_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    logger.info(f"Loaded {len(pairs)} instruction pairs")
    return pairs


def format_as_chat(pair: dict, tokenizer) -> str:
    """
    Format a Q&A pair into Qwen2.5 chat template format.
    Uses the tokenizer's built-in apply_chat_template for correctness.
    """
    user_content = pair["instruction"]
    if pair.get("input", "").strip():
        user_content += f"\n\nContext: {pair['input']}"

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user",   "content": user_content},
        {"role": "assistant", "content": pair["response"]},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def prepare_dataset(
    instructions_file: str,
    tokenizer,
    val_split: float = 0.1,
) -> tuple[Dataset, Dataset]:
    """
    Load, format, and split dataset into train/val.
    Returns (train_dataset, val_dataset).
    """
    pairs = load_instructions(instructions_file)

    formatted = []
    for pair in pairs:
        try:
            text = format_as_chat(pair, tokenizer)
            formatted.append({"text": text})
        except Exception as e:
            logger.warning(f"Skipping pair due to format error: {e}")

    logger.info(f"Formatted {len(formatted)} examples")

    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=val_split, seed=42)

    logger.info(f"Train: {len(split['train'])} | Val: {len(split['test'])}")
    return split["train"], split["test"]