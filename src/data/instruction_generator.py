"""
src/data/instruction_generator.py
Generates instruction-tuning Q&A pairs from AWS doc chunks
using local Ollama API.
"""

import json
import time
import logging
import requests
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

QUESTION_TYPES = ["factual", "procedural", "troubleshooting", "comparative"]

def make_prompt(chunk_text: str, service: str, question_type: str) -> str:
    type_instructions = {
        "factual": "Ask about a specific fact, definition, limit, or feature described in the text.",
        "procedural": "Ask how to perform a specific task or configuration described in the text.",
        "troubleshooting": "Ask about a problem, limitation, or best practice mentioned in the text.",
        "comparative": "Ask about a difference, trade-off, or comparison mentioned in the text.",
    }

    return f"""You are an expert AWS solutions architect creating training data for an AI assistant.
Based on this AWS {service.upper()} documentation excerpt, generate one {question_type} Q&A pair.

DOCUMENTATION:
{chunk_text[:800]}

INSTRUCTION TYPE: {type_instructions[question_type]}

Respond with ONLY this JSON structure, no explanation, no markdown:
{{
    "instruction": "the question a user would ask",
    "input": "",
    "response": "accurate, detailed answer based strictly on the documentation"
}}"""


def generate_qa_pair(
    chunk: dict,
    question_type: str,
    max_retries: int = 3,
) -> dict | None:
    """Call Ollama API to generate one Q&A pair from a chunk."""
    prompt = make_prompt(chunk["chunk_text"], chunk["service"], question_type)

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 600,
                    }
                },
                timeout=120,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()

            # Strip markdown fences if present
            if "```" in raw:
                parts = raw.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        raw = part
                        break

            # Extract JSON object if extra text present
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                logger.warning(f"No JSON found in response: {raw[:100]}")
                continue
            raw = raw[start:end]

            qa = json.loads(raw)

            # Validate required fields
            if not all(k in qa for k in ["instruction", "input", "response"]):
                logger.warning(f"Missing fields: {list(qa.keys())}")
                continue

            if len(qa["instruction"]) < 10 or len(qa["response"]) < 20:
                logger.warning("Response too short, skipping")
                continue

            return {
                "instruction": qa["instruction"].strip(),
                "input": qa.get("input", "").strip(),
                "response": qa["response"].strip(),
                "metadata": {
                    "source_chunk_id": chunk["chunk_id"],
                    "service": chunk["service"],
                    "url": chunk["url"],
                    "question_type": question_type,
                }
            }

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error attempt {attempt+1}: {e}")
            time.sleep(2)
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout attempt {attempt+1}, retrying...")
            time.sleep(5)
        except Exception as e:
            logger.warning(f"Error attempt {attempt+1}: {e}")
            time.sleep(3)

    return None


def generate_dataset(
    chunks_file: str = "data/processed/chunks.jsonl",
    output_dir: str = "data/instructions",
    pairs_per_chunk: int = 2,
    max_chunks: int | None = None,
) -> list[dict]:
    """
    Generate instruction dataset from all chunks.
    Saves incrementally — safe to resume if interrupted.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / "instructions.jsonl"

    # Load chunks
    chunks = []
    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if max_chunks:
        chunks = chunks[:max_chunks]

    logger.info(f"Generating Q&A pairs for {len(chunks)} chunks...")

    all_pairs = []
    failed = 0

    # Resume support — skip already processed chunk_ids
    processed_ids = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    processed_ids.add(rec["metadata"]["source_chunk_id"])
                except Exception:
                    pass
        logger.info(f"Resuming — {len(processed_ids)} chunks already done")

    with open(output_file, "a", encoding="utf-8") as out_f:
        for chunk in tqdm(chunks, desc="Generating Q&A pairs"):
            if chunk["chunk_id"] in processed_ids:
                continue

            q_types = QUESTION_TYPES[:pairs_per_chunk]

            for q_type in q_types:
                pair = generate_qa_pair(chunk, q_type)
                if pair:
                    out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    out_f.flush()
                    all_pairs.append(pair)
                else:
                    failed += 1

    logger.info(f"Done. Generated: {len(all_pairs)}, Failed: {failed}")
    return all_pairs