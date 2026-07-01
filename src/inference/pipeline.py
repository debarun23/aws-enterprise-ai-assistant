"""
src/inference/pipeline.py - Fixed RAG inference pipeline.
"""
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.retrieval.vector_store import AWSVectorStore
from src.utils import load_config, get_logger

logger = logging.getLogger(__name__)

AWS_KEYWORDS = {
    "s3", "ec2", "lambda", "iam", "rds", "vpc", "cloudwatch",
    "dynamodb", "aws", "amazon", "bucket", "instance", "function",
    "role", "policy", "subnet", "security group", "cloud",
}

OUT_OF_SCOPE_REPLY = (
    "I am specialized in AWS services only. "
    "This question appears to be outside my domain. "
    "Please ask about AWS services like S3, EC2, Lambda, IAM, RDS, VPC, "
    "CloudWatch, or DynamoDB."
)


class AWSInferencePipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger("inference", self.config)
        self._loaded = False
        self.model = None
        self.tokenizer = None
        self.retriever = None
        inf = self.config["inference"]
        self.max_new_tokens       = inf.get("max_new_tokens", 300)
        self.max_prompt_tokens    = inf.get("max_prompt_tokens", 400)
        self.ctx_tokens_per_chunk = inf.get("context_tokens_per_chunk", 60)
        self.temperature          = inf.get("temperature", 0.3)
        self.top_p                = inf.get("top_p", 0.9)
        self.confidence_threshold = inf.get("confidence_threshold", 0.6)
        self.refusal_threshold    = inf.get("refusal_threshold", 0.3)
        self.top_k                = self.config["retrieval"]["top_k"]

    def load(self):
        adapter_path = str(Path(self.config["training"]["output_dir"]) / "final_adapter")
        base_model = self.config["model"]["base_model"]

        self.logger.info("Loading tokenizer from: " + adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info("Loading quantized base model...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.logger.info("Applying LoRA adapter...")
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

        self.logger.info("Loading FAISS retriever...")
        self.retriever = AWSVectorStore(self.config)
        self.retriever.load()

        self._loaded = True
        self.logger.info("Pipeline ready")

    def _build_prompt(self, question: str, docs: list) -> str:
        system = (
            "You are an AWS expert assistant. "
            "Use ONLY the context below to answer. "
            "Be specific and complete."
        )
        trimmed_chunks = []
        for i, doc in enumerate(docs):
            service = doc.get("service", "AWS").upper()
            text = doc.get("text", doc.get("content", ""))
            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.ctx_tokens_per_chunk,
            )["input_ids"]
            trimmed = self.tokenizer.decode(ids, skip_special_tokens=True)
            label = "[Source " + str(i + 1) + " - " + service + "]"
            trimmed_chunks.append(label + "\n" + trimmed)

        context = "\n\n".join(trimmed_chunks)
        prompt = (
            system
            + "\n\nContext:\n" + context
            + "\n\nQuestion: " + question
            + "\nAnswer:"
        )
        ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_tokens,
        )["input_ids"]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _is_aws_question(self, question: str) -> bool:
        return any(kw in question.lower() for kw in AWS_KEYWORDS)

    def _estimate_confidence(self, docs: list) -> float:
        if not docs:
            return 0.0
        scores = [doc.get("score", 0.0) for doc in docs]
        raw = 1.0 / (1.0 + (sum(scores) / len(scores)))
        return round(min(max(raw * 2.5, 0.0), 1.0), 3)

    @torch.inference_mode()
    def _generate_answer(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_tokens,
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        self.logger.debug("Prompt tokens: " + str(prompt_len))
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )
        new_tokens = outputs[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate(self, question: str, top_k: int = None) -> dict:
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded. Call .load() first.")
        k = top_k or self.top_k
        if not self._is_aws_question(question):
            return {
                "question":   question,
                "answer":     OUT_OF_SCOPE_REPLY,
                "confidence": 0.0,
                "refused":    True,
                "sources":    [],
            }
        docs = self.retriever.retrieve(question, top_k=k)
        confidence = self._estimate_confidence(docs)
        refused = confidence < self.refusal_threshold
        if refused:
            return {
                "question":   question,
                "answer":     OUT_OF_SCOPE_REPLY,
                "confidence": confidence,
                "refused":    True,
                "sources":    [self._doc_to_source(d) for d in docs],
            }
        prompt = self._build_prompt(question, docs)
        answer = self._generate_answer(prompt)
        return {
            "question":   question,
            "answer":     answer,
            "confidence": confidence,
            "refused":    False,
            "sources":    [self._doc_to_source(d) for d in docs],
        }

    def _doc_to_source(self, doc: dict) -> dict:
        return {
            "service": doc.get("service", "unknown"),
            "title":   doc.get("title", "AWS Documentation"),
            "url":     doc.get("url", ""),
            "score":   round(doc.get("score", 0.0), 4),
        }
