import logging
import asyncio
import torch
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.retrieval.vector_store import AWSVectorStore

logger = logging.getLogger(__name__)

tokenizer = None
model = None
retriever = None

BASE_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "models/checkpoints/final_adapter"
INDEX_DIR    = "data/processed/faiss_index"

AWS_KEYWORDS = {
    "s3", "ec2", "lambda", "iam", "rds", "vpc", "cloudwatch",
    "dynamodb", "aws", "amazon", "bucket", "instance", "function",
    "role", "policy", "subnet", "security group", "cloud",
}

OUT_OF_SCOPE = (
    "I am specialized in AWS services only. "
    "Please ask about AWS services like S3, EC2, Lambda, IAM, RDS, or DynamoDB."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global tokenizer, model, retriever

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    logger.info("Loading retriever...")
    retriever = AWSVectorStore("sentence-transformers/all-MiniLM-L6-v2")
    retriever.load(INDEX_DIR)

    logger.info("API ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="AWS Enterprise AI Assistant",
    description="Domain-specific AI assistant fine-tuned on AWS documentation",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    refused: bool
    sources: list[str]


def build_prompt(question: str, docs: list[dict], ctx_tokens: int = 60) -> str:
    system = (
        "You are an AWS expert assistant. "
        "Use ONLY the context below to answer. "
        "Be specific and complete."
    )
    chunks = []
    for i, doc in enumerate(docs):
        service = doc.get("service", "AWS").upper()
        text    = doc.get("chunk_text", "")
        ids     = tokenizer(text, add_special_tokens=False,
                            truncation=True, max_length=ctx_tokens)["input_ids"]
        trimmed = tokenizer.decode(ids, skip_special_tokens=True)
        chunks.append(f"[Source {i+1} - {service}]\n{trimmed}")
    context = "\n\n".join(chunks)
    return system + "\n\nContext:\n" + context + "\n\nQuestion: " + question + "\nAnswer:"


@torch.inference_mode()
def run_inference(question: str, top_k: int = 3) -> dict:
    if not any(kw in question.lower() for kw in AWS_KEYWORDS):
        return {"answer": OUT_OF_SCOPE, "confidence": 0.0,
                "refused": True, "sources": []}

    docs       = retriever.query(question, top_k=top_k)
    scores     = [d.get("similarity_score", 0.0) for d in docs]
    confidence = round(min(sum(scores) / len(scores), 1.0), 3)

    prompt     = build_prompt(question, docs)
    inputs     = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=400).to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.15,
    )
    answer = tokenizer.decode(
        outputs[0][prompt_len:], skip_special_tokens=True
    ).strip()

    # Cut off at first "Explanation:" or "Corrected" — model self-commentary
    for stop in ["\nExplanation:", "\nCorrected", "\nNote:", "Be specific and complete"]:
        if stop in answer:
            answer = answer[:answer.index(stop)].strip()

    return {
        "answer":     answer,
        "confidence": confidence,
        "refused":    False,
        "sources":    [d.get("service", "?") for d in docs],
    }


@app.get("/health")
async def health_check() -> dict:
    return {"status": "healthy", "model": BASE_MODEL, "version": "1.0.0"}


@app.get("/services")
async def list_services() -> dict:
    return {"services": ["EC2","S3","Lambda","IAM","RDS","VPC","CloudWatch","DynamoDB"]}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: run_inference(request.question, request.top_k)
        )
        return AskResponse(**result, question=request.question)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_stream(request: AskRequest) -> StreamingResponse:
    async def generate():
        import json
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: run_inference(request.question, request.top_k)
        )
        words = result["answer"].split()
        for i, word in enumerate(words):
            yield f"data: {word if i == 0 else ' ' + word}\n\n"
            await asyncio.sleep(0.02)
        meta = {"confidence": result["confidence"],
                "refused": result["refused"], "sources": result["sources"]}
        yield f"data: [DONE] {json.dumps(meta)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")