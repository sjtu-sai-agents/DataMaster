#!/usr/bin/env python3
"""
Serve Qwen3-Embedding-0.6B from ModelScope behind a small HTTP API.
"""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_MAX_LENGTH = 8192


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return array / norms


def download_from_modelscope(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'modelscope'. Install it with: pip install modelscope"
        ) from exc

    try:
        return snapshot_download(model_name)
    except TypeError:
        return snapshot_download(model_id=model_name)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the final non-padding token, matching common decoder-only embedding usage."""
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class EmbeddingModel:
    def __init__(self, model_name: str, batch_size: int) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = download_from_modelscope(model_name)
        self.lock = threading.Lock()

        self.backend = "transformers"
        self.sentence_model = None
        self.tokenizer = None
        self.model = None

        try:
            from sentence_transformers import SentenceTransformer

            self.sentence_model = SentenceTransformer(self.model_dir, device=self.device)
            self.backend = "sentence_transformers"
            print(
                f"[INFO] Loaded {model_name} from {self.model_dir} with sentence-transformers on {self.device}",
                flush=True,
            )
            return
        except ImportError:
            print(
                "[INFO] sentence-transformers is not installed; falling back to transformers pooling.",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[WARN] Failed to load with sentence-transformers ({exc}); falling back to transformers.",
                flush=True,
            )

        from transformers import AutoModel, AutoTokenizer

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        print(
            f"[INFO] Loaded {model_name} from {self.model_dir} with transformers on {self.device}",
            flush=True,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []
        with self.lock:
            for batch in batched(texts, self.batch_size):
                if self.backend == "sentence_transformers":
                    assert self.sentence_model is not None
                    embeddings = self.sentence_model.encode(
                        batch,
                        batch_size=self.batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    all_embeddings.append(embeddings.astype(np.float32))
                    continue

                assert self.tokenizer is not None
                assert self.model is not None
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=DEFAULT_MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    pooled = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embeddings.append(pooled.detach().cpu().float().numpy())

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        return l2_normalize(embeddings).astype(np.float32)


def build_app(model_name: str, batch_size: int) -> FastAPI:
    embedding_model = EmbeddingModel(model_name=model_name, batch_size=batch_size)
    app = FastAPI(title="Qwen3 ModelScope Embedding Server")

    @app.post("/embed", response_model=EmbedResponse)
    def embed(payload: EmbedRequest) -> EmbedResponse:
        if payload.texts is None:
            raise HTTPException(status_code=400, detail="'texts' must be provided")
        if not isinstance(payload.texts, list):
            raise HTTPException(status_code=400, detail="'texts' must be a list")

        texts = ["" if text is None else str(text) for text in payload.texts]
        try:
            embeddings = embedding_model.encode(texts)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc

        return EmbedResponse(embeddings=embeddings.tolist())

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve Qwen3-Embedding-0.6B from ModelScope on POST /embed."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    app = build_app(model_name=args.model_name, batch_size=args.batch_size)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


# Run instruction:
# python scripts/serve_qwen_embedding_modelscope.py \
#   --host 127.0.0.1 \
#   --port 8010 \
#   --batch-size 32
