"""In-process embedding for serverless mode (MLX on Apple Silicon, ONNX elsewhere)."""

import os
import platform

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # suppress tokenizers warning

import sys
from typing import Optional

import numpy as np


def _use_mlx() -> bool:
    """Check if MLX backend is available (Apple Silicon only)."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def _hf_download(repo_id: str, filename: str, subfolder: str | None = None) -> str:
    """Download a single file from HuggingFace, trying offline cache first."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id, filename, subfolder=subfolder, local_files_only=True)
    except Exception:
        return hf_hub_download(repo_id, filename, subfolder=subfolder)


def _snapshot_download(repo_id: str) -> str:
    """Download model repo, trying offline cache first to avoid proxy issues."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        print("Downloading model for first time...", file=sys.stderr, flush=True)
        return snapshot_download(repo_id)

# Unified MLX repos with dynamic LoRA adapter switching
MLX_MODELS = {
    "jina-embeddings-v5-small": "jinaai/jina-embeddings-v5-text-small-mlx",
    "jina-embeddings-v5-nano": "jinaai/jina-embeddings-v5-text-nano-mlx",
}

# Code models: single checkpoint, no LoRA switching
CODE_MODELS_MAP = {
    "jina-code-embeddings-0.5b": "jinaai/jina-code-embeddings-0.5b-mlx",
    "jina-code-embeddings-1.5b": "jinaai/jina-code-embeddings-1.5b-mlx",
}

CODE_MODELS = set(CODE_MODELS_MAP.keys())

# ONNX model mapping: (model_short_name, task) -> (repo_id, onnx_subfolder_or_None)
ONNX_V5_MODELS = {
    ("jina-embeddings-v5-nano", "retrieval"): ("jinaai/jina-embeddings-v5-text-nano-retrieval", "onnx"),
    ("jina-embeddings-v5-nano", "text-matching"): ("jinaai/jina-embeddings-v5-text-nano-text-matching", "onnx"),
    ("jina-embeddings-v5-nano", "classification"): ("jinaai/jina-embeddings-v5-text-nano-classification", "onnx"),
    ("jina-embeddings-v5-nano", "clustering"): ("jinaai/jina-embeddings-v5-text-nano-clustering", "onnx"),
    ("jina-embeddings-v5-small", "retrieval"): ("jinaai/jina-embeddings-v5-text-small-retrieval", "onnx"),
    ("jina-embeddings-v5-small", "text-matching"): ("jinaai/jina-embeddings-v5-text-small-text-matching", "onnx"),
    ("jina-embeddings-v5-small", "classification"): ("jinaai/jina-embeddings-v5-text-small-classification", "onnx"),
    ("jina-embeddings-v5-small", "clustering"): ("jinaai/jina-embeddings-v5-text-small-clustering", "onnx"),
}

ONNX_CODE_MODELS = {
    "jina-code-embeddings-1.5b": ("herMaster/jina-code-embeddings-1.5b-ONNX", None),
}

# Prompt prefixes for v5 ONNX models (from config_sentence_transformers.json)
V5_PROMPTS = {
    "query": "Query: ",
    "document": "Document: ",
    "passage": "Document: ",
}

# Instruction prefixes for code models
CODE_INSTRUCTIONS = {
    "nl2code": {
        "query": "Find the most relevant code snippet given the following query:\n",
        "passage": "Candidate code snippet:\n",
    },
    "qa": {
        "query": "Find the most relevant answer given the following question:\n",
        "passage": "Candidate answer:\n",
    },
    "code2code": {
        "query": "Find an equivalent code snippet given the following code snippet:\n",
        "passage": "Candidate code snippet:\n",
    },
    "code2nl": {
        "query": "Find the most relevant comment given the following code snippet:\n",
        "passage": "Candidate comment:\n",
    },
    "code2completion": {
        "query": "Find the most relevant completion given the following start of code snippet:\n",
        "passage": "Candidate completion:\n",
    },
}

# Supported Matryoshka dimensions
MATRYOSHKA_DIMS = {32, 64, 128, 256, 512, 768, 1024}

VALID_TASKS = {"retrieval", "text-matching", "clustering", "classification"}
CODE_TASKS = {"nl2code", "qa", "code2code", "code2nl", "code2completion"}
ALL_TASKS = VALID_TASKS | CODE_TASKS
VALID_PROMPT_NAMES = {"query", "document", "passage"}

# Guardrails
MAX_BATCH_SIZE = 512
MAX_SEQ_LENGTH = {
    "jina-embeddings-v5-small": 32768,
    "jina-embeddings-v5-nano": 8192,
}

# Global model cache: key = model_name -> JinaMultiTaskModel or (code_model, tokenizer)
_models: dict = {}

_first_load = True


# --- ONNX Runtime backend ---

# Cache for ONNX sessions and tokenizers: key = (model_name, task) or model_name
_onnx_sessions: dict = {}
_onnx_tokenizers: dict = {}


def _last_token_pool(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Extract embedding from the last non-padding token and L2-normalize."""
    seq_lengths = attention_mask.sum(axis=1).astype(int) - 1
    batch_idx = np.arange(hidden_states.shape[0])
    embeddings = hidden_states[batch_idx, seq_lengths]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    return embeddings / norms


def _onnx_embed(
    texts: list[str],
    model_name: str,
    task: str,
    prompt_name: str | None = None,
) -> np.ndarray:
    """ONNX Runtime embedding for non-MLX platforms."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    global _first_load

    is_code = model_name in CODE_MODELS

    if is_code:
        if task not in CODE_TASKS:
            raise ValueError(f"Unsupported task for code model: {task}. Supported: {', '.join(CODE_TASKS)}")
        if model_name not in ONNX_CODE_MODELS:
            raise ValueError(
                f"ONNX not available for {model_name} on this platform. "
                f"Supported ONNX code models: {', '.join(ONNX_CODE_MODELS)}"
            )
        repo_id, subfolder = ONNX_CODE_MODELS[model_name]
        cache_key = model_name
    else:
        if task not in VALID_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported: {', '.join(VALID_TASKS)}")
        if model_name not in MLX_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported: {', '.join(list(MLX_MODELS) + list(CODE_MODELS_MAP))}")
        key = (model_name, task)
        if key not in ONNX_V5_MODELS:
            raise ValueError(f"ONNX not available for {model_name} with task {task}")
        repo_id, subfolder = ONNX_V5_MODELS[key]
        cache_key = key

    # Load or retrieve cached session
    if cache_key not in _onnx_sessions:
        if _first_load:
            print("Loading model...", end="", file=sys.stderr, flush=True)
            _first_load = False
            _print_done = True
        else:
            _print_done = False

        # Download ONNX model files
        onnx_path = _hf_download(repo_id, "model.onnx", subfolder=subfolder)
        # Also download external data file if exists
        try:
            _hf_download(repo_id, "model.onnx_data", subfolder=subfolder)
        except Exception:
            pass  # some models may not have external data

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        _onnx_sessions[cache_key] = session

        # Download and load tokenizer
        tok_path = _hf_download(repo_id, "tokenizer.json")
        _onnx_tokenizers[cache_key] = Tokenizer.from_file(tok_path)

        if _print_done:
            print(" done", file=sys.stderr, flush=True)

    session = _onnx_sessions[cache_key]
    tokenizer = _onnx_tokenizers[cache_key]

    # Apply instruction prefix
    if is_code:
        role = prompt_name or "query"
        if role == "document":
            role = "passage"
        prefix = CODE_INSTRUCTIONS.get(task, {}).get(role, "")
        texts = [prefix + t for t in texts]
    else:
        if task == "retrieval":
            role = prompt_name or "query"
            if role == "document":
                role = "passage"
            prefix = V5_PROMPTS.get(role, "")
            texts = [prefix + t for t in texts]

    # Tokenize with padding
    encodings = tokenizer.encode_batch(texts)
    max_len = max(len(e.ids) for e in encodings)
    input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
    attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)

    for i, enc in enumerate(encodings):
        length = len(enc.ids)
        input_ids[i, :length] = enc.ids
        attention_mask[i, :length] = 1

    # Run inference
    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    hidden_states = outputs[0]  # [batch, seq_len, hidden_size]

    return _last_token_pool(hidden_states, attention_mask)


def get_model(model_name: str, task: str):
    """Load or retrieve cached MLX model for the given model name.

    For v5 models, returns a JinaMultiTaskModel with dynamic adapter switching.
    For code models, returns (model, tokenizer) tuple.
    """
    global _first_load

    is_code = model_name in CODE_MODELS

    if not is_code and model_name not in MLX_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {', '.join(list(MLX_MODELS) + list(CODE_MODELS_MAP))}")

    if is_code:
        if task not in CODE_TASKS:
            raise ValueError(f"Unsupported task for code model: {task}. Supported: {', '.join(CODE_TASKS)}")
    else:
        if task not in VALID_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported: {', '.join(VALID_TASKS)}")

    if model_name not in _models:
        if _first_load:
            print("Loading model...", end="", file=sys.stderr, flush=True)
            _first_load = False
            _print_done = True
        else:
            _print_done = False

        if is_code:
            import importlib.util
            import json

            import mlx.core as mx
            from tokenizers import Tokenizer

            model_dir = _snapshot_download(CODE_MODELS_MAP[model_name])

            with open(os.path.join(model_dir, "config.json")) as f:
                config = json.load(f)

            spec = importlib.util.spec_from_file_location(
                f"jina_mlx_model_{model_name}",
                os.path.join(model_dir, "model.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            model = mod.JinaCodeEmbeddingModel(config)
            weights = mx.load(os.path.join(model_dir, "model.safetensors"))
            model.load_weights(list(weights.items()))
            mx.eval(model.parameters())

            tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
            _models[model_name] = (model, tokenizer)
        else:
            # v5 models: use unified repo with dynamic LoRA

            model_dir = _snapshot_download(MLX_MODELS[model_name])

            # Import utils.py from the downloaded repo
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                f"jina_mlx_utils_{model_name}",
                os.path.join(model_dir, "utils.py"),
            )
            utils_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils_mod)

            multi_model = utils_mod.load_model(model_dir)
            _models[model_name] = multi_model

        if _print_done:
            print(" done", file=sys.stderr, flush=True)

    return _models[model_name]


class LocalEmbedder:
    """In-process embedding, same interface as EmbeddingClient.

    Uses MLX on Apple Silicon, ONNX Runtime elsewhere.
    """

    def embed(
        self,
        texts: list[str],
        model: str = "jina-embeddings-v5-small",
        task: str = "retrieval",
        prompt_name: str = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        if not _use_mlx():
            return _onnx_embed(texts, model, task, prompt_name)

        # MLX path (Apple Silicon only)
        import mlx.core as mx

        cached = get_model(model, task)
        is_code = model in CODE_MODELS

        if is_code:
            model_obj, tokenizer = cached
            prompt_type = prompt_name or "query"
            if prompt_type == "document":
                prompt_type = "passage"
            embeddings = model_obj.encode(
                texts,
                tokenizer,
                task=task,
                prompt_type=prompt_type,
            )
        else:
            # JinaMultiTaskModel with dynamic LoRA switching
            multi_model = cached
            multi_model.switch_task(task)

            if task == "retrieval":
                pn = prompt_name or "query"
                if pn not in VALID_PROMPT_NAMES:
                    raise ValueError(f"Invalid prompt_name: {pn}")
                task_type = f"retrieval.{pn}" if pn == "query" else "retrieval.passage"
            else:
                task_type = task

            embeddings = multi_model.encode(texts, task_type=task_type)

        mx.eval(embeddings)
        return np.array(embeddings.tolist())

    def health_check(self) -> bool:
        return True

    def close(self):
        pass
