# <img src="logo.svg?v=2" alt="" width="28" height="28" style="vertical-align: middle;"/> jina-grep

Semantic grep powered by Jina embeddings, running locally via MLX (macOS Apple Silicon) or ONNX Runtime (Windows/Linux).

> **Fork note:** This is a cross-platform fork of [jina-ai/jina-grep-cli](https://github.com/jina-ai/jina-grep-cli). The original only supports macOS Apple Silicon. This fork adds Windows and Linux support via ONNX Runtime, while keeping full MLX compatibility on Mac.

Four modes: pipe grep output for semantic reranking, search files directly with natural language, zero-shot classification, or code search.


| Model | Params | Dims | Max Seq | Matryoshka | Tasks |
|-------|--------|------|---------|------------|-------|
| jina-embeddings-v5-small | 677M | 1024 | 32768 | 32-1024 | retrieval, text-matching, clustering, classification |
| jina-embeddings-v5-nano | 239M | 768 | 8192 | 32-768 | retrieval, text-matching, clustering, classification |
| jina-code-embeddings-1.5b | 1.54B | 1536 | 32768 | 128-1536 | nl2code, code2code, code2nl, code2completion, qa |
| jina-code-embeddings-0.5b | 0.49B | 896 | 32768 | 64-896 | nl2code, code2code, code2nl, code2completion, qa |

## Platform Support

| Platform | Backend | How it works |
|----------|---------|-------------|
| macOS Apple Silicon | MLX | Unified MLX checkpoints with dynamic LoRA adapter switching. Pure Metal GPU, no PyTorch. |
| Windows / Linux | ONNX Runtime | Pre-exported ONNX models from HuggingFace. CPU inference, lightweight (~50MB runtime). |

Backend is auto-detected at runtime. No configuration needed.

**ONNX model availability:**

| Model | ONNX Status |
|-------|-------------|
| v5-nano (all 4 tasks) | Official ONNX exports by Jina AI |
| v5-small (all 4 tasks) | Official ONNX exports by Jina AI |
| code-1.5b | Community ONNX export ([herMaster/jina-code-embeddings-1.5b-ONNX](https://huggingface.co/herMaster/jina-code-embeddings-1.5b-ONNX)) |
| code-0.5b | Not available (MLX only) |

## Install

**macOS (Apple Silicon):**

```bash
git clone https://github.com/dalan2014/jina-grep-cli.git && cd jina-grep-cli
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

**Windows / Linux:**

```bash
git clone https://github.com/dalan2014/jina-grep-cli.git && cd jina-grep-cli
pip install .
```

Dependencies are platform-conditional: `mlx` is only installed on macOS ARM64, `onnxruntime` is installed elsewhere.

Requirements: Python 3.10+.

## Usage

Two modes:

**Serverless (default):** Model loads in-process, runs the query, exits. No server, no background processes. Best for: occasional use, scripts, CI.

- On macOS: Uses MLX with dynamic LoRA adapter switching. MLX loads weights via mmap, so macOS keeps them in page cache after exit.
- On Windows/Linux: Uses ONNX Runtime. Models are downloaded from HuggingFace on first use and cached locally.

**Persistent server:** Keep a server running across invocations. Model stays in memory, every query is fast. Best for: interactive sessions, batch workloads.

```bash
jina-grep serve start   # keep running in background
# ... run as many queries as you want ...
jina-grep serve stop    # stop when done
```

Serverless mode auto-detects a running persistent server and uses it via HTTP (without stopping it afterwards).

### Pipe mode: rerank grep output

```bash
grep -rn "error" src/ | jina-grep "error handling logic"
grep -rn "def.*test" . | jina-grep "unit tests for authentication"
grep -rn "TODO" . | jina-grep "performance optimization"
```

### Standalone mode: direct semantic search

```bash
jina-grep "memory leak" src/
jina-grep -r --threshold 0.3 "database connection pooling" .
jina-grep --top-k 5 "retry with exponential backoff" *.py
```

### Code search

Use `--model` to switch to code embeddings and `--task` for code-specific tasks:

```bash
# Natural language to code: find code that matches a description
jina-grep --model jina-code-embeddings-1.5b --task nl2code "sort a list in descending order" src/

# Code to code: find similar code snippets
jina-grep --model jina-code-embeddings-1.5b --task code2code "for i in range(len(arr))" src/

# Pipe mode works too
grep -rn "def " src/ | jina-grep --model jina-code-embeddings-1.5b --task nl2code "HTTP retry with backoff"
```

> Note: `jina-code-embeddings-0.5b` is only available on macOS Apple Silicon (no ONNX export exists yet).

Code tasks:
- `nl2code` - natural language query to code (default for code models)
- `code2code` - find similar code snippets
- `code2nl` - find comments/docs for code
- `code2completion` - find completions for partial code
- `qa` - question answering over code

### Zero-shot classification

Use `-e` to specify labels. Each line gets classified to the best matching label.

```bash
# Classify code by category
jina-grep -e "database" -e "error handling" -e "data processing" -e "configuration" src/

# Read labels from file
echo -e "bug\nfeature\nrefactor\ndocs" > labels.txt
jina-grep -f labels.txt src/

# Output only the label (pipe-friendly)
jina-grep -o -e "positive" -e "negative" -e "neutral" reviews.txt

# Count per label
jina-grep -c -e "bug" -e "feature" -e "docs" src/
```

Output shows all label scores, best label highlighted:

```
src/main.py:10:def handle_error(error_code, message):  [error handling:0.744 data processing:0.756 ...]
src/config.py:1:# Configuration settings  [configuration:0.210 database:0.217 ...]
```

### Server management

```bash
jina-grep serve start [--port 8089] [--host 127.0.0.1] [--foreground]
jina-grep serve stop
jina-grep serve status
```

## Options

```
Grep-compatible flags:
  -r, -R          Recursive search (standalone mode)
  -l              Print only filenames with matches
  -L              Print only filenames without matches
  -c              Print match count per file
  -n              Print line numbers (default: on)
  -H / --no-filename   Show / hide filename
  -A/-B/-C NUM    Context lines after/before/both
  --include=GLOB  Search only matching files
  --exclude=GLOB  Skip matching files
  --exclude-dir   Skip matching directories
  --color=WHEN    never/always/auto
  -v              Invert match (lowest similarity)
  -m NUM          Max matches per file
  -q              Quiet mode

Semantic flags:
  --threshold     Cosine similarity threshold (default: 0.5)
  --top-k         Max results (default: 10)
  --model         Model name (default: jina-embeddings-v5-nano)
  --task          v5: retrieval/text-matching/clustering/classification
                  code: nl2code/code2code/code2nl/code2completion/qa
  --server        Server URL (default: http://localhost:8089)
  --granularity   line/paragraph/sentence/token (default: token)
```

## Changes from upstream

This fork adds cross-platform support while keeping full backward compatibility with the original:

- **`embedder.py`**: Added ONNX Runtime backend with auto-detection. `LocalEmbedder` transparently uses MLX or ONNX based on platform.
- **`server.py`**: Replaced inline MLX calls with `LocalEmbedder` abstraction. Fixed Windows signal handling, daemon creation, and process management.
- **`cli.py`**: Fixed `select.select()` pipe detection for Windows.
- **`pyproject.toml`**: Platform-conditional dependencies (`mlx` on macOS ARM64, `onnxruntime` elsewhere).

## Upstream

Original project: [jina-ai/jina-grep-cli](https://github.com/jina-ai/jina-grep-cli)

jina-grep is also available as `jina grep` in the unified [Jina CLI](https://github.com/jina-ai/cli).

## License

Apache-2.0
