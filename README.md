# test_paskal_litgpt

Minimal **LitGPT** codepath for **offline proactive audio reply (PASKAL)** evaluation: LitGPT decoder + Qwen2.5-Omni **audio tower**, driven by `litgpt/finetune/generate/offline_paskal.py`.

Python package name (editable install): **`litgpt-paskal-offline`** (see `pyproject.toml`). This folder is intentionally small (only what inference needs).

## Requirements

- **Python** ≥ 3.10  
- **CUDA** recommended for full benchmarks (CPU is supported but very slow)  
- **Artifacts you must supply**: tokenizer directory (`model_config.yaml` + tokenizer files), one or more LitGPT checkpoint **files** (paths you pass explicitly), Qwen2.5-Omni config directory for the audio tower, adapted audio-tower `.pt` weights, and an evaluation JSONL (see below).

## Installation

**Option A — editable install (recommended; installs the `litgpt` package):**

```bash
cd /path/to/test_paskal_litgpt
pip install -e .
```

**Option B — requirements file only:**

```bash
pip install -r requirements.txt
# Then ensure this repo root is on PYTHONPATH, or still use: pip install -e .
```

Install **PyTorch** with the CUDA build that matches your machine ([pytorch.org](https://pytorch.org/)). The pins in `requirements.txt` / `pyproject.toml` assume `torch>=2.5,<2.7`.

**Optional:**

- **Semantic reranking** (metrics that use `FlagEmbedding` in `offline_paskal.py`): `pip install FlagEmbedding`
- **FlashAttention**: only if your model forward uses it — install `flash-attn` per upstream instructions.

## Evaluation JSONL

Each line should be a JSON object with at least:

- `path` — path to an audio file readable on disk  
- `decision` — ground-truth label (e.g. `RESPOND` / `IGNORE`)  
- `id` — optional but needed if you use semantic standard answers keyed by id  

## Running offline evaluation

Paths are **never hard-coded**: use CLI flags or `PASKAL_*` environment variables (CLI overrides env).

| CLI flag | Environment variable | Description |
|----------|----------------------|-------------|
| `--tokenizer-dir` | `PASKAL_TOKENIZER_DIR` | Directory with `model_config.yaml` and tokenizer files |
| `--checkpoint` | `PASKAL_CHECKPOINTS` | LitGPT checkpoint file path; repeat `--checkpoint` for multiple, or set env to comma-separated paths |
| `--audio-tower-config` | `PASKAL_AUDIO_TOWER_CONFIG` | HF-style directory with Qwen2.5-Omni config (audio tower) |
| `--audio-tower-weights` | `PASKAL_AUDIO_TOWER_WEIGHTS` | `.pt` state dict for the adapted audio tower |
| `--dataset-jsonl` | `PASKAL_DATASET_JSONL` | Evaluation JSONL |
| `--output-dir` | `PASKAL_OUTPUT_DIR` | Output root (one subdirectory per checkpoint) |
| `--semantic-standard-jsonl` | `PASKAL_SEMANTIC_STANDARD_JSONL` | Optional JSONL with `standard_answers` per `id` |
| `--semantic-model-dir` | `PASKAL_SEMANTIC_MODEL_DIR` | Optional local reranker model directory |
| `--semantic-threshold` | `PASKAL_SEMANTIC_THRESHOLD` | Default `0.5` |
| `--max-jobs-per-gpu` | `PASKAL_MAX_JOBS_PER_GPU` | Parallel checkpoint tasks per GPU (default `2`) |
| `--system-prompt-file` | — | Optional UTF-8 text file overriding the default system prompt |

**Temporary audio segments:** `PASKAL_AUDIO_BUFFER` (exact directory) or `PASKAL_AUDIO_BUFFER_ROOT` (parent; defaults to system temp). Segments are written under `litgpt_paskal_segments/<pid>/`.

### Example

```bash
python litgpt/finetune/generate/offline_paskal.py \
  --tokenizer-dir /path/to/tokenizer_dir \
  --checkpoint /path/to/checkpoints/step-065000/step_065000_statedict.pt \
  --audio-tower-config /path/to/qwen_omni_config_dir \
  --audio-tower-weights /path/to/audio_tower.pt \
  --dataset-jsonl /path/to/dataset.jsonl \
  --output-dir /path/to/results
```

Multiple checkpoints:

```bash
python litgpt/finetune/generate/offline_paskal.py \
  --tokenizer-dir /path/to/tokenizer_dir \
  --checkpoint /path/to/a.pt \
  --checkpoint /path/to/b.pt \
  ...
```

Full CLI: `python litgpt/finetune/generate/offline_paskal.py --help`.

### Outputs

Under `--output-dir`:

- Per checkpoint: `results.jsonl`, `stats.json`  
- Summary: `proactive_test_results_all_checkpoints_summary.json`

## Replication checklist

To match published numbers, fix **dataset revision**, **exact checkpoint file(s)** used for evaluation, **tokenizer**, **audio tower weights**, **system prompt** (if custom), and **software pins** . Document CUDA/driver versions if you report GPU results.

## Full workflow from zero

1. **Machine**: Linux recommended; NVIDIA GPU + driver for CUDA; Python ≥ 3.10.
2. **Enter the repo** (clone your fork or copy this directory):

   ```bash
   cd /path/to/test_paskal_litgpt
   ```

3. **Virtualenv (recommended)**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install PyTorch** with the CUDA build that matches your GPU ([pytorch.org](https://pytorch.org/)), then install this project:

   ```bash
   pip install -e .
   ```

   Or: `pip install -r requirements.txt` then `pip install -e .` so imports resolve.

5. **Collect artifacts** on disk: tokenizer dir (`model_config.yaml` + tokenizer files), LitGPT checkpoint file path(s) (`lit_model.pth` or `*_statedict.pt`), Qwen2.5-Omni HF config dir for the audio tower, adapted audio-tower `.pt`, evaluation JSONL (`path`, `decision`, optional `id`).
6. **Run** (CLI example — adjust paths):

   ```bash
   python litgpt/finetune/generate/offline_paskal.py \
     --tokenizer-dir /path/to/tokenizer_dir \
     --checkpoint /path/to/lit_model.pth \
     --audio-tower-config /path/to/qwen_omni_config_dir \
     --audio-tower-weights /path/to/audio_tower.pt \
     --dataset-jsonl /path/to/dataset.jsonl \
     --output-dir /path/to/results
   ```

   Equivalent: set `PASKAL_TOKENIZER_DIR`, `PASKAL_CHECKPOINTS` (comma-separated checkpoint files), `PASKAL_AUDIO_TOWER_CONFIG`, `PASKAL_AUDIO_TOWER_WEIGHTS`, `PASKAL_DATASET_JSONL`, `PASKAL_OUTPUT_DIR` and run the script with no path flags.

7. **Optional — semantic metrics**: install `FlagEmbedding`, pass `--semantic-standard-jsonl` and `--semantic-model-dir` (see table above).

8. **Optional — publish to GitHub** (first time from this folder):

   ```bash
   git init
   git branch -M main
   git add .
   git commit -m "Initial commit"
   git remote add origin git@github.com:YOUR_USER/YOUR_REPO.git
   ssh -T git@github.com   # should say “successfully authenticated”
   git push -u origin main
   ```

   Use HTTPS + Personal Access Token instead of SSH if you prefer; see GitHub docs for `credential.helper`.

## License

See [LICENSE](LICENSE) (Apache 2.0).
