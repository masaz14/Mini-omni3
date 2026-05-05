# Mini-Omni3
<p align="center"><strong style="font-size: 18px;">
Mini-Omni 3: Towards Streaming Large Audio-Language Models
</strong>
<p align="center">
🤗 <a href="https://huggingface.co/masaz14/mini-omini3">Hugging Face</a>  
| 📖 <a href="https://github.com/masaz14/Mini-omni3">Github</a> 
| 📑 <a href="">Technical report</a> |
🤗 <a href="https://huggingface.co/datasets/masaz14/Proactive-Sound-Effect-Benchmark">Proactive Benchmark</a>
| 📖<a href="https://github.com/masaz14/Proactive-Sound-Effect-Benchmark">Benchmark</a> 
</p>

A minimal inference-only implementation for offline proactive audio reply evaluation.
This repository contains:
- the LitGPT decoder
- the Qwen2.5-Omni audio tower
- the offline evaluation pipeline in `litgpt/finetune/generate/offline_paskal.py`

The repository is intentionally lightweight and includes only the components required for inference and evaluation.

## Install

Create a new conda environment and install the required packages:

```sh
conda create -n Mini-omni3 python=3.10
conda activate Mini-omni3

git clone https://github.com/masaz14/Mini-omni3.git
pip install -r requirements.txt
```

## Evaluation JSONL
Please refer to <a href="https://github.com/masaz14/Proactive-Sound-Effect-Benchmark">Benchmark</a>:
Each line should be a JSON object with at least:
- `path` — path to an audio file readable on disk  
- `decision` — ground-truth label (e.g. `RESPOND` / `IGNORE`)  
- `id` — optional but needed if you use semantic standard answers keyed by id  

## Running offline evaluation

Paths are never hard-coded: use CLI flags or `PASKAL_*` environment variables (CLI overrides env).

| CLI flag | Environment variable | Description |
|----------|----------------------|-------------|
| `--tokenizer-dir` | `PASKAL_TOKENIZER_DIR` | Directory with `model_config.yaml` and tokenizer files |
| `--checkpoint` | `PASKAL_CHECKPOINT` | LitGPT checkpoint file path (**single file**) |
| `--audio-tower-config` | `PASKAL_AUDIO_TOWER_CONFIG` | HF-style directory with Qwen2.5-Omni config (audio tower) |
| `--audio-tower-weights` | `PASKAL_AUDIO_TOWER_WEIGHTS` | `.pt` state dict for the adapted audio tower |
| `--dataset-jsonl` | `PASKAL_DATASET_JSONL` | Evaluation JSONL |
| `--output-dir` | `PASKAL_OUTPUT_DIR` | Output root (a subdirectory will be created for this checkpoint) |
| `--semantic-standard-jsonl` | `PASKAL_SEMANTIC_STANDARD_JSONL` | Optional JSONL with `standard_answers` per `id` |
| `--semantic-model-dir` | `PASKAL_SEMANTIC_MODEL_DIR` | Optional local reranker model directory |
| `--semantic-threshold` | `PASKAL_SEMANTIC_THRESHOLD` | Default `0.5` |
| `--system-prompt-file` | — | Optional UTF-8 text file overriding the default system prompt |


### Example

```bash
python litgpt/finetune/generate/offline_paskal.py \
  --tokenizer-dir /path/to/tokenizer_dir \
  --checkpoint /path/to/lit_model.pth \
  --audio-tower-config /path/to/qwen_omni_config_dir \
  --audio-tower-weights /path/to/audio_tower.pt \
  --dataset-jsonl /path/to/dataset.jsonl \
  --output-dir /path/to/results
```
Full CLI: `python litgpt/finetune/generate/offline_paskal.py --help`.

### Outputs

Under `--output-dir`:
- `results.jsonl`  
- `stats.json`

 **Optional — semantic metrics**: install `FlagEmbedding`, pass `--semantic-standard-jsonl` and `--semantic-model-dir` (see table above).


## License

See [LICENSE](LICENSE) (Apache 2.0).
