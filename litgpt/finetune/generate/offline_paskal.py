# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""
Offline evaluation for proactive audio reply (PASKAL) using LitGPT + Qwen2.5-Omni audio tower.

Paths are configured via CLI or environment variables (see ``parse_args`` / ``--help``).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import lightning as L
import numpy as np
import torch
from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_rp = str(_REPO_ROOT)
if _rp not in sys.path:
    sys.path.insert(0, _rp)

from litgpt.generate.base_paskal_offline import generate_fn_PASK  # noqa: E402
from litgpt.model_pask import GPT, Config  # noqa: E402
from litgpt.tokenizer import Tokenizer  # noqa: E402
from litgpt.utils import get_default_supported_precision, load_checkpoint  # noqa: E402

# -----------------------------------------------------------------------------
# Special tokens (Qwen-style vocabulary offsets used by this recipe)
# -----------------------------------------------------------------------------
VOCAB_SHIFT = 151600
USER = VOCAB_SHIFT + 1
ASSISTANT = VOCAB_SHIFT + 2
TEXT_BEGIN = 151644
TEXT_END = 151643
KEEP_SILENCE = VOCAB_SHIFT + 5
AUDIO_BEGIN = 151647
AUDIO_END = 151648
PAD = VOCAB_SHIFT + 8
ONLINE = VOCAB_SHIFT + 9
ENGLISH = VOCAB_SHIFT + 11
SYSTEM = 151644

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. When there is no user text, if the audio contains a question, please answer it. If it is a sound effect, determine based on the sound whether help is needed. If the audio indicates physiological danger or environmental safety risks, provide appropriate assistance.If it reflects negative emotions, offer mental comfort.If it suggests equipment malfunction, provide warnings or safety guidance.If none of the above apply,just respond with  If the audio indicates physiological danger or environmental safety risks, provide appropriate assistance.If it reflects negative emotions, offer mental comfort.If it suggests equipment malfunction, provide warnings or safety guidance.If none of the above apply,just respond with "".",
)

SUMMARY_JSON_NAME = "proactive_test_results_all_checkpoints_summary.json"


@dataclass
class OfflinePaskalConfig:
    """Runtime paths and options (CLI + env)."""

    tokenizer_dir: Path
    system_prompt: str
    checkpoint_paths: Tuple[Path, ...]
    audio_tower_config: Path
    audio_tower_weights: Path
    dataset_jsonl: Path
    output_dir: Path
    semantic_standard_file: Optional[Path] = None
    semantic_model_path: Optional[Path] = None
    semantic_threshold: float = 0.5
    max_jobs_per_gpu: int = 2


def _env_str(key: str) -> Optional[str]:
    v = os.environ.get(key)
    return v if v is not None and str(v).strip() != "" else None


def parse_args(argv: Optional[list[str]] = None) -> OfflinePaskalConfig:
    p = argparse.ArgumentParser(
        description="Offline proactive reply benchmark with PASKAL (LitGPT + audio encoder)."
    )
    p.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Directory with model_config.yaml and tokenizer files. Env: PASKAL_TOKENIZER_DIR",
    )
    p.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Optional UTF-8 text file overriding the default system prompt.",
    )
    p.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        dest="checkpoints",
        metavar="PATH",
        help=(
            "Path to a LitGPT checkpoint file (e.g. lit_model.pth or *_statedict.pt). "
            "Repeat flag for multiple. Env: PASKAL_CHECKPOINTS (comma-separated paths)."
        ),
    )
    p.add_argument(
        "--audio-tower-config",
        type=Path,
        default=None,
        help="Hugging Face-style directory for Qwen2.5-Omni config (audio tower). Env: PASKAL_AUDIO_TOWER_CONFIG",
    )
    p.add_argument(
        "--audio-tower-weights",
        type=Path,
        default=None,
        help="State dict for the adapted audio tower weights (.pt). Env: PASKAL_AUDIO_TOWER_WEIGHTS",
    )
    p.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=None,
        help="JSONL with fields path, decision, id, ... Env: PASKAL_DATASET_JSONL",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Results root (per-checkpoint subfolders created here). Env: PASKAL_OUTPUT_DIR",
    )
    p.add_argument(
        "--semantic-standard-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL with standard_answers for semantic reranking. Env: PASKAL_SEMANTIC_STANDARD_JSONL",
    )
    p.add_argument(
        "--semantic-model-dir",
        type=Path,
        default=None,
        help="Optional local reranker model dir (e.g. BGE reranker). Env: PASKAL_SEMANTIC_MODEL_DIR",
    )
    p.add_argument(
        "--semantic-threshold",
        type=float,
        default=float(_env_str("PASKAL_SEMANTIC_THRESHOLD") or 0.5),
        help="Similarity threshold for counting RESPOND as correct after reranking. Env: PASKAL_SEMANTIC_THRESHOLD",
    )
    p.add_argument(
        "--max-jobs-per-gpu",
        type=int,
        default=int(_env_str("PASKAL_MAX_JOBS_PER_GPU") or 2),
        help="Worker tasks per visible GPU when running multiple checkpoints. Env: PASKAL_MAX_JOBS_PER_GPU",
    )

    args = p.parse_args(argv)

    def need(path_arg: Optional[Path], env_key: str, human_name: str) -> Path:
        raw = path_arg if path_arg is not None else None
        if raw is None:
            env_val = _env_str(env_key)
            if env_val:
                raw = Path(env_val)
        if raw is None:
            p.error(f"{human_name} is required (CLI or {env_key}).")
        out = raw.expanduser().resolve()
        return out

    tokenizer_dir = need(args.tokenizer_dir, "PASKAL_TOKENIZER_DIR", "--tokenizer-dir")
    audio_tower_config = need(args.audio_tower_config, "PASKAL_AUDIO_TOWER_CONFIG", "--audio-tower-config")
    audio_tower_weights = need(args.audio_tower_weights, "PASKAL_AUDIO_TOWER_WEIGHTS", "--audio-tower-weights")
    dataset_jsonl = need(args.dataset_jsonl, "PASKAL_DATASET_JSONL", "--dataset-jsonl")
    output_dir = need(args.output_dir, "PASKAL_OUTPUT_DIR", "--output-dir")

    ck_raw: list[Path] = []
    if args.checkpoints:
        ck_raw.extend(args.checkpoints)
    else:
        env_ck = _env_str("PASKAL_CHECKPOINTS")
        if env_ck:
            for part in env_ck.replace(";", ",").split(","):
                s = part.strip()
                if s:
                    ck_raw.append(Path(s))
    if not ck_raw:
        p.error(
            "At least one checkpoint is required: repeat --checkpoint PATH "
            "or set PASKAL_CHECKPOINTS (comma-separated paths)."
        )
    checkpoint_paths_resolved: list[Path] = []
    for cp in ck_raw:
        rp = cp.expanduser().resolve()
        if not rp.is_file():
            p.error(f"Checkpoint is not a file: {rp}")
        checkpoint_paths_resolved.append(rp)

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_file is not None:
        system_prompt = args.system_prompt_file.expanduser().read_text(encoding="utf-8").strip()
        if not system_prompt:
            p.error("--system-prompt-file is empty.")

    sem_std = args.semantic_standard_jsonl
    if sem_std is None and _env_str("PASKAL_SEMANTIC_STANDARD_JSONL"):
        sem_std = Path(_env_str("PASKAL_SEMANTIC_STANDARD_JSONL"))
    if sem_std is not None:
        sem_std = sem_std.expanduser().resolve()

    sem_model = args.semantic_model_dir
    if sem_model is None and _env_str("PASKAL_SEMANTIC_MODEL_DIR"):
        sem_model = Path(_env_str("PASKAL_SEMANTIC_MODEL_DIR"))
    if sem_model is not None:
        sem_model = sem_model.expanduser().resolve()

    return OfflinePaskalConfig(
        tokenizer_dir=tokenizer_dir,
        system_prompt=system_prompt,
        checkpoint_paths=tuple(checkpoint_paths_resolved),
        audio_tower_config=audio_tower_config,
        audio_tower_weights=audio_tower_weights,
        dataset_jsonl=dataset_jsonl,
        output_dir=output_dir,
        semantic_standard_file=sem_std,
        semantic_model_path=sem_model,
        semantic_threshold=float(args.semantic_threshold),
        max_jobs_per_gpu=int(args.max_jobs_per_gpu),
    )


class SuppressStderr:
    def __enter__(self):
        self.old_stderr = sys.stderr
        sys.stderr = StringIO()
        return self

    def __exit__(self, *args):
        sys.stderr = self.old_stderr


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_folder_name(path: str) -> str:
    """Subset label from paths containing ``.../sound_v3/<Domain>/<Subcategory>/...``."""
    match = re.search(r"/sound_v3/([^/]+/[^/]+)/", path)
    if match:
        return match.group(1)
    return "Unknown"


def extract_domain_name(path: str) -> str:
    """Top-level domain under ``sound_v3``."""
    match = re.search(r"/sound_v3/([^/]+)/", path)
    if match:
        return match.group(1)
    return "Unknown"


def infer_decision_from_reply(reply_text: str) -> str:
    return "RESPOND" if (reply_text and reply_text.strip()) else "IGNORE"


def load_jsonl(file_path: str | Path) -> list:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_standard_map(standard_data: list) -> dict:
    standard_map = {}
    for item in standard_data:
        item_id = item.get("id")
        answers = item.get("standard_answers", [])
        if item_id and isinstance(answers, list) and answers:
            standard_map[item_id] = answers
    return standard_map


def load_reranker(model_path: str | Path):
    try:
        from FlagEmbedding import FlagReranker
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "FlagEmbedding is required for semantic scoring. Install with: pip install FlagEmbedding"
        ) from e

    use_fp16 = torch.cuda.is_available()
    try:
        reranker = FlagReranker(str(model_path), use_fp16=use_fp16)
        print(f"Semantic reranker loaded (use_fp16={use_fp16}).")
        return reranker
    except Exception as exc:
        print(f"Semantic reranker failed to load ({exc}); retrying with use_fp16=False.")
        if use_fp16:
            reranker = FlagReranker(str(model_path), use_fp16=False)
            print("Semantic reranker loaded (use_fp16=False).")
            return reranker
        raise


def calculate_similarities_batch(reranker, query, documents, normalize: bool = True):
    pairs = [[query, doc] for doc in documents]
    scores = reranker.compute_score(pairs, normalize=normalize)
    return [float(score) for score in scores]


def load_all_data(dataset_path: Path) -> list:
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    print(f"\nLoaded dataset: {len(all_data)} rows.")
    return all_data


def get_input_ids_A_T(tokenizer, system_prompt: str) -> list:
    system_ids = tokenizer.encode(system_prompt).cpu().tolist()
    return [ONLINE, ENGLISH, SYSTEM, TEXT_BEGIN] + system_ids + [TEXT_END]


def checkpoint_output_stem(ckpt_path: str) -> str:
    """Output folder name: parent dir for lit_model.pth else statedict stem."""
    base = os.path.basename(ckpt_path)
    stem, _ = os.path.splitext(base)
    if stem == "lit_model":
        parent = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
        if parent:
            return parent
    return stem


def initialize_model(cfg: OfflinePaskalConfig, ckpt_path: str, cuda_device: int = 0):
    """``cuda_device`` is the logical index inside the current process (respect ``CUDA_VISIBLE_DEVICES``)."""
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)
        fabric = L.Fabric(
            accelerator="cuda",
            devices=[cuda_device],
            num_nodes=1,
            strategy="auto",
            precision=get_default_supported_precision(training=False),
            loggers="tensorboard",
        )
        map_dev = f"cuda:{cuda_device}"
    else:
        fabric = L.Fabric(
            devices=1,
            num_nodes=1,
            strategy="auto",
            precision=get_default_supported_precision(training=False),
            loggers="tensorboard",
        )
        map_dev = "cpu"

    set_seed(1337)

    model_yaml = cfg.tokenizer_dir / "model_config.yaml"
    config = Config.from_file(model_yaml)
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
    model = fabric.setup(model)

    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint not found at {ckpt_path}")
    load_checkpoint(fabric, model, ckpt_path, strict=True)
    tokenizer = Tokenizer(cfg.tokenizer_dir)

    qwen_omni_config = AutoConfig.from_pretrained(str(cfg.audio_tower_config))
    audio_encoder = Qwen2_5OmniForConditionalGeneration._from_config(
        qwen_omni_config
    ).thinker.audio_tower
    audio_encoder.load_state_dict(
        torch.load(str(cfg.audio_tower_weights), map_location=map_dev)
    )
    audio_encoder.to(map_dev).requires_grad_(False).eval()

    return fabric, model, tokenizer, audio_encoder


def process_single_round_audio(
    fabric,
    model,
    tokenizer,
    audio_encoder,
    audio_path: str,
    system_prompt: str,
):
    try:
        with fabric.init_tensor():
            model.set_kv_cache(batch_size=1)
        model.eval()

        input_ids = torch.LongTensor(get_input_ids_A_T(tokenizer, system_prompt)).to(model.device)

        with torch.inference_mode(), SuppressStderr():
            output = generate_fn_PASK(
                model,
                audio_encoder,
                tokenizer,
                input_ids,
                max_returned_tokens=4096,
                stop_token=TEXT_END,
                keep_silence_token=KEEP_SILENCE,
                break_silence_token=TEXT_BEGIN,
                audio_begin_token=AUDIO_BEGIN,
                assistant_token=ASSISTANT,
                pad_token=PAD,
                conversation_round=1,
                audio_path=audio_path,
            )

            decoded_outputs = []
            for o in output:
                decoded_text = tokenizer.decode(torch.tensor(o[2:-1]))
                if decoded_text:
                    decoded_outputs.append(decoded_text)

        reply = " ".join(decoded_outputs).strip()
        model.clear_kv_cache()
        return reply
    except Exception as e:
        model.clear_kv_cache()
        return f"Error: {str(e)}"


def test_single_audio(data_item, fabric, model, tokenizer, audio_encoder, system_prompt: str):
    audio_path = data_item["path"]
    ground_truth = data_item.get("decision")
    data_id = data_item.get("id", "")
    folder_name = extract_folder_name(audio_path)
    domain_name = extract_domain_name(audio_path)

    if not os.path.exists(audio_path):
        return {
            "id": data_id,
            "path": audio_path,
            "folder": folder_name,
            "domain": domain_name,
            "decision": "IGNORE",
            "reply": "Error: File not found",
            "ground_truth": ground_truth,
            "is_correct": False,
        }

    reply = process_single_round_audio(
        fabric, model, tokenizer, audio_encoder, audio_path, system_prompt
    )
    if reply.startswith("Error:"):
        final_decision = "IGNORE"
        final_reply = reply
        is_correct = False
    else:
        final_decision = infer_decision_from_reply(reply)
        final_reply = reply
        is_correct = (
            final_decision.upper() == ground_truth.upper() if ground_truth else False
        )

    return {
        "id": data_id,
        "path": audio_path,
        "folder": folder_name,
        "domain": domain_name,
        "decision": final_decision,
        "reply": final_reply,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
    }


def run_single_checkpoint(
    cfg: OfflinePaskalConfig, ckpt_path: str, test_data: list, cuda_device: int = 0
):
    ckpt_stem = checkpoint_output_stem(ckpt_path)
    ckpt_output_dir = os.path.join(str(cfg.output_dir), ckpt_stem)
    os.makedirs(ckpt_output_dir, exist_ok=True)

    output_file = os.path.join(ckpt_output_dir, "results.jsonl")
    stats_output_path = os.path.join(ckpt_output_dir, "stats.json")

    if os.path.exists(output_file):
        os.remove(output_file)

    print(f"\n===== Evaluating checkpoint: {ckpt_stem} =====")
    print("Loading model...")
    fabric, model, tokenizer, audio_encoder = initialize_model(
        cfg, ckpt_path, cuda_device=cuda_device
    )
    print("Model loaded.")

    results = []
    with open(output_file, "a", encoding="utf-8", buffering=1) as result_writer:
        for idx, data_item in enumerate(test_data, 1):
            print(f"\n[{idx}/{len(test_data)}]", end=" ")
            result = test_single_audio(
                data_item,
                fabric,
                model,
                tokenizer,
                audio_encoder,
                cfg.system_prompt,
            )
            result_writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            result_writer.flush()
            os.fsync(result_writer.fileno())
            results.append(result)

    def _is_valid_item(item):
        gt = (item.get("ground_truth") or "").upper()
        dec = (item.get("decision") or "").upper()
        reply = (item.get("reply") or "").strip()
        if gt not in {"RESPOND", "IGNORE"}:
            return False
        if dec not in {"RESPOND", "IGNORE"}:
            return False
        if reply.startswith("Error:"):
            return False
        return True

    valid_results = [r for r in results if _is_valid_item(r)]

    correct_count_before = sum(1 for r in valid_results if r.get("is_correct") is True)
    overall_accuracy_before = (
        correct_count_before / len(valid_results) if valid_results else 0
    )
    print(
        f"\n{ckpt_stem} done (before semantic filter): "
        f"{correct_count_before}/{len(valid_results)} = {overall_accuracy_before:.2%}"
    )

    semantic_available = False
    standard_map = {}
    reranker = None
    semantic_error = ""
    try:
        sem_file = cfg.semantic_standard_file
        sem_model = cfg.semantic_model_path
        if sem_file and sem_model and sem_file.is_file() and sem_model.exists():
            standard_data = load_jsonl(sem_file)
            standard_map = build_standard_map(standard_data)
            reranker = load_reranker(sem_model)
            semantic_available = True
        else:
            semantic_error = "Semantic standard file or reranker path missing; skipping semantic metrics."
            print(semantic_error)
    except Exception as e:
        semantic_error = str(e)
        print(f"Semantic pipeline failed ({semantic_error}); skipping semantic metrics.")
        semantic_available = False

    before_folder_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    after_folder_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    before_domain_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    after_domain_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    gt_respond_total = 0
    gt_respond_before_correct = 0
    gt_respond_after_correct = 0

    thr = cfg.semantic_threshold

    if semantic_available:
        for item in valid_results:
            folder = item.get("folder", "Unknown")
            domain = item.get("domain") or extract_domain_name(item.get("path", ""))
            gt = item.get("ground_truth")
            dec = item.get("decision")
            reply_text = (item.get("reply") or "").strip()
            item_id = item.get("id", "")

            if gt != dec:
                before_correct = False
                after_correct = False
            elif gt == "IGNORE" and dec == "IGNORE":
                before_correct = True
                after_correct = True
            else:
                before_correct = True
                after_correct = False

            before_folder_stats[folder]["total"] += 1
            if before_correct:
                before_folder_stats[folder]["correct"] += 1

            before_domain_stats[domain]["total"] += 1
            if before_correct:
                before_domain_stats[domain]["correct"] += 1

            if (gt or "").upper() == "RESPOND":
                gt_respond_total += 1
                if before_correct:
                    gt_respond_before_correct += 1

            if gt == "RESPOND" and dec == "RESPOND":
                standard_answers = standard_map.get(item_id, [])
                if standard_answers and reply_text:
                    sims = calculate_similarities_batch(
                        reranker, reply_text, standard_answers, normalize=True
                    )
                    max_sim = max(sims) if sims else 0.0
                    if max_sim > thr:
                        after_correct = True

            after_folder_stats[folder]["total"] += 1
            if after_correct:
                after_folder_stats[folder]["correct"] += 1

            after_domain_stats[domain]["total"] += 1
            if after_correct:
                after_domain_stats[domain]["correct"] += 1

            if (gt or "").upper() == "RESPOND" and after_correct:
                gt_respond_after_correct += 1
    else:
        for item in valid_results:
            folder = item.get("folder", "Unknown")
            domain = item.get("domain") or extract_domain_name(item.get("path", ""))
            is_correct = bool(item.get("is_correct", False))
            gt = (item.get("ground_truth") or "").upper()
            before_folder_stats[folder]["total"] += 1
            after_folder_stats[folder]["total"] += 1
            if is_correct:
                before_folder_stats[folder]["correct"] += 1
                after_folder_stats[folder]["correct"] += 1
            before_domain_stats[domain]["total"] += 1
            after_domain_stats[domain]["total"] += 1
            if is_correct:
                before_domain_stats[domain]["correct"] += 1
                after_domain_stats[domain]["correct"] += 1
            if gt == "RESPOND":
                gt_respond_total += 1
                if is_correct:
                    gt_respond_before_correct += 1
                    gt_respond_after_correct += 1

    def _build_folder_report(folder_stats):
        report = {}
        total_all = 0
        correct_all = 0
        for folder_name in sorted(folder_stats.keys()):
            stats = folder_stats[folder_name]
            total = stats["total"]
            correct = stats["correct"]
            acc = correct / total if total > 0 else 0.0
            total_all += total
            correct_all += correct
            report[folder_name] = {
                "total_samples": total,
                "correct_samples": correct,
                "accuracy": acc,
            }
        overall = (correct_all / total_all) if total_all > 0 else 0.0
        return report, overall, {"total_samples": total_all, "correct_samples": correct_all}

    by_folder_before, overall_by_folder_before, overall_counts_before = _build_folder_report(
        before_folder_stats
    )
    by_folder_after, overall_by_folder_after, overall_counts_after = _build_folder_report(
        after_folder_stats
    )
    by_domain_before, _, _ = _build_folder_report(before_domain_stats)
    by_domain_after, _, _ = _build_folder_report(after_domain_stats)
    gt_respond_before_accuracy = (
        (gt_respond_before_correct / gt_respond_total) if gt_respond_total > 0 else 0.0
    )
    gt_respond_after_accuracy = (
        (gt_respond_after_correct / gt_respond_total) if gt_respond_total > 0 else 0.0
    )

    stats_report = {
        "checkpoint": ckpt_path,
        "semantic_config": {
            "standard_file": str(cfg.semantic_standard_file) if cfg.semantic_standard_file else None,
            "model_path": str(cfg.semantic_model_path) if cfg.semantic_model_path else None,
            "threshold": thr,
            "semantic_available": semantic_available,
            "semantic_error": semantic_error,
        },
        "before_semantic": {
            "overall": {
                "total_samples": overall_counts_before["total_samples"],
                "correct_samples": overall_counts_before["correct_samples"],
                "accuracy": overall_by_folder_before,
            },
            "ground_truth_respond_accuracy": gt_respond_before_accuracy,
            "by_folder": by_folder_before,
            "by_domain": by_domain_before,
        },
        "after_semantic": {
            "overall": {
                "total_samples": overall_counts_after["total_samples"],
                "correct_samples": overall_counts_after["correct_samples"],
                "accuracy": overall_by_folder_after,
            },
            "ground_truth_respond_accuracy": gt_respond_after_accuracy,
            "by_folder": by_folder_after,
            "by_domain": by_domain_after,
        },
    }
    with open(stats_output_path, "w", encoding="utf-8") as f:
        json.dump(stats_report, f, ensure_ascii=False, indent=2)

    print(f"Wrote results: {output_file}")
    print(f"Wrote stats: {stats_output_path}")

    del model, audio_encoder, tokenizer, fabric
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "checkpoint": ckpt_path,
        "accuracy_before_semantic_valid": stats_report["before_semantic"]["overall"]["accuracy"],
        "accuracy_after_semantic_valid": stats_report["after_semantic"]["overall"]["accuracy"],
    }


def split_checkpoints_for_workers(checkpoint_paths, num_gpus: int, jobs_per_gpu: int):
    total_workers = max(1, num_gpus * jobs_per_gpu)
    worker_specs = []
    for worker_idx in range(total_workers):
        gpu_id = worker_idx % max(1, num_gpus)
        worker_specs.append({"worker_idx": worker_idx, "gpu_id": gpu_id, "checkpoints": []})

    for idx, ckpt_path in enumerate(checkpoint_paths):
        worker_specs[idx % total_workers]["checkpoints"].append(ckpt_path)

    return [w for w in worker_specs if w["checkpoints"]]


def run_worker(worker_idx: int, gpu_id: int, checkpoints: list, test_data: list, cfg: OfflinePaskalConfig):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    vis = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    vis_hint = f", CUDA_VISIBLE_DEVICES={vis}" if vis else " (unmasked; gpu_id is logical index)"
    worker_results = []
    print(
        f"[worker-{worker_idx}] started; logical GPU={gpu_id}{vis_hint}; "
        f"checkpoints={len(checkpoints)}"
    )
    for ckpt_path in checkpoints:
        worker_results.append(run_single_checkpoint(cfg, ckpt_path, test_data, cuda_device=gpu_id))
    print(f"[worker-{worker_idx}] done.")
    return worker_results


def main(argv: Optional[list[str]] = None) -> None:
    cfg = parse_args(argv)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Running full dataset (no sampling).")
    test_data = load_all_data(cfg.dataset_jsonl)
    if not test_data:
        print("Dataset is empty or unreadable.")
        return

    checkpoint_paths = [str(p) for p in cfg.checkpoint_paths]

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        print("No CUDA GPUs detected; running sequentially on CPU.")
        summary = [
            run_single_checkpoint(cfg, ckpt_path, test_data, cuda_device=0)
            for ckpt_path in checkpoint_paths
        ]
    else:
        print(
            f"\nDetected {num_gpus} GPU(s); up to {cfg.max_jobs_per_gpu} concurrent task(s) per GPU; "
            f"{len(checkpoint_paths)} checkpoint(s)."
        )
        worker_specs = split_checkpoints_for_workers(
            checkpoint_paths, num_gpus=num_gpus, jobs_per_gpu=cfg.max_jobs_per_gpu
        )
        print(f"Worker processes: {len(worker_specs)}")

        summary = []
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(worker_specs), mp_context=ctx) as executor:
            futures = [
                executor.submit(
                    run_worker,
                    spec["worker_idx"],
                    spec["gpu_id"],
                    spec["checkpoints"],
                    test_data,
                    cfg,
                )
                for spec in worker_specs
            ]
            for future in as_completed(futures):
                summary.extend(future.result())

    summary_path = os.path.join(str(cfg.output_dir), SUMMARY_JSON_NAME)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nAll checkpoints finished. Summary: {summary_path}")


if __name__ == "__main__":
    main()
