# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""PASKAL offline generation (minimal subset for `offline_paskal.py`)."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Any, Dict, Iterator, List, Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import whisper
from scipy.io import wavfile as scipy_wavfile

from litgpt.model_pask import GPT
from litgpt.tokenizer import Tokenizer


def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    sorted_indices_to_remove[-1:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        0, sorted_indices, sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    logits = logits[0, -1]
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    if temperature > 0.0 or top_p > 0.0:
        if temperature > 0.0:
            logits = logits / temperature
        if top_p < 1.0:
            logits = sample_top_p(logits, top_p)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)


def sample_penalty(
    logits: torch.Tensor,
    past_tokens: Optional[torch.Tensor] = None,
    repetition_penalty: float = 1.0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    logits = logits[0, -1]

    if past_tokens is not None and repetition_penalty != 1.0:
        for token in set(past_tokens.tolist()):
            if logits[token] < 0:
                logits[token] *= repetition_penalty
            else:
                logits[token] /= repetition_penalty

    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)

    if temperature > 0.0 or top_p > 0.0:
        if temperature > 0.0:
            logits = logits / temperature
        if top_p < 1.0:
            logits = sample_top_p(logits, top_p)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return multinomial_num_samples_1(probs)

    return torch.argmax(logits, dim=-1, keepdim=True)


def next_token_LALM(
    model: GPT,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    audio_feat: Dict,
    input_pos_maxp1: Optional[torch.Tensor] = None,
    past_tokens=None,
    repetition_penalty: float = 1.0,
    get_audiofeat_in_forward: bool = False,
    audio_tokens_per_chunck=10,
    **sample_kwargs: Dict[str, Any],
) -> torch.Tensor:
    if audio_feat is None:
        logits = model(
            x,
            None,
            1,
            None,
            input_pos,
            input_pos_maxp1=input_pos_maxp1,
            get_audiofeat_in_forward=get_audiofeat_in_forward,
            audio_tokens_per_chunck=audio_tokens_per_chunck,
        )
    else:
        logits = model(
            x,
            None,
            1,
            audio_feat,
            input_pos,
            input_pos_maxp1=input_pos_maxp1,
            get_audiofeat_in_forward=get_audiofeat_in_forward,
            audio_tokens_per_chunck=audio_tokens_per_chunck,
        )

    if past_tokens is None:
        _next = sample(logits, **sample_kwargs).to(dtype=torch.int64)
    else:
        _next = sample_penalty(
            logits,
            past_tokens=torch.tensor(past_tokens),
            repetition_penalty=repetition_penalty,
            **sample_kwargs,
        ).to(dtype=torch.int64)
    return _next


def _audio_segment_buffer_dir() -> str:
    explicit = os.environ.get("PASKAL_AUDIO_BUFFER")
    if explicit:
        return explicit
    root = os.environ.get("PASKAL_AUDIO_BUFFER_ROOT", tempfile.gettempdir())
    return os.path.join(root, "litgpt_paskal_segments", str(os.getpid()))


BUFFERPATH = _audio_segment_buffer_dir()


def _write_audio_segment(
    segment_path: str, segment: np.ndarray, sr: int, max_retries: int = 3
) -> None:
    segment = np.asarray(segment, dtype=np.float32)
    segment = np.nan_to_num(segment, nan=0.0, posinf=1.0, neginf=-1.0)
    segment = np.clip(segment, -1.0, 1.0)

    for attempt in range(max_retries):
        try:
            sf.write(segment_path, segment, sr)
            return
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
            else:
                try:
                    segment_int16 = (segment * 32767).astype(np.int16)
                    scipy_wavfile.write(segment_path, sr, segment_int16)
                    return
                except Exception as fallback_e:
                    raise RuntimeError(
                        f"Failed to write {segment_path}: sf.error={e}, scipy.error={fallback_e}"
                    ) from fallback_e


def split_audio(audio_path, folder_path, audio_piece_idx, segment_duration_s=0.4):
    audio, sr = librosa.load(audio_path, sr=None)

    total_length_samples = len(audio)
    segment_duration_samples = int(segment_duration_s * sr)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    segment_count = 0
    for start_sample in range(0, total_length_samples, segment_duration_samples):
        end_sample = start_sample + segment_duration_samples
        segment = audio[start_sample:end_sample]

        segment_length_samples = len(segment)
        segment_length_s = segment_length_samples / sr

        if segment_length_samples == segment_duration_samples:
            segment_name = f"{audio_piece_idx + segment_count + 1}_full_{segment_duration_s}s.wav"
        else:
            segment_name = f"{audio_piece_idx + segment_count + 1}_{segment_length_s:.1f}s.wav"

        segment_path = os.path.join(folder_path, segment_name)
        _write_audio_segment(segment_path, segment, sr)

        segment_count += 1

    print(f"Split audio segments written to {folder_path}")


def delete_all_in_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}: {e}")
    else:
        print(f"Folder missing or not a directory: {folder_path}")


def load_audio_qwenomni_batch(audio_path):
    audio = whisper.load_audio(audio_path, sr=16000).tolist()

    audio = audio + [0] * 6400

    if len(audio) % 160 != 0:
        audio += [0] * (160 - len(audio) % 160)

    audio = np.array(audio, dtype=np.float32)
    mel = whisper.log_mel_spectrogram(audio, n_mels=128)
    len_feature = mel.shape[1]

    return mel, len_feature


def split_into_chunks(number, chunk_size):
    chunks = [chunk_size] * (number // chunk_size)
    remainder = number % chunk_size
    if remainder:
        chunks.append(remainder)
    return chunks


def count_lengths(input_lengths):
    if isinstance(input_lengths, list):
        return [(i - 1) // 2 + 1 for i in input_lengths]
    if isinstance(input_lengths, int):
        return (input_lengths - 1) // 2 + 1
    raise ValueError("input_lengths should be int or list")


def get_audio_feats(audiopath, audio_piece_idx, audio_encoder):
    folder_path = BUFFERPATH
    os.makedirs(folder_path, exist_ok=True)
    mel, len_feature = load_audio_qwenomni_batch(audiopath)
    split_audio(
        audio_path=audiopath,
        folder_path=folder_path,
        audio_piece_idx=audio_piece_idx,
        segment_duration_s=0.4,
    )

    len_feature_split = split_into_chunks(len_feature, 40)
    len_input_split = count_lengths(len_feature)

    audio_feats = []

    audio_encoder.to("cuda")
    with torch.no_grad():
        feat = audio_encoder(
            torch.tensor(mel).to("cuda"),
            torch.tensor(len_feature_split).to("cuda"),
            torch.tensor(len_input_split).to("cuda"),
        ).last_hidden_state

    len_feat = feat.shape[0]
    feat = feat[: len_feat - len_feat % 10]

    for i in range(0, len_feat - len_feat % 10, 10):
        audio_feats.append(feat[i : i + 10])

    return audio_feats


def generate_fn_PASK(
    model: GPT,
    audio_encoder: torch.nn.Module,
    tokenizer: Tokenizer,
    input_id: torch.Tensor,
    max_returned_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    stop_token: int = 151643,
    keep_silence_token: int = 151605,
    break_silence_token: int = 151644,
    audio_begin_token: int = 151647,
    assistant_token: int = 151602,
    audio_tokens_per_chunck: int = 10,
    pad_token: int = 151608,
    conversation_round: int = 1,
    audio_path: Optional[str] = None,
) -> Union[Iterator[torch.Tensor], List]:
    prompt_size = input_id.size(0)

    device = input_id.device

    tokens = []
    round_outputs = []
    token = input_id
    input_pos = torch.arange(0, prompt_size, device=device, dtype=torch.int64)

    if not any(m.__class__.__name__ == "ThunderModule" for m in model.modules()):
        input_pos_maxp1 = torch.tensor(prompt_size, device=device)
    else:
        input_pos_maxp1 = None

    delete_all_in_folder(folder_path=BUFFERPATH)
    audio_piece_idx = 0
    LISTENING = True
    multi_round = (conversation_round is not None and conversation_round > 1) or isinstance(
        audio_path, (list, tuple)
    )

    for conversation_idx in range(conversation_round):
        audio_feat_idx = -1
        round_start = len(tokens)

        if audio_path is None:
            current_audio_path = input(f"Enter your round {conversation_idx} audio path here: ")
        elif isinstance(audio_path, (list, tuple)):
            if conversation_idx >= len(audio_path):
                raise ValueError(
                    f"audio_path list length {len(audio_path)} < conversation_round {conversation_round}"
                )
            current_audio_path = audio_path[conversation_idx]
        else:
            current_audio_path = audio_path
        audio_feats = get_audio_feats(
            current_audio_path, audio_piece_idx, audio_encoder=audio_encoder
        )

        nums_audio_pieces = len(audio_feats)
        if audio_path is None:
            print(f"There are {nums_audio_pieces} audio pieces in this conversation.")

        for _current_idx in range(max_returned_tokens - len(input_pos)):
            if LISTENING:
                audio_feat_idx += 1
                if audio_feat_idx >= nums_audio_pieces:
                    break

                new_tokens = torch.LongTensor(
                    [audio_begin_token]
                    + [pad_token] * audio_tokens_per_chunck
                    + [assistant_token]
                ).to(device)
                token = torch.cat((token, new_tokens), dim=0)

                last_input_pos = input_pos[-1]
                _input_pos = torch.tensor(
                    [last_input_pos + i for i in range(1, 3 + audio_tokens_per_chunck)]
                ).to(device)
                input_pos = torch.cat((input_pos, _input_pos), dim=0)

                if input_pos_maxp1 is not None:
                    input_pos_maxp1.add_(2 + audio_tokens_per_chunck)

                if audio_path is None:
                    print(f"input_pos: {input_pos}")
                token = next_token_LALM(
                    model,
                    input_pos,
                    token.view(1, -1),
                    audio_feats[audio_feat_idx].to(device),
                    input_pos_maxp1=input_pos_maxp1,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    get_audiofeat_in_forward=False,
                    audio_tokens_per_chunck=audio_tokens_per_chunck,
                )

                int_token = token.item()
                if audio_path is None:
                    print(int_token)

                input_pos = input_pos[-1].unsqueeze(0).add_(1)

                if input_pos_maxp1 is not None:
                    input_pos_maxp1.add_(1)

                if int_token == break_silence_token:
                    LISTENING = False
                    if audio_path is None:
                        print("!!!!!!!!!!!!!! BREAKING SILENCE !!!!!!!!!!!!!!!")
                    tokens.append([int_token])
                elif int_token == keep_silence_token:
                    tokens.append([int_token])
                    continue
                else:
                    raise ValueError(f"Unexpected token {int_token} in generate_fn_PASK")

            else:
                if audio_path is None:
                    print(f"input_pos: {input_pos}")
                token = next_token_LALM(
                    model,
                    input_pos,
                    token.view(1, -1),
                    None,
                    input_pos_maxp1=input_pos_maxp1,
                )

                input_pos.add_(1)
                if input_pos_maxp1 is not None:
                    input_pos_maxp1.add_(1)

                int_token = token.item()
                tokens[-1].append(int_token)

                if int_token == stop_token:
                    LISTENING = True
                    if audio_path is None:
                        print("!!!!!!!!!!!!!! STARTING LISTENING !!!!!!!!!!!!!!!")
            if audio_path is None:
                print(tokens)
        if multi_round:
            round_outputs.append(tokens[round_start:])

        if audio_path is None:
            decoded_piece_idx = 0
            for o in tokens:
                decoded_piece_idx += 1
                decoded_text = tokenizer.decode(torch.tensor(o[2:-1]))
                if decoded_text == "":
                    continue
                print("===============================================================")
                print(f"Audio piece: {decoded_piece_idx}:")
                print(decoded_text)
                print("===============================================================")
    return round_outputs if multi_round else tokens
