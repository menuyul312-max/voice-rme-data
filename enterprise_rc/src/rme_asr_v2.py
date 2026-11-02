#!/usr/bin/env python3
"""
Voice-to-RME ASR v2: data curation, training recipe and robust decoding for Whisper large-v3-turbo.

What changed vs v1 (whisper_turbo_rme_enterprise_a100_adapter: LoRA r=32 on q_proj/v_proj, 3 epochs) and why:
  * split by VIDEO into train/dev/test. In v1 every test video was also in train, so generalisation to a new
    doctor/room/microphone was never measured (audit finding #2).
  * label curation. The references are YouTube auto-captions. Dropped from training: clips >30 s (Whisper only
    sees 30 s, so the tail of the label teaches it to invent text), implausibly slow labels (<0.6 words/s:
    caption missing most of the speech, teaches deletions), implausibly fast labels and looping captions
    ("hai hai hai hai").
  * LoRA on every attention and MLP projection of encoder and decoder (v1: q_proj/v_proj only, ~0.4% params).
  * SpecAugment + waveform augmentation (gain, noise at random SNR, speed perturbation) against overfitting to
    the recording conditions of a few hundred videos.
  * checkpoint selection by WER on unseen dev videos (v1 had no WER-based selection).
  * decoding: anti-loop guard plus targeted re-decode of degenerate outputs (see benchmark_real.degenerate_reason).
  * export: merged fp16 weights with use_cache=True and Indonesian/transcribe generation defaults.
Pure-python parts are unit-tested on CPU (test_rme_asr_v2.py); GPU parts are exercised in Colab.
"""

import collections
import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmark_real import SAMPLE_RATE, collapse_repetitions, normalize, word_error_counts

BASE_MODEL_ID = "openai/whisper-large-v3-turbo"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
MAX_AUDIO_SEC = 30.0
MAX_LABEL_TOKENS = 448


# ---------------------------------------------------------------------------
# Video-level split
# ---------------------------------------------------------------------------

def _rank(seed: int, video_id: str) -> str:
    return hashlib.md5(f"{seed}:{video_id}".encode()).hexdigest()


def split_videos(rows: Sequence[Dict[str, Any]], dev_frac: float = 0.05, test_frac: float = 0.10,
                 seed: int = 13) -> Dict[str, str]:
    """video_id -> "train" | "dev" | "test". Whole videos only, so no speaker/room leaks across splits.
    Stratified by each video's majority level; fractions are measured in clips, not videos, so one very long
    video cannot silently become most of the test set. Deterministic for a given seed and manifest."""
    clips = collections.Counter(r["video_id"] for r in rows)
    levels = collections.defaultdict(collections.Counter)
    for r in rows:
        levels[r["video_id"]][r["level"]] += 1
    by_level = collections.defaultdict(list)
    for v, c in levels.items():
        by_level[c.most_common(1)[0][0]].append(v)
    out = {}
    for vids in by_level.values():
        vids.sort(key=lambda v: _rank(seed, v))
        total = sum(clips[v] for v in vids)
        test_n = dev_n = 0
        for v in vids:
            if test_n < test_frac * total:
                out[v], test_n = "test", test_n + clips[v]
            elif dev_n < dev_frac * total:
                out[v], dev_n = "dev", dev_n + clips[v]
            else:
                out[v] = "train"
    return out


# ---------------------------------------------------------------------------
# Label curation
# ---------------------------------------------------------------------------

def label_issue(row: Dict[str, Any], min_dur: float = 1.0, max_dur: float = MAX_AUDIO_SEC,
                min_wps: float = 0.6, max_wps: float = 4.5, min_words: int = 2) -> Optional[str]:
    """Reason a training clip should be dropped, or None if it looks usable."""
    words = normalize(row["ref"]).split()
    dur = float(row["dur"])
    if len(words) < min_words:
        return "TOO_FEW_WORDS"
    if dur > max_dur:
        return "TOO_LONG"
    if dur < min_dur:
        return "TOO_SHORT"
    wps = len(words) / dur
    if wps < min_wps:
        return "SLOW_LABEL"
    if wps > max_wps:
        return "FAST_LABEL"
    if len(collapse_repetitions(" ".join(words)).split()) < 0.85 * len(words):
        return "LOOPING_LABEL"
    return None


def curate(rows: Sequence[Dict[str, Any]], **kw) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Split rows into kept clips and a Counter of drop reasons."""
    kept, dropped = [], collections.Counter()
    for r in rows:
        why = label_issue(r, **kw)
        if why:
            dropped[why] += 1
        else:
            kept.append(r)
    return kept, dict(dropped)


# ---------------------------------------------------------------------------
# Waveform augmentation (numpy only, runs inside DataLoader workers)
# ---------------------------------------------------------------------------

@dataclass
class WaveAugment:
    p_speed: float = 0.3
    speed_range: tuple = (0.9, 1.1)
    p_noise: float = 0.4
    snr_db: tuple = (10.0, 30.0)
    p_gain: float = 0.5
    gain_db: tuple = (-6.0, 6.0)

    def __call__(self, audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32)
        if len(x) == 0:
            return x
        if rng.random() < self.p_speed:
            speed = rng.uniform(*self.speed_range)
            n = int(round(len(x) / speed))
            if n / SAMPLE_RATE <= MAX_AUDIO_SEC:  # never push a clip past Whisper's window
                x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        if rng.random() < self.p_noise:
            white = rng.standard_normal(len(x)).astype(np.float32)
            brown = np.cumsum(white)
            brown -= np.linspace(brown[0], brown[-1], len(brown))  # remove drift
            noise = white if rng.random() < 0.5 else brown
            sig_p = float(np.mean(x ** 2)) + 1e-10
            noise_p = float(np.mean(noise ** 2)) + 1e-10
            snr = rng.uniform(*self.snr_db)
            x = x + noise * np.sqrt(sig_p / (noise_p * 10 ** (snr / 10)))
        if rng.random() < self.p_gain:
            x = x * 10 ** (rng.uniform(*self.gain_db) / 20)
        return np.clip(x, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Training data plumbing
# ---------------------------------------------------------------------------

def load_audio(path: str) -> np.ndarray:
    import librosa
    return librosa.load(path, sr=SAMPLE_RATE)[0]


class ClipDataset:
    """List-backed dataset of {path, text, aug}; audio is decoded lazily in the collator (no 30 GB feature cache)."""

    def __init__(self, rows: Sequence[Dict[str, Any]], audio_dir: str, augment: bool):
        self.items = [{"path": os.path.join(audio_dir, r["audio_file"]), "text": r["ref"], "aug": augment}
                      for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@dataclass
class OnTheFlyCollator:
    """Decode audio, augment train items, extract log-mel features and build masked labels."""
    processor: Any
    augment: Optional[WaveAugment] = None
    seed: int = 0
    _rng: Optional[np.random.Generator] = field(default=None, repr=False)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        if self._rng is None:  # one stream per DataLoader worker
            self._rng = np.random.default_rng(int(torch.initial_seed()) % (2 ** 63))
        audios = []
        for f in features:
            a = load_audio(f["path"])[: int(MAX_AUDIO_SEC * SAMPLE_RATE)]
            if f.get("aug") and self.augment is not None:
                a = self.augment(a, self._rng)
            audios.append(a)
        batch = self.processor.feature_extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        tok = self.processor.tokenizer
        ids = [tok(f["text"]).input_ids for f in features]
        if any(len(x) > MAX_LABEL_TOKENS for x in ids):
            raise ValueError("Label exceeds decoder window; curate before training, never silently truncate")
        width = max(len(x) for x in ids)
        labels = torch.full((len(ids), width), -100, dtype=torch.long)
        for i, x in enumerate(ids):
            labels[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        if (labels[:, 0] == sot).all():  # the model prepends decoder_start_token_id itself
            labels = labels[:, 1:]
        return {"input_features": batch["input_features"], "labels": labels}


def corpus_wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Corpus WER (%) with the same normalisation as the benchmark."""
    if len(refs) != len(hyps):
        raise ValueError("reference/hypothesis length mismatch")
    err = n = 0
    for r, h in zip(refs, hyps):
        rw, hw = normalize(r).split(), normalize(h).split()
        s, d, i = word_error_counts(rw, hw)
        err, n = err + s + d + i, n + max(len(rw), 1)
    return 100.0 * err / max(n, 1)


# ---------------------------------------------------------------------------
# Model construction (GPU side)
# ---------------------------------------------------------------------------

def from_pretrained(cls, path: str, dtype=None, **kw):
    """transformers renamed torch_dtype -> dtype; accept both versions."""
    try:
        return cls.from_pretrained(path, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(path, torch_dtype=dtype, **kw)


def set_generation_defaults(model) -> None:
    model.config.use_cache = True
    gc = model.generation_config
    gc.language, gc.task = "indonesian", "transcribe"
    gc.forced_decoder_ids = None
    if hasattr(gc, "use_cache"):
        gc.use_cache = True


def build_lora_model(model_id: str = BASE_MODEL_ID, r: int = 32, alpha: int = 64, dropout: float = 0.05,
                     spec_augment: bool = True, dtype=None):
    """Whisper + LoRA on all attention/MLP projections of encoder and decoder, with SpecAugment on."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration
    model = from_pretrained(WhisperForConditionalGeneration, model_id, dtype=dtype or torch.float32)
    if spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob, model.config.mask_time_length = 0.05, 10
        model.config.mask_feature_prob, model.config.mask_feature_length = 0.05, 10
    set_generation_defaults(model)
    model.config.use_cache = False  # incompatible with gradient checkpointing; re-enabled at export
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                             target_modules=LORA_TARGETS, bias="none"))
    # conv1 is frozen, so its output has no grad and checkpointed blocks would drop the LoRA gradients
    model.base_model.model.model.encoder.conv1.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    return model


def make_compute_metrics(tokenizer):
    def compute(pred):
        p = np.where(pred.predictions < 0, tokenizer.pad_token_id, pred.predictions)
        l = np.where(pred.label_ids < 0, tokenizer.pad_token_id, pred.label_ids)
        return {"wer": corpus_wer(tokenizer.batch_decode(l, skip_special_tokens=True),
                                  tokenizer.batch_decode(p, skip_special_tokens=True))}
    return compute


def build_trainer(model, processor, train_ds, dev_ds, out_dir: str, *, epochs: float = 3, lr: float = 1e-4,
                  batch: int = 16, accum: int = 2, eval_steps: int = 250, workers: int = 4, max_steps: int = -1,
                  augment: Optional[WaveAugment] = None, patience: int = 4, seed: int = 0, bf16: Optional[bool] = None):
    """Seq2SeqTrainer selecting the checkpoint with the lowest dev WER (unseen videos)."""
    import math
    import torch
    from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments
    cuda = torch.cuda.is_available()
    if bf16 is None:
        bf16 = cuda and torch.cuda.is_bf16_supported()
    total = max_steps if max_steps > 0 else math.ceil(len(train_ds) / (batch * accum)) * epochs
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum, learning_rate=lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * total)), num_train_epochs=epochs, max_steps=max_steps, weight_decay=0.01,
        bf16=bf16, fp16=cuda and not bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps,
        save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=225, generation_num_beams=1,
        logging_steps=25, report_to=["none"], remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=workers, dataloader_persistent_workers=workers > 0, seed=seed)
    return Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=OnTheFlyCollator(processor, augment if augment is not None else WaveAugment(), seed),
        compute_metrics=make_compute_metrics(processor.tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)])


def export_merged(peft_model, processor, out_dir: str) -> str:
    """Merge LoRA into the base weights, save fp16 with inference-ready generation defaults."""
    import torch
    model = peft_model.merge_and_unload()
    model.config.apply_spec_augment = False
    set_generation_defaults(model)
    model = model.to(torch.float16)
    model.save_pretrained(out_dir, safe_serialization=True)
    processor.save_pretrained(out_dir)
    return out_dir