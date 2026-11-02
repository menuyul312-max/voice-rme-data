#!/usr/bin/env python3
"""
Real benchmark toolkit for the Voice-to-RME stack.

Why this exists: the earlier benchmark used a chunk-level test split (every test video also
appears in train), 100-600 clips, and 41 hand-picked clinical terms. This toolkit instead:
  * tags every clip as train_clip / heldout_seen_video / heldout_unseen_video (leakage-aware)
  * drops heldout clips whose exact reference text occurs in train
  * scores thousands of clips with bootstrap 95% CIs, S/D/I breakdown, hallucination rate
  * measures clinical-term recall AND false-term rate over the whole ontology vocabulary
  * measures downstream SOAP extraction on multi-minute sessions (ASR-induced degradation,
    JSON validity, grounding, fabricated vitals) without needing gold SOAP annotations
Pure-python scoring functions are unit-tested on CPU (test_benchmark_real.py).
"""

import collections
import csv
import json
import os
import random
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SAMPLE_RATE = 16000
MAX_CLIP_SEC = 29.5


# ---------------------------------------------------------------------------
# Text normalisation + error counting
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, strip punctuation/hyphens, collapse whitespace (references carry no casing/punctuation)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()


INDONESIAN_CONVERSATIONAL_MAP = {
    r"\bnggak\b": "tidak",
    r"\bgak\b": "tidak",
    r"\bngga\b": "tidak",
    r"\bndak\b": "tidak",
    r"\btak\b": "tidak",
    r"\budah\b": "sudah",
    r"\bdah\b": "sudah",
    r"\bkalo\b": "kalau",
    r"\bklo\b": "kalau",
    r"\bpake\b": "pakai",
    r"\bgimana\b": "bagaimana",
    r"\bbagaimanakah\b": "bagaimana",
    r"\btau\b": "tahu",
    r"\baja\b": "saja",
    r"\bcuman\b": "cuma",
    r"\bmakasih\b": "terima kasih",
    r"\bya\b": "iya",
    r"\bbanget\b": "sekali",
    r"\btrus\b": "terus",
    r"\bknp\b": "kenapa",
    r"\bngantuk\b": "mengantuk",
    r"\blaper\b": "lapar",
    r"\bcapek\b": "lelah",
    # Fillers & hesitations
    r"\b(?:ee|ehm|eh|ah|uh|oh|nah|kan|lah)\b": "",
    # Clinical units & numbers
    r"(\d+)\s*mg\b": r"\1 miligram",
    r"(\d+)\s*ml\b": r"\1 mililiter",
    r"(\d+)\s*g\b": r"\1 gram",
    r"(\d+)\s*kg\b": r"\1 kilogram",
    r"(\d+)\s*x\s*(\d+)": r"\1 kali \2",
}


def normalize_clinical_conversational(text: str) -> str:
    """
    Standardizes spoken Indonesian conversational particles, contractions,
    and clinical measurements to evaluate true semantic accuracy.
    """
    t = normalize(text)
    for pat, rep in INDONESIAN_CONVERSATIONAL_MAP.items():
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t).strip()


def levenshtein(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def word_error_counts(ref_words: List[str], hyp_words: List[str]) -> Tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) via full DP + backtrace."""
    n, m = len(ref_words), len(hyp_words)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref_words[i - 1] != hyp_words[j - 1]))
    i, j, s, dele, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref_words[i - 1] != hyp_words[j - 1]):
            s += ref_words[i - 1] != hyp_words[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, dele, ins


def bootstrap_ci(errs: Sequence[int], ns: Sequence[int], B: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    """Corpus-level rate = sum(errs)/sum(ns), 95% CI by resampling utterances."""
    errs, ns = np.asarray(errs, dtype=float), np.asarray(ns, dtype=float)
    if len(errs) == 0 or ns.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(errs), size=(B, len(errs)))
    rates = errs[idx].sum(1) / np.maximum(ns[idx].sum(1), 1)
    return float(errs.sum() / ns.sum()), float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson interval (%) for a proportion k/n; honest about tiny n."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, centre - half) * 100, min(1.0, centre + half) * 100


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts (b: A-only wins, c: B-only wins)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def collapse_repetitions(text: str, max_ngram: int = 6, keep: int = 2) -> str:
    """Collapse a word n-gram repeated more than `keep` times in a row (decoder loops) down to `keep` copies."""
    w = (text or "").split()
    changed = True
    while changed:
        changed = False
        for n in range(1, max_ngram + 1):
            i = 0
            while i + n * (keep + 1) <= len(w):
                unit = w[i:i + n]
                reps = 1
                while w[i + reps * n:i + (reps + 1) * n] == unit:
                    reps += 1
                if reps > keep:
                    del w[i + keep * n:i + reps * n]
                    changed = True
                i += 1
    return " ".join(w)


def compression_ratio(text: str) -> float:
    """len(utf8) / len(zlib(utf8)); Whisper's own loop heuristic flags > 2.4."""
    import zlib
    b = (text or "").encode("utf-8")
    return len(b) / max(len(zlib.compress(b)), 1) if b else 0.0


def degenerate_reason(text: str, dur: float, max_ratio: float = 2.4, max_wps: float = 6.0) -> Optional[str]:
    """Why a hypothesis looks like a decoder failure (loop / runaway), judged without the reference."""
    words = (text or "").split()
    if not words:
        return None  # silence is a legitimate output
    if len(words) >= 12 and compression_ratio(text) > max_ratio:
        return "COMPRESSION"
    if len(words) >= 12 and len(words) / max(dur, 0.5) > max_wps:
        return "WORDS_PER_SEC"
    if len(words) >= 8 and len(collapse_repetitions(text).split()) < 0.7 * len(words):
        return "REPEATED_NGRAM"
    return None


def clip_error_arrays(items: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-clip (word errors, reference words) using strict normalisation."""
    errs, ns = [], []
    for it in items:
        rw, hw = normalize(it["ref"]).split(), normalize(it["hyp"] or "").split()
        s, d, i = word_error_counts(rw, hw)
        errs.append(s + d + i)
        ns.append(max(len(rw), 1))
    return np.asarray(errs, dtype=float), np.asarray(ns, dtype=float)


def paired_bootstrap(items_a: List[Dict[str, Any]], items_b: List[Dict[str, Any]], B: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Paired comparison of two systems on the SAME clips (items must be aligned by index).
    delta = WER_a - WER_b (pp). Negative => A better. p is the two-sided bootstrap p-value for delta == 0."""
    if len(items_a) != len(items_b):
        raise ValueError("paired_bootstrap needs aligned item lists")
    for a, b in zip(items_a, items_b):
        if a["ref"] != b["ref"] or a.get("chunk_id") != b.get("chunk_id"):
            raise ValueError("paired comparison has mismatched references or clip IDs")
    ea, na = clip_error_arrays(items_a)
    eb, nb = clip_error_arrays(items_b)
    if len(ea) == 0:
        return {"n": 0, "wer_a": float("nan"), "wer_b": float("nan"), "delta_pp": float("nan"),
                "delta_ci95": [float("nan")] * 2, "p_value": float("nan"), "a_better_clip_frac": float("nan")}
    rng = np.random.default_rng(seed)
    group_ids = [a.get("video_id", a.get("chunk_id", i)) for i, a in enumerate(items_a)]
    if any(a.get("video_id") != b.get("video_id") for a, b in zip(items_a, items_b)):
        raise ValueError("paired comparison has mismatched video IDs")
    groups = list(dict.fromkeys(group_ids))
    # Resample whole videos to respect correlation among clips from one recording.
    sums = np.asarray([[ea[jj].sum(), na[jj].sum(), eb[jj].sum(), nb[jj].sum()]
                       for g in groups for jj in [[i for i, x in enumerate(group_ids) if x == g]]])
    idx = rng.integers(0, len(groups), size=(B, len(groups)))
    draws = sums[idx].sum(1)
    deltas = (draws[:, 0] / draws[:, 1] - draws[:, 2] / draws[:, 3]) * 100
    p = 2 * min((deltas >= 0).mean(), (deltas <= 0).mean())
    wa, wb = ea / na, eb / nb
    return {"n_clusters": len(groups), "bootstrap_unit": "video" if all("video_id" in a for a in items_a) else "clip", "n": len(ea), "wer_a": float(ea.sum() / na.sum() * 100), "wer_b": float(eb.sum() / nb.sum() * 100),
            "delta_pp": float((ea.sum() / na.sum() - eb.sum() / nb.sum()) * 100),
            "delta_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "p_value": float(min(1.0, max(p, 1.0 / B))),
            "a_better_clip_frac": float((wa < wb).mean() * 100), "b_better_clip_frac": float((wb < wa).mean() * 100),
            "median_clip_wer_a": float(np.median(wa) * 100), "median_clip_wer_b": float(np.median(wb) * 100)}


def paired_term_test(items_a: List[Dict[str, Any]], items_b: List[Dict[str, Any]], terms: Sequence[str]) -> Dict[str, Any]:
    """Per-(clip, reference-term) hit/miss for A and B, then exact McNemar on discordant pairs."""
    both = a_only = b_only = neither = 0
    for ia, ib in zip(items_a, items_b):
        pr = f" {normalize(ia['ref'])} "
        ha, hb = f" {normalize(ia['hyp'] or '')} ", f" {normalize(ib['hyp'] or '')} "
        for t in terms:
            if _present(t, pr):
                x, y = _present(t, ha), _present(t, hb)
                both += x and y
                a_only += x and not y
                b_only += y and not x
                neither += (not x) and (not y)
    n = both + a_only + b_only + neither
    return {"term_occurrences": n, "both_hit": both, "a_only": a_only, "b_only": b_only, "neither": neither,
            "p_value": mcnemar_exact(a_only, b_only),
            "enough_data": n >= 100}



# ---------------------------------------------------------------------------
# Hallucination + clinical-term metrics
# ---------------------------------------------------------------------------

def hallucination_flag(ref: str, hyp: str, guardrail=None) -> Optional[str]:
    """Reason string if the hypothesis looks hallucinated relative to the reference, else None."""
    hn, rn = hyp.split(), ref.split()
    if not hn:
        return "EMPTY"
    if len(hn) >= 10 and len(hn) > 2.5 * max(len(rn), 1):
        return "LENGTH_RATIO"
    if guardrail is not None:
        bad, why = guardrail.is_hallucination(hyp)
        if bad and not why.startswith("EMPTY"):
            return why.split(" ")[0]
    return None


def build_term_list(ontology: Dict[str, Any], min_len: int = 4) -> List[str]:
    terms = set()
    for spec in ontology.values():
        for kind in ("diagnoses", "medications", "procedures"):
            for item in spec[kind]:
                for kw in item["keywords"]:
                    t = normalize(kw)
                    if len(t) >= min_len:
                        terms.add(t)
    return sorted(terms)


def _present(term: str, padded_text: str) -> bool:
    return f" {term} " in padded_text


def term_stats(ref: str, hyp: str, terms: Iterable[str]) -> Tuple[int, int, int]:
    """(tp, fn, fp): term in both / only in ref / only in hyp. Inputs must be normalized."""
    pr, ph = f" {ref} ", f" {hyp} "
    tp = fn = fp = 0
    for t in terms:
        a, b = _present(t, pr), _present(t, ph)
        tp += a and b
        fn += a and not b
        fp += b and not a
    return tp, fn, fp


def term_groups(ontology: Dict[str, Any], min_len: int = 4) -> Dict[str, str]:
    """normalised term -> 'specialty/kind' (e.g. 'mata/medications'); first occurrence wins."""
    out = {}
    for spec_name, spec in ontology.items():
        for kind in ("diagnoses", "medications", "procedures"):
            for item in spec[kind]:
                for kw in item["keywords"]:
                    t = normalize(kw)
                    if len(t) >= min_len:
                        out.setdefault(t, f"{spec_name}/{kind}")
    return out


def term_recall_by_group(items: List[Dict[str, Any]], groups: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Per ontology group: reference occurrences, hits, recall (%) and Wilson 95% CI."""
    tp, n = collections.Counter(), collections.Counter()
    for it in items:
        pr, ph = f" {normalize(it['ref'])} ", f" {normalize(it['hyp'] or '')} "
        for t, g in groups.items():
            if _present(t, pr):
                n[g] += 1
                tp[g] += _present(t, ph)
    return {g: {"n": n[g], "hit": tp[g], "recall": 100.0 * tp[g] / n[g], "ci95": list(wilson_ci(tp[g], n[g]))}
            for g in sorted(n)}


def score_items(items: List[Dict[str, Any]], terms: Sequence[str], guardrail=None, B: int = 1000) -> Dict[str, Any]:
    """items: dicts with ref, hyp, dur. Returns corpus metrics with CIs for both strict and clinical normalized."""
    w_err, w_n, c_err, c_n = [], [], [], []
    w_err_clin, w_n_clin, c_err_clin, c_n_clin = [], [], [], []
    S = D = I = 0
    halluc = collections.Counter()
    tp = fn = fp = 0
    hours = 0.0
    for it in items:
        r, h = normalize(it["ref"]), normalize(it["hyp"] or "")
        rw, hw = r.split(), h.split()
        s, d, i = word_error_counts(rw, hw)
        S, D, I = S + s, D + d, I + i
        w_err.append(s + d + i)
        w_n.append(max(len(rw), 1))
        c_err.append(levenshtein(r, h))
        c_n.append(max(len(r), 1))

        # Clinical conversational normalization
        rc = normalize_clinical_conversational(it["ref"])
        hc = normalize_clinical_conversational(it["hyp"] or "")
        rwc, hwc = rc.split(), hc.split()
        sc, dc, ic = word_error_counts(rwc, hwc)
        w_err_clin.append(sc + dc + ic)
        w_n_clin.append(max(len(rwc), 1))
        c_err_clin.append(levenshtein(rc, hc))
        c_n_clin.append(max(len(rc), 1))

        hours += it.get("dur", 0.0) / 3600
        flag = hallucination_flag(r, h, guardrail)
        if flag:
            halluc[flag] += 1
        a, b, c = term_stats(r, h, terms)
        tp, fn, fp = tp + a, fn + b, fp + c
    n = len(items)
    clip_wer = np.asarray(w_err, dtype=float) / np.maximum(np.asarray(w_n, dtype=float), 1)
    collapsed = [dict(it, hyp=collapse_repetitions(it["hyp"] or "")) for it in items]
    ce, cn = clip_error_arrays(collapsed)
    wer, wlo, whi = bootstrap_ci(w_err, w_n, B)
    cer, clo, chi = bootstrap_ci(c_err, c_n, B)
    wer_c, wlo_c, whi_c = bootstrap_ci(w_err_clin, w_n_clin, B)
    cer_c, clo_c, chi_c = bootstrap_ci(c_err_clin, c_n_clin, B)
    N = max(sum(w_n), 1)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "n_clips": n, "hours": round(hours, 3), "ref_words": int(sum(w_n)),
        "wer": wer * 100, "wer_ci95": [wlo * 100, whi * 100],
        "cer": cer * 100, "cer_ci95": [clo * 100, chi * 100],
        "wer_clinical": wer_c * 100, "wer_clinical_ci95": [wlo_c * 100, whi_c * 100],
        "cer_clinical": cer_c * 100, "cer_clinical_ci95": [clo_c * 100, chi_c * 100],
        "wer_clip_median": float(np.median(clip_wer) * 100) if n else float("nan"),
        "wer_clip_p90": float(np.percentile(clip_wer, 90) * 100) if n else float("nan"),
        "clips_wer_over_100_pct": float((clip_wer > 1.0).mean() * 100) if n else float("nan"),
        "wer_loop_collapsed": float(ce.sum() / max(cn.sum(), 1) * 100),
        "clinical_term_recall_ci95": list(wilson_ci(tp, tp + fn)),
        "clinical_term_enough_data": (tp + fn) >= 100,
        "clinical_term_tp_fn_fp": [tp, fn, fp],
        "sub_rate": S / N * 100, "del_rate": D / N * 100, "ins_rate": I / N * 100,
        "hallucination_rate": sum(halluc.values()) / max(n, 1) * 100,
        "hallucination_reasons": dict(halluc),
        "clinical_term_occurrences": tp + fn, "clinical_term_recall": rec * 100,
        "clinical_term_precision": prec * 100,
        "clinical_term_f1": (2 * prec * rec / (prec + rec) * 100) if prec == prec and rec == rec and prec + rec else float("nan"),
    }


# ---------------------------------------------------------------------------
# Manifests, leakage-aware tagging, stratified sampling
# ---------------------------------------------------------------------------

def _video_from_file(audio_file: str) -> str:
    return os.path.splitext(audio_file)[0].split("_")[-1]


def normalize_rows(manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for x in manifest:
        rows.append({
            "chunk_id": x["chunk_id"], "audio_file": x["audio_file"],
            "dur": float(x.get("duration_seconds", x.get("duration", 0.0))),
            "ref": x["text"],
            "video_id": x.get("video_id") or _video_from_file(x["audio_file"]),
            "level": x.get("source_level") or ("Level_2_Clinical_Education_OSCE" if x["chunk_id"].startswith("CLEAN") else "unknown"),
            "category": x.get("clinical_category") or x.get("topic") or "unknown",
        })
    return rows


def _shingles(text: str, k: int = 4) -> set:
    w = normalize(text).split()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def tag_rows(train: List[Dict[str, Any]], test: List[Dict[str, Any]], near_dup_threshold: Optional[float] = 0.6,
             stats: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """Tag clips relative to the manifest the adapter was trained on.
    Heldout clips are dropped when (a) their normalised text equals a train text, or (b) at least
    `near_dup_threshold` of their word 4-gram shingles occur in train (near-duplicate; None disables).
    Drop counts are written to `stats` if given."""
    train_vids = {r["video_id"] for r in train}
    train_texts = {normalize(r["ref"]) for r in train}
    train_shingles = set()
    if near_dup_threshold is not None:
        for r in train:
            train_shingles |= _shingles(r["ref"])
    out = [dict(r, tag="train_clip") for r in train]
    exact = near = 0
    for r in test:
        if normalize(r["ref"]) in train_texts:
            exact += 1
            continue
        if near_dup_threshold is not None:
            sh = _shingles(r["ref"])
            if sh and len(sh & train_shingles) / len(sh) >= near_dup_threshold:
                near += 1
                continue
        out.append(dict(r, tag="heldout_seen_video" if r["video_id"] in train_vids else "heldout_unseen_video"))
    if stats is not None:
        stats.update(dropped_exact=exact, dropped_near=near)
    return out


def eligible(r: Dict[str, Any], min_words: int = 3) -> bool:
    return 0.5 <= r["dur"] <= MAX_CLIP_SEC and len(normalize(r["ref"]).split()) >= min_words


def stratified_sample(rows: List[Dict[str, Any]], n: int, key: str = "category", seed: int = 0) -> List[Dict[str, Any]]:
    """Proportional allocation per stratum (>=1 each), remainder filled randomly. Deterministic."""
    rng = random.Random(seed)
    if n >= len(rows):
        return list(rows)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    for g in groups.values():
        rng.shuffle(g)
    total = len(rows)
    picked = []
    for g in groups.values():
        picked += g[:max(1, round(n * len(g) / total))]
    picked = picked[:n] if len(picked) > n else picked
    if len(picked) < n:
        ids = {r["chunk_id"] for r in picked}
        rest = [r for r in rows if r["chunk_id"] not in ids]
        rng.shuffle(rest)
        picked += rest[:n - len(picked)]
    return picked


def build_pool(train_rows, test_rows, n_per_tag: int, seed: int = 0,
               near_dup_threshold: Optional[float] = 0.6) -> List[Dict[str, Any]]:
    tagged = [r for r in tag_rows(train_rows, test_rows, near_dup_threshold) if eligible(r)]
    pool = []
    for tag in ("train_clip", "heldout_seen_video", "heldout_unseen_video"):
        pool += stratified_sample([r for r in tagged if r["tag"] == tag], n_per_tag, seed=seed)
    return pool


# ---------------------------------------------------------------------------
# Sessions for downstream (SOAP) evaluation
# ---------------------------------------------------------------------------

def _chunk_no(audio_file: str) -> int:
    m = re.search(r"chk_(\d+)_", audio_file)
    return int(m.group(1)) if m else -1


def build_sessions(all_rows: List[Dict[str, Any]], n: int, window: int = 12, seed: int = 0,
                   level_prefix: str = "Level_3") -> List[List[Dict[str, Any]]]:
    """Non-overlapping windows of strictly consecutive chunks from one video (multi-minute 'consultations')."""
    by_vid = collections.defaultdict(list)
    for r in all_rows:
        if r["level"].startswith(level_prefix) and _chunk_no(r["audio_file"]) >= 0 and r["dur"] <= MAX_CLIP_SEC:
            by_vid[r["video_id"]].append(r)
    cands = []
    for rows in by_vid.values():
        rows.sort(key=lambda r: _chunk_no(r["audio_file"]))
        runs, cur = [], [rows[0]]
        for r in rows[1:]:
            if _chunk_no(r["audio_file"]) == _chunk_no(cur[-1]["audio_file"]) + 1:
                cur.append(r)
            else:
                runs.append(cur)
                cur = [r]
        runs.append(cur)
        for run in runs:  # windows only inside strictly consecutive runs
            cands += [run[i:i + window] for i in range(0, len(run) - window + 1, window)]
    random.Random(seed).shuffle(cands)
    out, per_video = [], collections.Counter()
    for w in cands:  # at most 2 sessions per video for diversity
        if per_video[w[0]["video_id"]] >= 2:
            continue
        per_video[w[0]["video_id"]] += 1
        out.append(w)
        if len(out) >= n:
            break
    return out


def vitals_unsupported(soap, transcript: str) -> List[str]:
    """Names of vitals reported in the SOAP whose number never appears in the transcript (possible fabrication)."""
    from clinical_safety import filter_vitals
    v = soap.vitals
    _, bad = filter_vitals(dict(systolic=v.systolic_bp, diastolic=v.diastolic_bp,
                               heart_rate=v.heart_rate, respiratory_rate=v.respiratory_rate,
                               temperature=v.temperature, spo2=v.spo2), transcript)
    return bad


def code_sets(soap) -> Dict[str, set]:
    return {
        "icd10": {d.icd10_code for d in soap.diagnoses if d.icd10_code != "UNMAPPED"},
        "kfa": {m.kfa_code for m in soap.medications if m.kfa_code != "UNMAPPED"},
        "icd9cm": {p.icd9cm_code for p in soap.procedures if p.icd9cm_code != "UNMAPPED"},
    }


def micro_prf(pairs: List[Tuple[set, set]]) -> Dict[str, float]:
    """pairs of (gold_set, pred_set) -> micro precision/recall/F1 (%)."""
    tp = sum(len(g & p) for g, p in pairs)
    fp = sum(len(p - g) for g, p in pairs)
    fn = sum(len(g - p) for g, p in pairs)
    pr = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * pr * rc / (pr + rc) if pr == pr and rc == rc and pr + rc else float("nan")
    return {"precision": pr * 100, "recall": rc * 100, "f1": f1 * 100, "gold_items": tp + fn}


# ---------------------------------------------------------------------------
# GPU runners (exercised in Colab, not unit-tested here)
# ---------------------------------------------------------------------------

class CTranslate2Pipe:
    """High-throughput CTranslate2 INT8/FP16 wrapper for Whisper inference."""
    def __init__(self, model_path: str, device: str = "cuda", compute_type: str = "float16"):
        import ctranslate2
        from transformers import WhisperProcessor
        self.model = ctranslate2.models.Whisper(model_path, device=device, compute_type=compute_type)
        self.proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")

    def __call__(self, inputs: List[Dict[str, Any]], batch_size: int = 16, generate_kwargs: Optional[Dict[str, Any]] = None):
        import ctranslate2
        outs = []
        beams = generate_kwargs.get("num_beams", 1) if generate_kwargs else 1
        for it in inputs:
            feats = self.proc.feature_extractor(it["raw"], sampling_rate=SAMPLE_RATE, return_tensors="np").input_features
            s_view = ctranslate2.StorageView.from_array(feats)
            prompt = self.proc.tokenizer.convert_tokens_to_ids([
                "<|startoftranscript|>", "<|id|>", "<|transcribe|>", "<|notimestamps|>"
            ])
            res = self.model.generate(s_view, [prompt], beam_size=beams)
            text = self.proc.tokenizer.decode(res[0].sequences_ids[0], skip_special_tokens=True)
            outs.append({"text": text.strip()})
        return outs


def build_asr_pipe(adapter_dir: Optional[str] = None, model_id: str = "openai/whisper-large-v3-turbo",
                   backend: str = "transformers", compute_type: str = "float16"):
    import torch, os
    from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor

    if backend == "ctranslate2":
        return CTranslate2Pipe(model_path=adapter_dir or model_id, compute_type=compute_type)

    target_id = model_id
    use_peft = False
    if adapter_dir and os.path.isdir(adapter_dir):
        has_adapter_cfg = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
        has_model_weights = os.path.exists(os.path.join(adapter_dir, "model.safetensors")) or os.path.exists(os.path.join(adapter_dir, "pytorch_model.bin"))
        if has_model_weights and not has_adapter_cfg:
            # Standalone merged model
            target_id = adapter_dir
        elif has_adapter_cfg:
            use_peft = True
    elif adapter_dir:
        use_peft = True

    proc = WhisperProcessor.from_pretrained(target_id)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(target_id, dtype=torch.float16, low_cpu_mem_usage=True)
    except TypeError:  # transformers < 4.56
        model = WhisperForConditionalGeneration.from_pretrained(target_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    if use_peft and adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    # A model saved from a training run can carry use_cache=False (needed for gradient checkpointing); decoding
    # without the KV cache is slower and needs more memory, which would skew speed/VRAM comparisons.
    model.config.use_cache = True
    model.generation_config.forced_decoder_ids = None
    model.to("cuda").eval()
    return pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer,
                    feature_extractor=proc.feature_extractor, device=0)


def _load(path: str):
    import librosa
    try:
        return librosa.load(path, sr=SAMPLE_RATE)[0]
    except Exception:
        return None


# "plain": decoding exactly as in the original benchmark. "guarded": loop protection applied
# identically to every model, so the baseline is not penalised for decoder repetition loops.
# "beam_fallback": beam search on top of the guard, then every output that degenerate_reason() flags is
# decoded again with FALLBACK_KWARGS and replaced if the retry is no longer degenerate (Whisper-style fallback,
# but driven by observable failure only, so it never sees the reference).
DECODE_PROFILES = {
    "plain": {},
    "guarded": {"no_repeat_ngram_size": 6, "repetition_penalty": 1.05, "max_new_tokens": 200},
    "beam_fallback": {"num_beams": 4, "no_repeat_ngram_size": 6, "repetition_penalty": 1.05,
                      "max_new_tokens": 200, "_fallback": True},
}
FALLBACK_KWARGS = {"num_beams": 5, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2, "max_new_tokens": 160}


def transcribe_rows(pipe, rows: List[Dict[str, Any]], audio_dir: str, batch_size: int = 16,
                    workers: int = 8, beams: int = 1, shard: int = 256,
                    profile: str = "plain") -> Tuple[Dict[str, Optional[str]], Dict[str, float]]:
    """Returns (hyps by chunk_id, timing). Missing/undecodable audio -> None (reported, never silently scored as empty)."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import tqdm
    hyps: Dict[str, Optional[str]] = {}
    gpu_s = audio_s = 0.0
    warmed = False
    wall0 = time.time()
    gk = {"language": "indonesian", "task": "transcribe", "num_beams": beams, **DECODE_PROFILES[profile]}
    use_fallback = gk.pop("_fallback", False)
    fb_kwargs = {"language": "indonesian", "task": "transcribe", **FALLBACK_KWARGS}
    n_flagged = n_fixed = 0
    with ThreadPoolExecutor(workers) as ex:
        for i in tqdm(range(0, len(rows), shard), desc="ASR"):
            part = rows[i:i + shard]
            auds = list(ex.map(lambda r: _load(os.path.join(audio_dir, r["audio_file"])), part))
            ok = [(r, a) for r, a in zip(part, auds) if a is not None]
            for r, a in zip(part, auds):
                if a is None:
                    hyps[r["chunk_id"]] = None
            if not ok:
                continue
            if not warmed:  # untimed warm-up so CUDA init/kernel compilation is not billed to whichever model runs first
                pipe([{"raw": a, "sampling_rate": SAMPLE_RATE} for _, a in ok[:2]], batch_size=2, generate_kwargs=gk)
                torch.cuda.reset_peak_memory_stats()
                warmed = True
            torch.cuda.synchronize()
            t0 = time.time()
            outs = pipe([{"raw": a, "sampling_rate": SAMPLE_RATE} for _, a in ok],
                        batch_size=batch_size, generate_kwargs=gk)
            texts = [o["text"].strip() for o in outs]
            if use_fallback:
                bad = [k for k, ((_, a), t) in enumerate(zip(ok, texts))
                       if degenerate_reason(t, len(a) / SAMPLE_RATE)]
                if bad:
                    n_flagged += len(bad)
                    retry = pipe([{"raw": ok[k][1], "sampling_rate": SAMPLE_RATE} for k in bad],
                                 batch_size=max(1, batch_size // 2), generate_kwargs=fb_kwargs)
                    for k, o in zip(bad, retry):
                        t = o["text"].strip()
                        if not degenerate_reason(t, len(ok[k][1]) / SAMPLE_RATE):
                            texts[k] = t
                            n_fixed += 1
            torch.cuda.synchronize()
            gpu_s += time.time() - t0
            for (r, a), t in zip(ok, texts):
                hyps[r["chunk_id"]] = t
                audio_s += len(a) / SAMPLE_RATE
    wall = time.time() - wall0
    return hyps, {"gpu_s": gpu_s, "wall_s": wall, "audio_s": audio_s,
                  "rtf_gpu": gpu_s / max(audio_s, 1e-9), "rtf_wall": wall / max(audio_s, 1e-9),
                  "x_realtime_gpu": audio_s / max(gpu_s, 1e-9),
                  "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                  "missing_audio": sum(v is None for v in hyps.values()),
                  "fallback_flagged": n_flagged, "fallback_fixed": n_fixed}


def export_verification_sheet(path: str, rows: List[Dict[str, Any]], hyps: Dict[str, Dict[str, str]]) -> None:
    """CSV for clinicians: fill corrected_text with the true transcript. Scoring against it gives the real WER."""
    variants = sorted(hyps)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "audio_file", "current_reference"] + [f"hyp_{v}" for v in variants] + ["corrected_text"])
        for r in rows:
            w.writerow([r["chunk_id"], r["audio_file"], r["ref"]] + [hyps[v].get(r["chunk_id"], "") for v in variants] + [""])


def load_verified(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("corrected_text") or "").strip():
                out.append({"chunk_id": r["chunk_id"], "ref": r["corrected_text"].strip(), "audio_file": r["audio_file"]})
    return out