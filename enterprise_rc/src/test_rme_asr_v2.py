"""CPU tests for the ASR v2 recipe and decoding fallback: python test_rme_asr_v2.py"""
import numpy as np

import benchmark_real as b
import rme_asr_v2 as v2
from local_med_llm import CLINICAL_ONTOLOGY


def _rows(n_videos=40, clips=10, levels=("L2", "L3")):
    return [dict(chunk_id=f"{v}_{i}", audio_file=f"x_{i}_{v}.mp3", dur=10.0, ref="satu dua tiga empat lima",
                 video_id=f"V{v}", level=levels[v % len(levels)], category="c")
            for v in range(n_videos) for i in range(clips)]


def test_split_is_by_video_deterministic_and_stratified():
    rows = _rows()
    s1, s2 = v2.split_videos(rows, 0.1, 0.2, seed=3), v2.split_videos(rows, 0.1, 0.2, seed=3)
    assert s1 == s2 and set(s1) == {r["video_id"] for r in rows}
    for lvl in ("L2", "L3"):  # every level gets its own test and dev videos
        got = {s1[r["video_id"]] for r in rows if r["level"] == lvl}
        assert got == {"train", "dev", "test"}
    test_clips = sum(s1[r["video_id"]] == "test" for r in rows)
    assert 0.15 * len(rows) <= test_clips <= 0.3 * len(rows)
    assert v2.split_videos(rows, 0.1, 0.2, seed=4) != s1


def test_split_counts_clips_not_videos():
    rows = _rows(n_videos=20, clips=5, levels=("L",))
    rows += [dict(r, chunk_id=f"big{i}", video_id="BIG") for i, r in enumerate(_rows(1, 200, ("L",)))]
    s = v2.split_videos(rows, 0.0, 0.2, seed=0)
    test_clips = sum(s[r["video_id"]] == "test" for r in rows)
    # one 200-clip video cannot be followed by more test videos once the clip budget is met
    assert test_clips <= 0.2 * len(rows) + 200


def test_label_issue_reasons():
    ok = dict(ref="pasien mengeluh nyeri dada sejak pagi tadi", dur=4.0)
    assert v2.label_issue(ok) is None
    assert v2.label_issue(dict(ok, dur=45.0)) == "TOO_LONG"
    assert v2.label_issue(dict(ok, dur=0.5)) == "TOO_SHORT"
    assert v2.label_issue(dict(ref="penting adalah memilih tidak kita", dur=13.6)) == "SLOW_LABEL"
    assert v2.label_issue(dict(ref=" ".join(["kata"] + [f"w{i}" for i in range(40)]), dur=5.0)) == "FAST_LABEL"
    assert v2.label_issue(dict(ref="hai hai hai hai hai hai hai hai hai hai", dur=5.0)) == "LOOPING_LABEL"
    assert v2.label_issue(dict(ref="ya", dur=1.5)) == "TOO_FEW_WORDS"
    kept, dropped = v2.curate([ok, dict(ok, dur=45.0)])
    assert len(kept) == 1 and dropped == {"TOO_LONG": 1}


def test_augment_bounded_and_window_safe():
    rng = np.random.default_rng(0)
    aug = v2.WaveAugment(p_speed=1.0, p_noise=1.0, p_gain=1.0)
    x = (0.3 * np.sin(np.linspace(0, 2000, 16000 * 5))).astype(np.float32)
    for _ in range(20):
        y = aug(x, rng)
        assert y.dtype == np.float32 and np.isfinite(y).all() and np.abs(y).max() <= 1.0
        assert 0.85 * len(x) <= len(y) <= 1.15 * len(x)
    long = np.zeros(16000 * 29, dtype=np.float32) + 0.1
    assert all(len(aug(long, rng)) <= 16000 * 30 for _ in range(20))
    assert len(aug(np.zeros(0, dtype=np.float32), rng)) == 0


def test_degenerate_detection():
    assert b.degenerate_reason("terima kasih " * 20, 5.0) == "COMPRESSION"
    assert b.degenerate_reason("pasien datang dengan keluhan nyeri dada kiri sejak dua hari yang lalu", 6.0) is None
    assert b.degenerate_reason("", 5.0) is None                       # silence is fine
    assert b.degenerate_reason("iya iya iya", 2.0) is None              # short natural repetition is fine
    words = " ".join("".join(chr(97 + (i * 7 + j * 3) % 26) for j in range(4 + i % 3)) for i in range(40))
    assert b.degenerate_reason(words, 2.0) == "WORDS_PER_SEC"
    assert b.compression_ratio("abc") < b.compression_ratio("abc " * 50)


def test_decode_profiles_have_internal_flag_only_in_fallback():
    assert b.DECODE_PROFILES["beam_fallback"]["_fallback"] is True
    assert not any(k.startswith("_") for p in ("plain", "guarded") for k in b.DECODE_PROFILES[p])
    assert not any(k.startswith("_") for k in b.FALLBACK_KWARGS)


def test_corpus_wer_matches_benchmark_normalisation():
    assert v2.corpus_wer(["Mata-kanan merah"], ["mata kanan merah"]) == 0.0
    assert abs(v2.corpus_wer(["a b c d"], ["a x c"]) - 50.0) < 1e-9


def test_term_recall_by_group():
    groups = b.term_groups(CLINICAL_ONTOLOGY)
    assert groups["kloramfenikol"].endswith("/medications")
    items = [dict(ref="resep kloramfenikol tetes", hyp="resep kloramfenikol tetes"),
             dict(ref="resep kloramfenikol tetes", hyp="resep klor tetes")]
    g = b.term_recall_by_group(items, groups)[groups["kloramfenikol"]]
    assert g["n"] == 2 and g["hit"] == 1 and g["recall"] == 50.0


if __name__ == "__main__":
    for n in [x for x in dir() if x.startswith("test_")]:
        globals()[n](); print("PASS", n)