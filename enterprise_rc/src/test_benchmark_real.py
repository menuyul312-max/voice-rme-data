"""CPU tests for benchmark scoring logic: python test_benchmark_real.py"""
import benchmark_real as b
from local_med_llm import CLINICAL_ONTOLOGY, DialogueTurn
from live_pipeline_t4 import ground_soap


def test_normalize_and_wer_counts():
    assert b.normalize("Mata-kanan, MERAH!!  ") == "mata kanan merah"
    assert b.normalize_clinical_conversational("ee gak papa udah minum 500mg nah") == "tidak papa sudah minum 500 miligram"
    assert b.word_error_counts("a b c d".split(), "a x c".split()) == (1, 1, 0)   # 1 sub, 1 del
    assert b.word_error_counts("a b".split(), "a b c d".split()) == (0, 0, 2)
    assert b.levenshtein("kitten", "sitting") == 3


def test_score_items_perfect_and_bad():
    terms = b.build_term_list(CLINICAL_ONTOLOGY)
    items = [{"ref": "pasien konjungtivitis akut", "hyp": "pasien konjungtivitis akut", "dur": 3},
             {"ref": "resep kloramfenikol tetes mata", "hyp": "terima kasih telah menonton video ini semua", "dur": 4}]
    m = b.score_items(items, terms, B=50)
    assert m["n_clips"] == 2 and m["wer"] > 50
    assert m["clinical_term_recall"] < 100 and m["clinical_term_occurrences"] >= 2
    perfect = b.score_items(items[:1], terms, B=50)
    assert perfect["wer"] == 0 and perfect["cer"] == 0 and perfect["hallucination_rate"] == 0


def test_hallucination_flags():
    assert b.hallucination_flag("halo dok", "") == "EMPTY"
    assert b.hallucination_flag("halo", "ya " * 30) == "LENGTH_RATIO"
    assert b.hallucination_flag("halo dok", "halo dok") is None


def test_tagging_no_leak_labels():
    tr = [dict(chunk_id="a", audio_file="x_1_V1.mp3", dur=5, ref="satu dua tiga", video_id="V1", level="L", category="c")]
    te = [dict(chunk_id="b", audio_file="x_2_V1.mp3", dur=5, ref="empat lima enam", video_id="V1", level="L", category="c"),
          dict(chunk_id="c", audio_file="x_3_V2.mp3", dur=5, ref="tujuh delapan sembilan", video_id="V2", level="L", category="c"),
          dict(chunk_id="d", audio_file="x_4_V2.mp3", dur=5, ref="Satu, dua tiga!", video_id="V2", level="L", category="c")]
    tags = {r["chunk_id"]: r["tag"] for r in b.tag_rows(tr, te)}
    assert tags == {"a": "train_clip", "b": "heldout_seen_video", "c": "heldout_unseen_video"}  # d dropped: exact text dup


def test_stratified_sample_deterministic():
    rows = [dict(chunk_id=str(i), category="A" if i < 90 else "B") for i in range(100)]
    s1, s2 = b.stratified_sample(rows, 20, seed=1), b.stratified_sample(rows, 20, seed=1)
    assert [r["chunk_id"] for r in s1] == [r["chunk_id"] for r in s2] and len(s1) == 20
    assert any(r["category"] == "B" for r in s1)


def test_sessions_require_consecutive_chunks():
    rows = [dict(chunk_id=f"c{i}", audio_file=f"lvl3_chk_{i:05d}_V1.mp3", dur=10, ref="x y z", video_id="V1",
                 level="Level_3_Real", category="c") for i in list(range(1, 13)) + [20, 21, 23, 24] + list(range(30, 42))]
    sess = b.build_sessions(rows, n=5, window=12, seed=0)
    assert len(sess) == 2 and all(len(w) == 12 for w in sess)
    nums = [b._chunk_no(r["audio_file"]) for r in sess[0]]
    assert nums == list(range(nums[0], nums[0] + 12))


def test_fabrication_and_code_sets():
    turns = [DialogueTurn(1, "DOKTER", 0, 5, "tensi 130 per 85 nadi 80 resep kloramfenikol")]
    meta = dict(patient_id="P", patient_name="N", doctor_name="D", encounter_id="E")
    soap = ground_soap({"vitals": {"systolic": 130, "diastolic": 85, "heart_rate": 99},
                        "medications": [{"term": "kloramfenikol", "evidence_turn": 1}]}, turns, meta)["soap"]
    assert soap.vitals.heart_rate is None
    soap.vitals.heart_rate = 99  # independently inject fabrication to test evaluator
    assert b.vitals_unsupported(soap, turns[0].text) == ["heart_rate"]
    assert b.code_sets(soap)["kfa"] == {"93000123"}
    assert b.micro_prf([({"a", "b"}, {"a", "c"})])["f1"] == 50.0


def test_verification_roundtrip(tmp="/tmp/_ver.csv"):
    rows = [dict(chunk_id="a", audio_file="a.mp3", ref="lama")]
    b.export_verification_sheet(tmp, rows, {"base": {"a": "baru"}})
    text = open(tmp, encoding="utf-8").read().replace(",\n", ",benar\n").replace(",\r\n", ",benar\r\n")
    open(tmp, "w", encoding="utf-8").write(text)
    assert b.load_verified(tmp)[0]["ref"] == "benar"


def test_collapse_repetitions():
    assert b.collapse_repetitions("a b a b a b a b c") == "a b a b c"
    assert b.collapse_repetitions("ya ya ya ya ya") == "ya ya"
    assert b.collapse_repetitions("tidak ada pengulangan di sini") == "tidak ada pengulangan di sini"


def test_wilson_and_mcnemar():
    lo, hi = b.wilson_ci(11, 12)
    assert 64 < lo < 70 and 98 < hi <= 100          # 11/12 is NOT precise: CI is ~65-99%
    assert b.mcnemar_exact(1, 0) == 1.0              # one discordant pair proves nothing
    assert b.mcnemar_exact(10, 0) < 0.01
    assert b.mcnemar_exact(0, 0) == 1.0


def test_paired_bootstrap_detects_real_gap_and_null():
    ref = "pasien datang dengan keluhan demam dan batuk"
    good = [dict(ref=ref, hyp=ref, dur=3) for _ in range(40)]
    bad = [dict(ref=ref, hyp="pasien datang dengan keluhan", dur=3) for _ in range(40)]
    r = b.paired_bootstrap(good, bad, B=300)
    assert r["delta_pp"] < -30 and r["delta_ci95"][1] < 0 and r["p_value"] < 0.05
    same = b.paired_bootstrap(good, good, B=300)
    assert same["delta_pp"] == 0 and same["p_value"] > 0.5


def test_loop_heavy_baseline_is_exposed_by_median_and_collapse():
    ref = "pasien mengeluh nyeri dada sejak pagi"
    items = [dict(ref=ref, hyp=ref, dur=3) for _ in range(19)]
    items.append(dict(ref=ref, hyp=ref + " terima kasih " * 40, dur=3))   # one decoder loop
    m = b.score_items(items, [], B=50)
    assert m["wer"] > 50                       # corpus WER dominated by a single loop clip
    assert m["wer_clip_median"] == 0           # typical clip is perfect
    assert m["clips_wer_over_100_pct"] == 5.0
    assert m["wer_loop_collapsed"] < m["wer"] / 3


def test_paired_term_test_counts():
    terms = ["konjungtivitis", "kloramfenikol"]
    ref = "diagnosis konjungtivitis resep kloramfenikol"
    a = [dict(ref=ref, hyp=ref, dur=1)] * 3
    c = [dict(ref=ref, hyp="diagnosis konjungtivitis resep", dur=1)] * 3
    r = b.paired_term_test(a, c, terms)
    assert (r["term_occurrences"], r["both_hit"], r["a_only"], r["b_only"]) == (6, 3, 3, 0)
    assert not r["enough_data"]
    m = b.score_items(a, terms, B=20)
    assert m["clinical_term_enough_data"] is False and m["clinical_term_recall_ci95"][0] < 100


def test_near_duplicate_filter():
    base = "dokter menyarankan pasien minum obat tiga kali sehari setelah makan dan istirahat cukup"
    tr = [dict(chunk_id="a", audio_file="x_1_V1.mp3", dur=5, ref=base, video_id="V1", level="L", category="c")]
    te = [dict(chunk_id="n", audio_file="x_2_V2.mp3", dur=5, ref=base + " ya", video_id="V2", level="L", category="c"),
          dict(chunk_id="k", audio_file="x_3_V2.mp3", dur=5, ref="hasil laboratorium menunjukkan kadar gula darah sangat tinggi pagi ini", video_id="V2", level="L", category="c")]
    st = {}
    tags = {r["chunk_id"] for r in b.tag_rows(tr, te, stats=st)}
    assert tags == {"a", "k"} and st["dropped_near"] == 1
    assert {r["chunk_id"] for r in b.tag_rows(tr, te, near_dup_threshold=None)} == {"a", "n", "k"}


def test_normalization_keeps_meaning():
    assert "cuma" not in b.normalize_clinical_conversational("hanya demam")
    assert b.normalize_clinical_conversational("capai sekali") == "capai sekali"


if __name__ == "__main__":
    for n in [x for x in dir() if x.startswith("test_")]:
        globals()[n](); print("PASS", n)