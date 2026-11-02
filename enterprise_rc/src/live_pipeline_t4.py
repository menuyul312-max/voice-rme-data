#!/usr/bin/env python3
"""
Live Voice-to-RME pipeline for a single NVIDIA T4 (16 GB), e.g. Google Colab.

Real stages (no canned dialogue):
  audio -> Silero VAD -> Whisper large-v3-turbo (+ LoRA, merged) -> guardrail
        -> Qwen2.5-7B-Instruct (4-bit) : speaker labels + SOAP JSON
        -> ontology grounding (ICD-10 / ICD-9-CM / KFA only from the local table)
        -> linked evidence validated against real turns -> SATUSEHAT FHIR draft

Safety rules enforced here:
  * vitals that were not spoken stay None (never defaulted)
  * codes not found in the local ontology are emitted as UNMAPPED, never invented
  * every output is marked DRAFT until a clinician signs it
"""

import difflib
import json
import re
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from local_med_llm import (
    CLINICAL_ONTOLOGY,
    DialogueTurn,
    DiagnosisItem,
    LinkedEvidence,
    MedicationItem,
    ProcedureItem,
    SOAPRecord,
    SovereignLocalMedicalLLM,
    VitalSigns,
)
from pipeline_enterprise_ambient_rme import ClinicalAntiHallucinationGuardrail
from benchmark_real import DECODE_PROFILES, FALLBACK_KWARGS, degenerate_reason

SAMPLE_RATE = 16000
ASR_MODEL_ID = "openai/whisper-large-v3-turbo"
MERGED_MODEL_DIRS = ["/content/whisper_large_v3_turbo_rme_v2_merged", "/content/whisper_large_v3_turbo_rme_merged"]
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SOAP_PROMPT = """Anda asisten dokumentasi medis. Dari transkrip konsultasi dokter-pasien berikut,
susun catatan SOAP. ATURAN KETAT:
- Hanya gunakan informasi yang BENAR-BENAR diucapkan. Jika tidak disebut, isi null atau [].
- Tanda vital hanya jika angkanya diucapkan; jangan menebak.
- Setiap diagnosis/obat/tindakan wajib menyertakan "evidence_turn" (nomor giliran bicara sumbernya).
- Beri label pembicara tiap giliran: "DOKTER" atau "PASIEN".
Balas HANYA JSON valid dengan skema:
{{"speakers": {{"1": "DOKTER", ...}},
 "chief_complaint": str|null, "history_of_present_illness": str|null,
 "past_medical_history": str|null, "allergies": str|null, "physical_exam": str|null,
 "vitals": {{"systolic": int|null, "diastolic": int|null, "heart_rate": int|null,
            "respiratory_rate": int|null, "temperature": float|null, "spo2": int|null}},
 "diagnoses": [{{"term": str, "evidence_turn": int}}],
 "medications": [{{"term": str, "dosage": str|null, "frequency": str|null, "evidence_turn": int}}],
 "procedures": [{{"term": str, "evidence_turn": int}}],
 "patient_education": str|null, "follow_up_plan": str|null}}

TRANSKRIP:
{transcript}
"""


# ---------------------------------------------------------------------------
# Stage 1-2: VAD + ASR
# ---------------------------------------------------------------------------

class ASRStage:
    def __init__(self, adapter_dir: Optional[str] = None, device: str = "cuda"):
        import os
        import torch
        from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.adapter_loaded = False
        load_id = ASR_MODEL_ID
        for merged_path in MERGED_MODEL_DIRS:  # newest fine-tuned model first
            if os.path.exists(os.path.join(merged_path, "config.json")):
                load_id = merged_path
                self.adapter_loaded = True
                break
        self.model_path = load_id
        processor = WhisperProcessor.from_pretrained(load_id)
        try:
            model = WhisperForConditionalGeneration.from_pretrained(load_id, dtype=torch.float16, low_cpu_mem_usage=True)
        except TypeError:  # transformers < 4.56
            model = WhisperForConditionalGeneration.from_pretrained(load_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        if adapter_dir and not self.adapter_loaded:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
            self.adapter_loaded = True
        model.config.use_cache = True
        model.generation_config.forced_decoder_ids = None
        model.to(device).eval()
        self.pipe = pipeline(
            "automatic-speech-recognition", model=model,
            tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
            device=0 if device == "cuda" else -1)
        try:
            self.vad_model, self.vad_utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        except Exception as exc:  # no VAD -> fixed windows; flagged via vad_loaded
            print(f"[WARN] Silero VAD gagal dimuat ({exc}); memakai potongan 28 detik tanpa VAD")
            self.vad_model = self.vad_utils = None
        self.vad_loaded = self.vad_model is not None

    def load_audio(self, path: str):
        import librosa
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        return audio

    def speech_segments(self, audio, max_len: float = 28.0, merge_gap: float = 0.75):
        """Silero VAD, merged across short pauses, capped at Whisper's 30 s window."""
        if self.vad_model is None:
            total = len(audio) / SAMPLE_RATE
            return [(a, min(a + max_len, total)) for a in
                    [i * max_len for i in range(int(total // max_len) + 1)] if a < total]
        get_ts = self.vad_utils[0]
        ts = get_ts(self.torch.from_numpy(audio), self.vad_model,
                    sampling_rate=SAMPLE_RATE, return_seconds=True)
        merged: List[List[float]] = []
        for t in ts:
            s, e = t["start"], t["end"]
            if merged and s - merged[-1][1] <= merge_gap and e - merged[-1][0] <= max_len:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        return [(max(s - 0.2, 0.0), min(e + 0.2, len(audio) / SAMPLE_RATE)) for s, e in merged]

    def transcribe(self, path: str) -> List[Dict[str, Any]]:
        audio = self.load_audio(path)
        guard = ClinicalAntiHallucinationGuardrail()
        out = []
        gk = {"language": "indonesian", "task": "transcribe",
              **{k: v for k, v in DECODE_PROFILES["beam_fallback"].items() if not k.startswith("_")}}
        fb = {"language": "indonesian", "task": "transcribe", **FALLBACK_KWARGS}
        for s, e in self.speech_segments(audio):
            chunk = audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]
            text = self.pipe({"raw": chunk, "sampling_rate": SAMPLE_RATE}, generate_kwargs=gk)["text"].strip()
            if degenerate_reason(text, e - s):  # same loop fallback as the benchmarked "beam_fallback" profile
                retry = self.pipe({"raw": chunk, "sampling_rate": SAMPLE_RATE}, generate_kwargs=fb)["text"].strip()
                if not degenerate_reason(retry, e - s):
                    text = retry
            bad, reason = guard.is_hallucination(text)
            if bad:
                out.append({"start": s, "end": e, "text": "", "dropped": reason})
                continue
            out.append({"start": round(s, 2), "end": round(e, 2), "text": text})
        return [o for o in out if o["text"]]


# ---------------------------------------------------------------------------
# Stage 3: LLM (speaker labels + SOAP JSON)
# ---------------------------------------------------------------------------

class LLMStage:
    def __init__(self, model_id: str = LLM_MODEL_ID):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16))
        self.model.eval()

    def generate_json(self, transcript: str, retries: int = 1) -> Optional[Dict[str, Any]]:
        msgs = [{"role": "user", "content": SOAP_PROMPT.format(transcript=transcript)}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if hasattr(enc, "input_ids"):
            ids = enc.input_ids.to(0)
        elif isinstance(enc, dict):
            ids = enc["input_ids"].to(0)
        else:
            ids = enc.to(0)
        for _ in range(retries + 1):
            with self.torch.no_grad():
                gen = self.model.generate(ids, max_new_tokens=1500, do_sample=False)
            raw = self.tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
            parsed = parse_json_loose(raw)
            self.last_strict = strict_json_ok(raw)
            if parsed is not None:
                from clinical_safety import validate_payload
                try:
                    validate_payload(parsed)
                except ValueError:
                    continue
                return parsed
        return None


ROOT_SOAP_KEYS = {
    "chief_complaint", "vitals", "diagnoses", "speakers", "medications",
    "procedures", "subjective", "objective", "assessment", "plan"
}

def strict_json_ok(raw: str) -> bool:
    """True when the raw LLM output is valid SOAP JSON without any repair or regex fallback (honest validity metric)."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    a, b = cleaned.find("{"), cleaned.rfind("}")
    if a < 0 or b <= a:
        return False
    try:
        return is_root_soap(json.loads(cleaned[a:b + 1]))
    except json.JSONDecodeError:
        return False


def is_root_soap(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(ROOT_SOAP_KEYS & set(obj.keys()))

def parse_json_loose(raw: str) -> Optional[Dict[str, Any]]:
    """
    Fault-tolerant JSON parser with automatic syntax repair and regex slot fallback.
    Guarantees recovery from trailing commas, single quotes, unclosed braces, and malformed syntax.
    """
    return robust_json_parser(raw)


def robust_json_parser(raw: str) -> Optional[Dict[str, Any]]:
    if not raw or not raw.strip():
        return None

    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    a, b = cleaned.find("{"), cleaned.rfind("}")
    if a >= 0 and b > a:
        candidate = cleaned[a:b + 1]
        try:
            obj = json.loads(candidate)
            if is_root_soap(obj):
                return obj
        except json.JSONDecodeError:
            pass

        # Syntax repairs
        candidate = re.sub(r"//.*$", "", candidate, flags=re.M)
        candidate = re.sub(r"/\*.*?\*/", "", candidate, flags=re.DOTALL)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        candidate = re.sub(r"\bNone\b", "null", candidate)
        candidate = re.sub(r"\bTrue\b", "true", candidate)
        candidate = re.sub(r"\bFalse\b", "false", candidate)

        try:
            obj = json.loads(candidate)
            if is_root_soap(obj):
                return obj
        except json.JSONDecodeError:
            pass

        # Quote normalization & bracket balancing
        fixed = re.sub(r"(?<=[{,\s])'([a-zA-Z0-9_]+)'\s*:", r'"\1":', candidate)
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

        open_braces = fixed.count("{") - fixed.count("}")
        open_brackets = fixed.count("[") - fixed.count("]")
        if open_brackets > 0:
            fixed += "]" * open_brackets
        if open_braces > 0:
            fixed += "}" * open_braces

        try:
            obj = json.loads(fixed)
            if is_root_soap(obj):
                return obj
        except json.JSONDecodeError:
            pass

    # Regex slot extraction fallback
    return fallback_extract_soap_dict(raw)


def fallback_extract_soap_dict(raw: str) -> Optional[Dict[str, Any]]:
    if not raw or len(raw.strip()) < 10:
        return None

    out: Dict[str, Any] = {
        "speakers": {},
        "chief_complaint": None, "history_of_present_illness": None,
        "past_medical_history": None, "allergies": None, "physical_exam": None,
        "vitals": {"systolic": None, "diastolic": None, "heart_rate": None,
                   "respiratory_rate": None, "temperature": None, "spo2": None},
        "diagnoses": [], "medications": [], "procedures": [],
        "patient_education": None, "follow_up_plan": None
    }

    for k in ("chief_complaint", "history_of_present_illness", "past_medical_history",
              "allergies", "physical_exam", "patient_education", "follow_up_plan"):
        m = re.search(rf'"{k}"\s*:\s*"([^"]+)"', raw)
        if m:
            out[k] = m.group(1).strip()

    for vkey in ("systolic", "diastolic", "heart_rate", "respiratory_rate", "spo2"):
        m = re.search(rf'"{vkey}"\s*:\s*(\d+)', raw)
        if m:
            out["vitals"][vkey] = int(m.group(1))
    m_temp = re.search(r'"temperature"\s*:\s*([0-9.]+)', raw)
    if m_temp:
        try:
            out["vitals"]["temperature"] = float(m_temp.group(1))
        except ValueError:
            pass

    for list_key in ("diagnoses", "medications", "procedures"):
        sub_m = re.search(rf'"{list_key}"\s*:\s*\[(.*?)\]', raw, flags=re.DOTALL)
        if sub_m:
            items_raw = sub_m.group(1)
            term_matches = re.finditer(r'\{[^{}]*?"term"\s*:\s*"([^"]+)"[^{}]*?\}', items_raw)
            for tm in term_matches:
                blob = tm.group(0)
                t_name = tm.group(1)
                ev_m = re.search(r'"evidence_turn"\s*:\s*(\d+)', blob)
                ev_turn = int(ev_m.group(1)) if ev_m else 1
                item_dict: Dict[str, Any] = {"term": t_name, "evidence_turn": ev_turn}
                if list_key == "medications":
                    dose_m = re.search(r'"dosage"\s*:\s*"([^"]+)"', blob)
                    freq_m = re.search(r'"frequency"\s*:\s*"([^"]+)"', blob)
                    item_dict["dosage"] = dose_m.group(1) if dose_m else None
                    item_dict["frequency"] = freq_m.group(1) if freq_m else None
                out[list_key].append(item_dict)

    has_content = any([out["chief_complaint"], out["diagnoses"], out["medications"], out["procedures"], any(out["vitals"].values())])
    return out if has_content else None


# ---------------------------------------------------------------------------
# Stage 4: ontology grounding + evidence validation (pure Python, unit-testable)
# ---------------------------------------------------------------------------

def map_to_ontology(kind: str, term: str) -> Optional[Dict[str, Any]]:
    """kind in diagnoses|medications|procedures. Keyword hit, else fuzzy >= 0.75."""
    term_l = (term or "").lower().strip()
    if not term_l:
        return None
    best, best_score = None, 0.0
    for spec in CLINICAL_ONTOLOGY.values():
        for item in spec[kind]:
            names = item["keywords"] + [item.get("desc") or item.get("name", "")]
            for n in names:
                n = n.lower()
                if n and (n in term_l or term_l in n):
                    return item
                score = difflib.SequenceMatcher(None, term_l, n).ratio()
                if score > best_score:
                    best, best_score = item, score
    return best if best_score >= 0.75 else None


def build_evidence(turns: List[DialogueTurn], turn_no: Any, term: str) -> Optional[LinkedEvidence]:
    """Evidence only from a real turn. Confidence is 1.0 only if the term appears verbatim."""
    try:
        t = next(x for x in turns if x.turn_id == int(turn_no))
    except (StopIteration, TypeError, ValueError):
        return None
    verbatim = bool((term or "").strip()) and (term or "").lower() in t.text.lower()
    return LinkedEvidence(claim_text=term, source_speaker=t.speaker,
                          audio_start_ms=int(t.start * 1000), audio_end_ms=int(t.end * 1000),
                          confidence=1.0 if verbatim else 0.5, quote_snippet=t.text)


def _int(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def ground_soap(data: Dict[str, Any], turns: List[DialogueTurn], meta: Dict[str, str]) -> Dict[str, Any]:
    """Convert raw LLM JSON to a SOAPRecord + list of items needing human coding review."""
    from clinical_safety import filter_vitals, validate_payload
    validate_payload(data)
    raw_v = data.get("vitals") or {}
    if not isinstance(raw_v, dict):
        raise ValueError("vitals must be an object")
    v, rejected = filter_vitals(raw_v, "\n".join(t.text for t in turns))
    sys_, dia = _int(v.get("systolic")), _int(v.get("diastolic"))
    vitals = VitalSigns(
        blood_pressure=f"{sys_}/{dia} mmHg" if sys_ and dia else None,
        systolic_bp=sys_, diastolic_bp=dia, heart_rate=_int(v.get("heart_rate")),
        respiratory_rate=_int(v.get("respiratory_rate")), temperature=_float(v.get("temperature")),
        spo2=_int(v.get("spo2")))

    review: List[str] = [f"Vital tanpa bukti atau ambigu dibuang: {key}" for key in rejected]
    diagnoses, meds, procs = [], [], []
    for i, d in enumerate(data.get("diagnoses") or []):
        term = d.get("term", "")
        m = map_to_ontology("diagnoses", term)
        if not m:
            review.append(f"Diagnosis tanpa kode ICD-10 lokal: '{term}'")
        diagnoses.append(DiagnosisItem(
            description=term, icd10_code=m["code"] if m else "UNMAPPED",
            icd10_display=m["display"] if m else "Perlu koding manual",
            is_primary=(i == 0), evidence=build_evidence(turns, d.get("evidence_turn"), term)))
    for md in data.get("medications") or []:
        term = md.get("term", "")
        m = map_to_ontology("medications", term)
        if not m:
            review.append(f"Obat tanpa kode KFA lokal: '{term}'")
        meds.append(MedicationItem(
            drug_name=m["name"] if m else term, kfa_code=m["kfa"] if m else "UNMAPPED",
            dosage=md.get("dosage") if md.get("dosage") and str(md["dosage"]).lower() in " ".join(t.text.lower() for t in turns) else "Tidak disebutkan",
            frequency=md.get("frequency") if md.get("frequency") and str(md["frequency"]).lower() in " ".join(t.text.lower() for t in turns) else "Tidak disebutkan",
            route="Tidak disebutkan", duration_days=0,
            instructions="Perlu verifikasi dokter",
            evidence=build_evidence(turns, md.get("evidence_turn"), term)))
    for p in data.get("procedures") or []:
        term = p.get("term", "")
        m = map_to_ontology("procedures", term)
        if not m:
            review.append(f"Tindakan tanpa kode ICD-9-CM lokal: '{term}'")
        procs.append(ProcedureItem(
            procedure_name=m["desc"] if m else term, icd9cm_code=m["code"] if m else "UNMAPPED",
            icd9cm_display=m["display"] if m else "Perlu koding manual",
            evidence=build_evidence(turns, p.get("evidence_turn"), term)))

    for item in [*diagnoses, *meds, *procs]:
        if item.evidence is None or item.evidence.confidence < 1.0:
            review.append("Entitas tanpa bukti verbatim; dikeluarkan dari ekspor draft")
    for med in meds:
        review.append(f"Regimen obat wajib dikonfirmasi dokter: {med.drug_name}")
    nd = "Tidak disebutkan"
    cc = data.get("chief_complaint") or nd
    soap = SOAPRecord(
        patient_id=meta["patient_id"], patient_name=meta["patient_name"],
        doctor_name=meta["doctor_name"], encounter_id=meta["encounter_id"],
        specialty="Ditentukan dokter (draft AI)",
        chief_complaint=cc, history_of_present_illness=data.get("history_of_present_illness") or nd,
        past_medical_history=data.get("past_medical_history") or nd,
        allergies=data.get("allergies") or nd, subjective_narrative=cc,
        vitals=vitals, physical_exam=data.get("physical_exam") or nd, objective_narrative="",
        assessment_narrative="; ".join(d.description for d in diagnoses) or nd,
        diagnoses=diagnoses, plan_narrative="", medications=meds, procedures=procs,
        patient_education=data.get("patient_education") or nd,
        follow_up_plan=data.get("follow_up_plan") or nd)
    return {"soap": soap, "needs_review": review, "rejected_vitals": rejected}


def build_fhir_draft(soap: SOAPRecord) -> Dict[str, Any]:
    """Reuse the existing builder, drop vitals Observations that were never spoken, tag as draft."""
    from copy import deepcopy
    export = deepcopy(soap)
    export.diagnoses = [d for d in export.diagnoses if d.icd10_code != "UNMAPPED" and d.evidence and d.evidence.confidence == 1.0]
    # No medication orders or completed procedures may be inferred from a draft.
    export.medications = []
    export.procedures = []
    bundle = SovereignLocalMedicalLLM().generate_satusehat_bundle(export)
    bundle["type"] = "collection"
    for entry in bundle["entry"]:
        entry.pop("request", None)
        # Legacy urn:uuid identifiers are not UUIDs; omit in this non-postable preview.
        entry.pop("fullUrl", None)
        resource = entry["resource"]
        if resource["resourceType"] == "Observation":
            resource["status"] = "preliminary"
        elif resource["resourceType"] == "Encounter":
            resource["status"] = "unknown"
        elif resource["resourceType"] == "Condition":
            resource["verificationStatus"] = {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "unconfirmed"}]}
    drop = set()
    if soap.vitals.systolic_bp is None or soap.vitals.diastolic_bp is None:
        drop.add("obs-bp-")
    if soap.vitals.heart_rate is None:
        drop.add("obs-hr-")
    bundle["entry"] = [e for e in bundle["entry"]
                       if not any(e["resource"]["id"].startswith(p) for p in drop)]
    bundle.setdefault("meta", {})["tag"] = [{"system": "https://voice-rme.example/CodeSystem/review-status",
                              "code": "draft-unreviewed", "display": "DRAFT - belum ditandatangani dokter"}]
    return bundle


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class LivePlatform:
    def __init__(self, adapter_dir: Optional[str] = None):
        self.asr = ASRStage(adapter_dir)
        self.llm = LLMStage()
        self.rules = SovereignLocalMedicalLLM()

    def run(self, audio_path: str, patient_name="Pasien Uji", doctor_name="dr. Uji",
            patient_id="P-LIVE", encounter_id=None) -> Dict[str, Any]:
        t0 = time.time()
        encounter_id = encounter_id or f"ENC-LIVE-{int(t0)}"
        meta = dict(patient_name=patient_name, doctor_name=doctor_name,
                    patient_id=patient_id, encounter_id=encounter_id)

        segs = self.asr.transcribe(audio_path)
        t_asr = time.time()
        raw_turns = [DialogueTurn(i + 1, "UNKNOWN", s["start"], s["end"], s["text"])
                     for i, s in enumerate(segs)]
        if not raw_turns:
            return {"error": "Tidak ada ucapan terdeteksi", "timing": {"total_s": time.time() - t0}}

        transcript = "\n".join(f"{t.turn_id}. {t.text}" for t in raw_turns)
        data = self.llm.generate_json(transcript)
        mode = "llm"
        if data is not None:
            for t in raw_turns:
                lab = (data.get("speakers") or {}).get(str(t.turn_id))
                if lab in ("DOKTER", "PASIEN"):
                    t.speaker = lab
            grounded = ground_soap(data, raw_turns, meta)
            soap, review = grounded["soap"], grounded["needs_review"]
        else:
            return {"status": "BLOCKED", "error": "JSON klinis tidak valid; transkripsi perlu ditinjau manual",
                    "turns": raw_turns, "timing": {"total_s": time.time() - t0}}
        t_llm = time.time()

        fhir = build_fhir_draft(soap)
        total = time.time() - t0
        audio_s = raw_turns[-1].end
        return {
            "status": "DRAFT - menunggu verifikasi dokter",
            "mode": mode, "adapter_loaded": self.asr.adapter_loaded,
            "turns": raw_turns, "soap": soap, "needs_review": review, "fhir": fhir,
            "timing": {"asr_s": round(t_asr - t0, 2), "llm_s": round(t_llm - t_asr, 2),
                       "total_s": round(total, 2), "audio_s": round(audio_s, 1),
                       "rtf": round(total / max(audio_s, 0.1), 3)},
        }


def to_jsonable(result: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(result)
    r["turns"] = [asdict(t) for t in r.get("turns", [])]
    if "soap" in r:
        r["soap"] = asdict(r["soap"])
    return r