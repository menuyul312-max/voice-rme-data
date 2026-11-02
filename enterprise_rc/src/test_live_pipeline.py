"""CPU-only tests for grounding logic (no torch / GPU needed): python -m pytest -q test_live_pipeline.py"""
from local_med_llm import DialogueTurn
from live_pipeline_t4 import ground_soap, build_fhir_draft, parse_json_loose, map_to_ontology

TURNS = [DialogueTurn(1, "PASIEN", 0, 3, "mata kanan saya merah dan berair"),
         DialogueTurn(2, "DOKTER", 3.5, 8, "ini konjungtivitis, saya resepkan kloramfenikol tetes mata")]
META = dict(patient_id="P1", patient_name="X", doctor_name="dr. Y", encounter_id="E1")


def test_no_invented_vitals_or_codes():
    data = {"chief_complaint": "mata merah", "vitals": {},
            "diagnoses": [{"term": "konjungtivitis", "evidence_turn": 2},
                          {"term": "penyakit antah berantah", "evidence_turn": 1}],
            "medications": [{"term": "obat ajaib xyz", "evidence_turn": 2}]}
    out = ground_soap(data, TURNS, META)
    soap = out["soap"]
    assert soap.vitals.heart_rate is None and soap.vitals.blood_pressure is None
    assert soap.diagnoses[0].icd10_code == "H10.1"
    assert soap.diagnoses[1].icd10_code == "UNMAPPED"
    assert soap.medications[0].kfa_code == "UNMAPPED"
    assert len(out["needs_review"]) >= 2
    types = [e["resource"]["resourceType"] for e in build_fhir_draft(soap)["entry"]]
    assert "Observation" not in types


def test_evidence_must_reference_real_turn():
    out = ground_soap({"diagnoses": [{"term": "konjungtivitis", "evidence_turn": 99}]}, TURNS, META)
    assert out["soap"].diagnoses[0].evidence is None


def test_spoken_vitals_kept_and_json_parser():
    d = parse_json_loose('```json\n{"vitals": {"systolic": 130, "diastolic": 85, "heart_rate": 80}}\n```')
    spoken = TURNS + [DialogueTurn(3, "DOKTER", 8, 10, "tekanan darah 130/85; nadi 80")]
    soap = ground_soap(d, spoken, META)["soap"]
    types = [e["resource"]["resourceType"] for e in build_fhir_draft(soap)["entry"]]
    assert soap.vitals.blood_pressure == "130/85 mmHg" and types.count("Observation") == 2
    assert parse_json_loose("bukan json") is None


def test_ontology_mapping():
    assert map_to_ontology("medications", "Kloramfenikol tetes")["kfa"] == "93000123"
    assert map_to_ontology("diagnoses", "") is None


def test_colloquial_ontology_mapping():
    assert map_to_ontology("diagnoses", "pegel linu")["code"] == "M79.1"
    assert map_to_ontology("diagnoses", "kesemutan")["code"] == "R20.2"
    assert map_to_ontology("diagnoses", "mriang dan demam")["code"] == "R50.9"
    assert map_to_ontology("diagnoses", "perut begah masuk angin")["code"] == "K30"
    assert map_to_ontology("medications", "parasetamol tablet")["kfa"] == "93006601"
    assert map_to_ontology("medications", "ibuprofen 400")["kfa"] == "93006602"


def test_robust_json_repairs_malformed_llm_outputs():
    # 1. Trailing comma
    d1 = parse_json_loose('{"chief_complaint": "pegel linu", "vitals": {"systolic": 120,},}')
    assert d1 is not None and d1["chief_complaint"] == "pegel linu" and d1["vitals"]["systolic"] == 120

    # 2. Single quotes
    d2 = parse_json_loose("{'chief_complaint': 'mriang', 'vitals': {'heart_rate': 78}}")
    assert d2 is not None and d2["chief_complaint"] == "mriang" and d2["vitals"]["heart_rate"] == 78

    # 3. Truncated unclosed braces
    d3 = parse_json_loose('{"chief_complaint": "kesemutan", "diagnoses": [{"term": "parestesia", "evidence_turn": 1}')
    assert d3 is not None and d3["chief_complaint"] == "kesemutan"

    # 4. Completely broken syntax recovered via regex fallback
    d4 = parse_json_loose('Berikut adalah hasil SOAP: "chief_complaint": "nyut-nyutan", "systolic": 130, "diagnoses": [{"term": "sefalgia"}]')
    assert d4 is not None and d4["chief_complaint"] == "nyut-nyutan" and d4["vitals"]["systolic"] == 130