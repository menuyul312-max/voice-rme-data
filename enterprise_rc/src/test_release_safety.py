from clinical_safety import extract_supported_vitals, filter_vitals
from release_policy import release_verdict
from local_med_llm import VitalSigns, SovereignLocalMedicalLLM, DialogueTurn
from live_pipeline_t4 import ground_soap, build_fhir_draft
from benchmark_real import vitals_unsupported, paired_bootstrap
from rme_asr_v2 import corpus_wer

META=dict(patient_id='P1', patient_name='X', doctor_name='Y', encounter_id='E1')

def test_vitals_no_default_or_number_collision():
    assert all(v is None for v in vars(VitalSigns()).values())
    assert not extract_supported_vitals('usia 80 tahun, berat 120 kg, tinggi 170 cm')
    assert SovereignLocalMedicalLLM().extract_vitals('tinggi 170').blood_pressure is None
    v, bad=filter_vitals({'heart_rate':80,'systolic':120,'diastolic':80}, 'usia 80 tahun berat 120 kg')
    assert not v and len(bad)==3

def test_vitals_context_negation_ambiguity_and_decimals():
    assert extract_supported_vitals('tekanan darah 130/85; nadi 80; suhu 36,7; rr 18; saturasi 98') == dict(systolic=130,diastolic=85,heart_rate=80,temperature=36.7,respiratory_rate=18,spo2=98)
    assert not extract_supported_vitals('nadi 80; nadi 90')
    assert not extract_supported_vitals('bukan nadi 80; suhu 90')
    assert filter_vitals({'heart_rate':True,'temperature':'NaN'}, 'nadi 80; suhu 36.7')[0]=={}
    assert not extract_supported_vitals('nadi 8000; suhu 36.789')

def test_live_drops_unsupported_vitals_and_regimen_defaults():
    turns=[DialogueTurn(1,'UNKNOWN',0,5,'konjungtivitis, kloramfenikol tetes mata')]
    data={'vitals':{'heart_rate':80},'medications':[{'term':'kloramfenikol','evidence_turn':1}], 'diagnoses':[{'term':'konjungtivitis','evidence_turn':1}]}
    out=ground_soap(data,turns,META); soap=out['soap']
    assert soap.vitals.heart_rate is None and out['needs_review']
    med=soap.medications[0]
    assert med.dosage==med.frequency=='Tidak disebutkan' and med.duration_days==0
    bundle=build_fhir_draft(soap)
    assert bundle['type']=='collection'
    assert all('request' not in e for e in bundle['entry'])
    assert not any(e['resource']['resourceType'] in ('MedicationRequest','Procedure','Observation') for e in bundle['entry'])

def test_benchmark_vitals_requires_type_context():
    soap=ground_soap({},[],META)['soap']; soap.vitals.heart_rate=80
    assert vitals_unsupported(soap,'usia 80')==['heart_rate']
    assert vitals_unsupported(soap,'nadi 80')==[]

def test_release_policy_failure_missing_and_unknown():
    gates=[dict(gerbang=f'G{i}',status='LULUS') for i in range(1,8)]
    assert 'LULUS GERBANG' in release_verdict(gates)
    gates[-1]['status']='GAGAL'
    assert 'BELUM LAYAK' in release_verdict(gates)
    gates[-1]['status']='N/A'
    assert 'BELUM DAPAT' in release_verdict(gates)
    assert 'BELUM DAPAT' in release_verdict([])
    try: release_verdict(gates+[gates[0]])
    except ValueError: pass
    else: raise AssertionError('duplicate gates accepted')

def test_alignment_errors_rejected():
    try: paired_bootstrap([dict(ref='a',hyp='a')],[dict(ref='b',hyp='b')])
    except ValueError: pass
    else: raise AssertionError('mismatched refs accepted')
    try: corpus_wer(['a'],[])
    except ValueError: pass
    else: raise AssertionError('truncated hypotheses accepted')

def test_cluster_bootstrap_resamples_recordings():
    a=[dict(chunk_id=str(i),video_id=str(i//2),ref='mata merah',hyp='mata') for i in range(8)]
    b=[dict(x,hyp='mata merah') for x in a]
    r=paired_bootstrap(a,b,B=200)
    assert r['n_clusters']==4 and r['bootstrap_unit']=='video'
    assert r['delta_pp']==50 and r['delta_ci95']==[50,50]

def test_invalid_clinical_shapes_blocked():
    from clinical_safety import validate_payload
    for payload in [[], {'vitals': '120/80'}, {'diagnoses': 'demam'},
                    {'medications': [None]}, {'diagnoses': [{'term': ''}]},
                    {'chief_complaint': ['nyeri']}]:
        try: validate_payload(payload)
        except ValueError: pass
        else: raise AssertionError(f'invalid clinical shape accepted: {payload}')
