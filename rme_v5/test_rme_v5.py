import csv,json,types
import pytest
from rme_data import *
from rme_profiles import lora_targets,parameter_report
from rme_eval import clinician_review,wilson_upper

def row(i,**kw):
    r=dict(id=str(i),split='train',cluster_id=str(i//4),group_id=str(i//4),synthetic=False,ref='tidak demam',
           audio_sha256=str(i),pcm_sha256=str(i),source='video',reference_verified=False)
    r.update(kw);return r

def test_source_cap_and_synthetic_limit():
    data=[row(i) for i in range(100)]+[row(100+i,synthetic=True,cluster_id='syn') for i in range(100)]
    kept,removed=cap_sources(data,max_per_group=2,synthetic_fraction=.05)
    assert max(__import__('collections').Counter(r['cluster_id'] for r in kept if not r['synthetic']).values())<=2
    assert sum(r['synthetic'] for r in kept)/len(kept)<=.05
    assert len(kept)+len(removed)==len(data)

def test_dev_groups_disjoint():
    a,b=independent_dev([row(i,split='dev') for i in range(100)])
    assert not {r['cluster_id'] for r in a}&{r['cluster_id'] for r in b}
    assert len(a)+len(b)==100
    assert len(balanced_sample(a,11))==11

def test_two_reviewers_and_split_guard(tmp_path):
    p=tmp_path/'r.csv'
    fields=['id','audio_sha256','reviewer_1','reviewer_2','adjudicated','verified_verbatim']
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(dict(zip(fields,['1','1','a','b','true','tidak batuk'])))
    r,info=apply_reviews([row(1)],p)
    assert r[0]['ref']=='tidak batuk' and r[0]['reference_verified']
    with pytest.raises(ValueError):apply_reviews([row(1,split='test')],p)

def test_teacher_never_replaces_label():
    rr=[row(1)];before=rr[0]['ref']
    out=teacher_review(rr,{'1':{'hyp':'demam','decoding':{'needs_review':False}}})
    assert out[0]['priority']==2 and rr[0]['ref']==before

def test_additional_corpus_leak_is_rejected():
    held=row(1,split='test',speaker_id='doctorA')
    new=row(2,reference_verified=True,commercial_use_reviewed=True,speaker_id='doctorA',session_id='new')
    with pytest.raises(ValueError):check_new_training([new],[held])

def test_clinician_review_invalid_denominator(tmp_path):
    p=tmp_path/'c.csv'
    p.write_text('id,adjudicated,reviewer_1,reviewer_2,critical_errors,critical_opportunities\n1,true,a,b,2,1\n')
    with pytest.raises(ValueError):clinician_review(p,['1'])
    assert wilson_upper(0,0) is None and wilson_upper(0,100)>0

def test_lora_top_encoder_only_targets():
    names=[f'model.encoder.layers.{i}.self_attn.q_proj' for i in range(32)]+['model.decoder.layers.0.self_attn.q_proj','model.encoder.conv1']
    fake=types.SimpleNamespace(config=types.SimpleNamespace(encoder_layers=32),named_modules=lambda:[(k,None) for k in names])
    targets=lora_targets(fake,'top8_qv')
    assert len(targets)==9 and not any(f'layers.{i}.' in t for t in targets for i in range(1,24))
    assert lora_targets(fake,'decoder_qv')==['model.decoder.layers.0.self_attn.q_proj']

def test_rapidfuzz_character_count_preserves_existing_metric():
    from rme_core import score,edit_counts,normalize
    for a,b in [('tidak demam','demam'),('0,5 mg','5 mg'),('','obat'),('a b c','a c')]:
        assert score(a,b)['char_err']==sum(edit_counts(list(normalize(a)),list(normalize(b))))

def test_near_duplicate_audit_preserves_rows():
    data=[row(1,ref='pasien mengeluh nyeri dada sejak kemarin disertai batuk kering',split='train'),row(2,ref='pasien mengeluh nyeri dada sejak kemarin disertai batuk kering',split='test')]
    report=near_duplicate_candidates(data)
    assert len(report['pairs'])==1 and len(data)==2

def test_source_cap_can_be_disabled_for_ablation():
    data=[row(i,cluster_id='one') for i in range(500)]
    kept,removed=cap_sources(data,max_per_group=None)
    assert len(kept)==500 and not removed
