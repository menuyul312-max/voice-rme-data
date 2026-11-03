import io,tarfile
import pytest
from rme_core import *

def test_numbers_and_negation_survive():
    assert normalize('Tidak, dosis 0,5 mg; TD 120/80.')=='tidak dosis 0.5 mg td 120/80'
    assert score('0,5 mg','5 mg')['err']==1
    assert score('tidak demam','demam')['d']==1

def test_empty_reference_insertions():
    m=corpus_metrics([{'id':'a','ref':''},{'id':'b','ref':'baik'}],{'a':'obat','b':'baik'})
    assert m['wer']==100 and m['i']==1

def test_no_silent_prediction_omission():
    with pytest.raises(ValueError): corpus_metrics([{'id':'a','ref':'obat'}],{})
    with pytest.raises(ValueError): corpus_metrics([{'id':'a','ref':'obat'}],{'a':None})

def test_cluster_pairing():
    rows=[{'id':str(i),'ref':'tidak demam','cluster_id':str(i//2)} for i in range(10)]
    a={r['id']:'tidak demam' for r in rows};b={r['id']:'demam' for r in rows}
    result=paired_cluster_ci(rows,a,b,B=100)
    assert result['clusters']==5 and result['ci95']==[-50.,-50.]
    with pytest.raises(ValueError): paired_cluster_ci(rows,a,{'x':'demam'})

def test_group_connected_components_and_synthetic_train_only():
    rows=[]
    for i in range(100):
        rows.append(dict(id=str(i),group_id='video'+str(i),speaker_id='speaker'+str(i//2),
             synthetic=False,pcm_sha256=str(i),sr=16000,ref='kata '+str(i)))
    rows.append(dict(id='syn',group_id='syn',speaker_id='',synthetic=True,pcm_sha256='syn',sr=16000,ref='sintetis'))
    out,dropped=grouped_split(rows)
    for key in ('group_id','speaker_id','pcm_sha256'):
        m={}
        for r in out:
            if r.get(key):
                assert m.setdefault(r[key],r['split'])==r['split']
    assert next(r for r in out if r['id']=='syn')['split']=='train'
    assert grouped_split(list(reversed(rows)))[0]==out

def test_dedup_heldout_text():
    rows=[dict(id=str(i),group_id=str(i),synthetic=False,pcm_sha256=str(i),sr=16000,
        ref='kalimat unik '+str(i)) for i in range(100)]
    # Duplicate speech joins same group; only one copy is retained.
    rows[1]['pcm_sha256']=rows[0]['pcm_sha256']
    out,drop=grouped_split(rows)
    assert len(out)==99 and drop[0]['drop_reason']=='duplicate_pcm'

def test_fail_closed_gate():
    assert release_status({})['status']=='NOT_READY'
    assert release_status({'human_gold_references':'True'})['status']=='NOT_READY'

def test_tar_traversal(tmp_path):
    p=tmp_path/'bad.tar'
    with tarfile.open(p,'w') as t:
        m=tarfile.TarInfo('../escape');m.size=1;t.addfile(m,io.BytesIO(b'x'))
    with pytest.raises(ValueError): extract_safe(p,tmp_path/'dest')

def test_youtube_underscore_group(tmp_path):
    audio=tmp_path/'train';audio.mkdir();(audio/'lvl3_chk_001_-qUTxy5a9_E.mp3').write_bytes(b'a')
    manifest=tmp_path/'manifest.json'
    write_json(manifest,[{'audio_file':'lvl3_chk_001_-qUTxy5a9_E.mp3','text':'nyeri'}])
    r=canonical_rows([manifest],tmp_path)[0]
    assert r['video_id']=='-qUTxy5a9_E'

def test_reject_ambiguous_basename(tmp_path):
    for s in ('train','test'):
        p=tmp_path/s;p.mkdir();(p/'same.wav').write_bytes(b'x')
    manifest=tmp_path/'manifest.json';write_json(manifest,[{'audio_file':'same.wav','text':'halo'}])
    with pytest.raises(ValueError):canonical_rows([manifest],tmp_path)

def test_numeric_proxy_not_semantic_certificate():
    assert critical_proxy('dosis 0,5 mg','dosis 5 mg')['number']
    assert critical_proxy('tidak demam','demam')['negation']
    assert not critical_proxy('tidak batuk, demam','batuk, tidak demam')['negation']

def test_term_precision_penalizes_invented_drug():
    m=term_metrics([{'id':'1','ref':'parasetamol'}],{'1':'parasetamol amoksisilin'},['parasetamol','amoksisilin'])
    assert m['recall']==1 and m['precision']==.5
    assert term_metrics([],{},[])['status']=='NOT_MEASURED'

def test_conflicting_same_audio_quarantines_both():
    rows=[{'sr':16000,'pcm_sha256':'same','ref':s,'reject_reasons':[]} for s in ('perut mulas','dada sesak')]
    assert all('identical_audio_conflicting_references' in r['reject_reasons'] for r in quarantine_conflicts(rows))

def test_manifest_uses_original_not_llm_rewrite(tmp_path):
    (tmp_path/'clean_chk_1_abcdefghijk.mp3').touch()
    p=tmp_path/'manifest.json'
    write_json(p,[{'audio_file':'clean_chk_1_abcdefghijk.mp3','raw_text':'tidak demam 0,5 mg',
        'clean_text':'demam 5 mg','group_id':'abcdefghijk','duration':3}])
    r=canonical_rows([p],tmp_path)[0]
    assert r['ref']=='tidak demam 0,5 mg' and r['declared_seconds']==3
    assert 'clean_changed_number' in r['label_review_flags']
    assert 'clean_changed_negation' in r['label_review_flags']

def test_audio_duration_mismatch_quarantined(tmp_path):
    import soundfile as sf
    sf.write(tmp_path/'x.wav',np.ones(16000,dtype=np.float32)*.1,16000)
    r=inspect_audio({'audio_file':'x.wav','ref':'nyeri dada','declared_seconds':2},tmp_path)
    assert 'duration_metadata_mismatch_requires_review' in r['reject_reasons']
    assert r['seconds']==1.
