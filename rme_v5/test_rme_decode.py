from rme_decode import loop_flags,cap_hit,choose_retry

def test_444_word_regression():
    assert 'repeated_phrase' in loop_flags('hai '*444)
    assert 'repeated_phrase' in loop_flags('selamat pagi '*80)

def test_valid_negation_not_removed():
    assert not loop_flags('tidak tidak dok, bukan lima, nol koma lima miligram')

def test_forced_eos_at_limit_is_cap():
    assert cap_hit([1]*447+[2],2,2,448)
    assert not cap_hit([1,3,2]+[2]*445,2,2,448)
    assert cap_hit([1]*448,2,2,448)

def test_retry_is_reference_blind_and_failed_text_preserved():
    assert choose_retry('hai '*444,'terima kasih',['loop'],[])=='terima kasih'
    assert choose_retry('raw','also bad',['loop'],['loop'])=='raw'
    assert choose_retry('good','different',[],[])=='good'

def test_stable_source_group_when_clip_added():
    from rme_core import grouped_split
    rows=[dict(id=str(i),group_id=str(i),synthetic=False,pcm_sha256=str(i),sr=16000,ref=str(i)) for i in range(100)]
    a,_=grouped_split(rows)
    more=dict(rows[1],id='extra',pcm_sha256='extra',ref='extra')
    b,_=grouped_split(rows+[more])
    assert {r['id']:r['split'] for r in a}=={r['id']:r['split'] for r in b if r['id']!='extra'}

def test_generate_retry_records_original(monkeypatch):
    import sys,types,contextlib,numpy as np
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(inference_mode=contextlib.nullcontext))
    from rme_decode import guarded_generate,DECODE
    class Model:
        calls=0
        def generate(self,**kw):
            self.calls+=1
            return np.array([[1,5,2]]) if self.calls==1 else np.array([[1,6,2]])
    class Proc:
        tokenizer=types.SimpleNamespace(eos_token_id=2,pad_token_id=2)
        def batch_decode(self,seq,**kw):return ['hai '*444 if seq[0,1]==5 else 'terima kasih']
    m=Model();texts,tokens,meta=guarded_generate(m,Proc(),{'input_features':np.zeros((1,2))},DECODE)
    assert texts==['terima kasih'] and m.calls==2
    assert meta[0]['raw_primary']=='hai '*444 and meta[0]['retry_used'] and not meta[0]['needs_review']

def test_failed_retry_remains_visible(monkeypatch):
    import sys,types,contextlib,numpy as np
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(inference_mode=contextlib.nullcontext))
    from rme_decode import guarded_generate
    model=types.SimpleNamespace(generate=lambda **kw:np.array([[1,5,2]]))
    proc=types.SimpleNamespace(tokenizer=types.SimpleNamespace(eos_token_id=2,pad_token_id=2),
        batch_decode=lambda seq,**kw:['hai '*444])
    text,_,meta=guarded_generate(model,proc,{'input_features':np.zeros((1,2))})
    assert text==['hai '*444] and meta[0]['needs_review'] and not meta[0]['retry_used']
