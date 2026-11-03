"""Reference-blind generation checks and conservative VAD. No transcript rewriting."""
import re,zlib
import numpy as np
from rme_core import SR,normalize

DECODE={'language':'indonesian','task':'transcribe','num_beams':3,'do_sample':False,
        'max_length':448,'use_cache':True,'no_repeat_ngram_size':6,'forced_eos_token_id':None}

def loop_flags(text):
    words=normalize(text).split();flags=[]
    for n in range(1,7):
        for i in range(max(0,len(words)-n*8+1)):
            block=words[i:i+n]
            if all(words[i+j*n:i+(j+1)*n]==block for j in range(1,8)):
                flags.append('repeated_phrase');break
        if flags:break
    b=text.encode()
    if len(words)>=20 and len(b)/max(1,len(zlib.compress(b)))>2.4:flags.append('high_compression')
    return flags

def cap_hit(tokens,eos,pad,limit=448):
    # Count through FIRST EOS; padding cannot disguise a naturally early ending.
    ids=[int(x) for x in tokens]
    first=next((i for i,x in enumerate(ids[1:],1) if x==eos),len(ids))
    return first>=limit-1 or (len(ids)>=limit and eos not in ids[1:])

def choose_retry(primary,retry,primary_flags,retry_flags):
    return retry if primary_flags and not retry_flags else primary

class SpeechAudit:
    def __init__(self):
        from silero_vad import load_silero_vad
        self.model=load_silero_vad(onnx=True)
    def __call__(self,wave):
        import torch
        from silero_vad import get_speech_timestamps
        spans=get_speech_timestamps(torch.from_numpy(np.asarray(wave,dtype=np.float32)),self.model,
            sampling_rate=SR,threshold=.35,min_speech_duration_ms=100,
            min_silence_duration_ms=500,speech_pad_ms=300)
        return {'speech_seconds':sum(x['end']-x['start'] for x in spans)/SR,
                'spans':spans,'vad_no_speech':not spans}

def guarded_generate(model,proc,batch,decode=None):
    import torch
    cfg=dict(DECODE if decode is None else decode)
    with torch.inference_mode():tokens=model.generate(**batch,**cfg)
    raw=proc.batch_decode(tokens,skip_special_tokens=True)
    texts=list(raw);meta=[]
    for i,(t,seq) in enumerate(zip(raw,tokens)):
        flags=loop_flags(t)
        capped=cap_hit(seq,proc.tokenizer.eos_token_id,proc.tokenizer.pad_token_id,cfg.get('max_length',448))
        if capped:flags.append('generation_limit')
        entry={'raw_primary':t,'primary_flags':flags,'retry_used':False,'token_cap_hit':capped}
        if flags:
            retry_cfg=dict(cfg,num_beams=5,no_repeat_ngram_size=3,repetition_penalty=1.05)
            one={k:v[i:i+1] for k,v in batch.items()}
            with torch.inference_mode():retry_tokens=model.generate(**one,**retry_cfg)
            retry=proc.batch_decode(retry_tokens,skip_special_tokens=True)[0]
            rf=loop_flags(retry)
            if cap_hit(retry_tokens[0],proc.tokenizer.eos_token_id,proc.tokenizer.pad_token_id,retry_cfg.get('max_length',448)):rf.append('generation_limit')
            texts[i]=choose_retry(t,retry,flags,rf)
            entry.update(retry_text=retry,retry_flags=rf,retry_used=not rf)
        entry['needs_review']=bool(loop_flags(texts[i])) or (bool(flags) and not entry['retry_used'])
        meta.append(entry)
    return texts,tokens,meta
