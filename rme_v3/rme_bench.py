"""Reference-blind ASR inference, fingerprinted resume, explicit coverage and timing."""
import gc,time
from pathlib import Path
import numpy as np
from rme_core import BASE,SR,digest,file_hash,write_json
from rme_train import load_wave

def model_fingerprint(path):
    p=Path(path)
    files=sorted(f for f in p.rglob('*') if f.is_file() and f.suffix in ('.json','.safetensors','.bin'))
    if not any(f.suffix in ('.safetensors','.bin') for f in files): raise ValueError('No model weights: '+str(p))
    return digest([(str(f.relative_to(p)),file_hash(f)) for f in files])

def infer(rows,root,target,revision,out,decode,batch_size=2):
    import torch
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    from peft import PeftModel
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    is_base=target==BASE
    adapter=not is_base and (Path(target)/'adapter_config.json').exists()
    fingerprint=revision if is_base else model_fingerprint(target)
    run_id=digest({'model':fingerprint,'revision':revision,'decode':decode,
                   'data':[(r['id'],r['audio_sha256']) for r in rows],'batch_size':batch_size,
                   'engine':'transformers-4.57.3-fp16-sdpa','torch':torch.__version__,
                   'code':[file_hash(Path(__file__).parent/x) for x in ('rme_core.py','rme_train.py','rme_bench.py')],
                   'gpu':torch.cuda.get_device_name(0)})
    cache=out/(run_id+'.json')
    data=__import__('json').loads(cache.read_text()) if cache.exists() else {'run_id':run_id,'records':{},'complete':False}
    if data['complete']:
        if set(data['records'])!={r['id'] for r in rows}: raise ValueError('Corrupt cache coverage')
        return data
    proc=WhisperProcessor.from_pretrained(BASE if is_base or adapter else target,**({'revision':revision} if is_base or adapter else {}))
    model=WhisperForConditionalGeneration.from_pretrained(BASE if is_base or adapter else target,
        **({'revision':revision} if is_base or adapter else {}),torch_dtype=torch.float16,attn_implementation='sdpa')
    if adapter:model=PeftModel.from_pretrained(model,target).merge_and_unload()
    model=model.to('cuda').eval();model.config.use_cache=True;model.config.apply_spec_augment=False
    model.generation_config.forced_decoder_ids=None
    def predict(waves):
        batch=proc.feature_extractor(waves,sampling_rate=SR,return_tensors='pt',return_attention_mask=True)
        batch={k:v.to('cuda',dtype=torch.float16 if k=='input_features' else v.dtype) for k,v in batch.items()}
        with torch.inference_mode():tokens=model.generate(**batch,**decode)
        return proc.batch_decode(tokens,skip_special_tokens=True),tokens
    pending=[r for r in rows if r['id'] not in data['records']]
    if pending:predict([load_wave(Path(root)/pending[0]['audio_file'])])
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    for start in range(0,len(pending),batch_size):
        part=pending[start:start+batch_size]
        torch.cuda.synchronize();t0=time.perf_counter()
        waves=[load_wave(Path(root)/r['audio_file']) for r in part]
        texts,tokens=predict(waves)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t0
        for r,text,seq in zip(part,texts,tokens):
            # Timing is batch wall time, never advertised as single-request latency.
            capped=len(seq)>=decode['max_length'] and int(seq[-1]) not in (proc.tokenizer.eos_token_id,proc.tokenizer.pad_token_id)
            data['records'][r['id']]={'hyp':text,'seconds':r['seconds'],'batch_s':elapsed,
                'batch_n':len(part),'token_cap_hit':capped,'audio_sha256':r['audio_sha256']}
        write_json(cache,data)
    # Batch=1 latency sample, deterministic and separate from throughput. Includes decode/preprocessing.
    lat=[];rtfs=[]
    sample=sorted(rows,key=lambda r:digest(r['id']))[:min(100,len(rows))]
    for r in sample:
        torch.cuda.synchronize();t0=time.perf_counter()
        predict([load_wave(Path(root)/r['audio_file'])])
        torch.cuda.synchronize();dt=time.perf_counter()-t0
        lat.append(dt);rtfs.append(dt/r['seconds'])
    wall=sum(v['batch_s']/v['batch_n'] for v in data['records'].values())
    seconds=sum(r['seconds'] for r in rows)
    data.update(complete=True,timing={'warm_model_batch1_n':len(lat),'latency_p50_s':float(np.quantile(lat,.5)),
        'latency_p95_s':float(np.quantile(lat,.95)),'batch1_rtf_p95':float(np.quantile(rtfs,.95)),
        'batch_size':batch_size,'throughput_rtf':wall/seconds,'peak_allocated_gb':torch.cuda.max_memory_allocated()/2**30,
        'note':'Local audio decode + feature extraction + generation; excludes model load, queue, network, RME editing.'})
    write_json(cache,data)
    del model,proc;gc.collect();torch.cuda.empty_cache()
    return data

def challenge_infer(manifest, root, target, revision, out):
    """Optional independent real silence/noise/long-form cohort. HF overlap-chunked ASR, no raw truncation."""
    import json,math,soundfile as sf,torch
    from scipy.signal import resample_poly
    from transformers import WhisperProcessor,WhisperForConditionalGeneration,pipeline
    rows=json.loads(Path(manifest).read_text())
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty/duplicate challenge IDs')
    # Caller supplies independent approved cohort; this function cannot prove independence.
    proc=WhisperProcessor.from_pretrained(target)
    model=WhisperForConditionalGeneration.from_pretrained(target,torch_dtype=torch.float16,attn_implementation='sdpa')
    model.config.use_cache=True;model.config.apply_spec_augment=False;model.generation_config.forced_decoder_ids=None
    pipe=pipeline('automatic-speech-recognition',model=model,tokenizer=proc.tokenizer,
                  feature_extractor=proc.feature_extractor,device=0)
    results=[]
    for r in rows:
        if r['kind'] not in ('silence','noise','long_form','clinical'):raise ValueError('Unknown challenge kind')
        p=(Path(root)/r['audio_file']).resolve()
        if not p.is_relative_to(Path(root).resolve()):raise ValueError('Audio outside challenge root')
        x,sr=sf.read(p,dtype='float32',always_2d=True);x=x.mean(1)
        if not len(x) or not np.isfinite(x).all():raise ValueError('Invalid challenge audio')
        dur=len(x)/sr
        if r['kind'] in ('silence','noise') and r['ref'].strip():raise ValueError('Nonspeech reference must be empty')
        if r['kind']=='long_form' and dur<=30:raise ValueError('long_form must exceed 30 s')
        if sr!=SR:
            g=math.gcd(sr,SR);x=resample_poly(x,SR//g,sr//g)
        torch.cuda.synchronize();t=time.perf_counter()
        h=pipe({'raw':x,'sampling_rate':SR},chunk_length_s=30,stride_length_s=5,return_timestamps=True,
               generate_kwargs={'language':'indonesian','task':'transcribe','num_beams':1,'do_sample':False,'max_length':448})
        torch.cuda.synchronize()
        results.append(dict(r,hyp=h['text'],seconds=dur,wall_s=time.perf_counter()-t,audio_sha256=file_hash(p)))
    write_json(out,{'records':results,'model':model_fingerprint(target),'base_revision':revision,
                    'protocol':'HF overlapping chunks 30s/5s; independent diagnostic; no streaming claim'})
    del pipe,model,proc;gc.collect();torch.cuda.empty_cache()
    return results
