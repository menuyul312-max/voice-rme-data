"""RME ASR v3: shared data contracts and transparent metrics. No clinical auto-correction."""
import collections, hashlib, json, math, re, tarfile, unicodedata
from pathlib import Path
import numpy as np

SR = 16000
BASE = 'openai/whisper-large-v3-turbo'

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(p)

def normalize(text):
    # Keep decimals, dose separators, percent and negation. No synonym/filler rewriting.
    t = unicodedata.normalize('NFKC', str(text)).lower()
    t = re.sub(r'(?<=\d),(?=\d)', '.', t)
    t = re.sub(r'(?<!\d)\.|\.(?!\d)', ' ', t)
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s./%+\-]', ' ', t)).strip()

def extract_safe(archive, dest):
    dest = Path(dest).resolve(); dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        for m in t:
            p = (dest/m.name).resolve()
            if not p.is_relative_to(dest) or not (m.isfile() or m.isdir()):
                raise ValueError('Unsafe archive member: ' + m.name)
            t.extract(m, dest, filter='data')

def read_records(path):
    p = Path(path)
    if p.suffix == '.jsonl': return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    if p.suffix == '.csv':
        import csv
        return list(csv.DictReader(p.open()))
    obj = json.loads(p.read_text())
    if isinstance(obj, list): return obj
    for key in ('data', 'records', 'samples', 'rows'):
        if isinstance(obj.get(key), list): return obj[key]
    raise ValueError(f'Unsupported manifest structure: {p}')

def canonical_rows(manifests, root):
    root = Path(root).resolve()
    files = collections.defaultdict(list)
    for p in root.rglob('*'):
        if p.suffix.lower() in ('.wav','.mp3','.flac','.ogg'): files[p.name].append(p)
    rows = []; seen = set()
    for manifest in manifests:
        for x in read_records(manifest):
            name = x.get('audio_file') or x.get('audio_path') or x.get('path') or x.get('file')
            text = x.get('raw_text', x.get('text', x.get('transcript', x.get('ref'))))
            if isinstance(text,str): text=re.sub(r'</?[A-Za-z][^>]*>', '', text)  # remove corpus language tags only
            if not name or not isinstance(text,str): raise ValueError('Manifest needs audio_file and text')
            name = str(name); candidates = [root/name, Path(manifest).parent/name]
            found = [p.resolve() for p in candidates if p.is_file() and p.resolve().is_relative_to(root)]
            if not found:
                found = files.get(Path(name).name, [])
                if len(found) != 1: raise ValueError('Missing/ambiguous audio: ' + name)
            p = found[0]; rel = str(p.relative_to(root))
            if rel in seen: raise ValueError('Duplicate manifest row: ' + rel)
            seen.add(rel)
            stem = p.stem
            syn = stem.startswith('SYN_') or str(x.get('source','')).lower() in ('synthetic','tts')
            source = 'synthetic' if syn else 'medisco' if stem.startswith('medisco_') else 'video' if re.match(r'(lvl3|clean)_chk_',stem) else 'fleurs' if stem.startswith('fleurs_') else str(x.get('source','unknown'))
            video = str(x.get('video_id') or '')
            if not video:
                match = re.match(r'(?:lvl3|clean)_chk_\d+_(.{11})$', stem)
                if match: video = match[1]  # YouTube IDs may contain underscores.
            speaker = str(x.get('speaker_id') or '')
            if source == 'medisco' and not speaker:
                match = re.search(r'(?:female|male)(?:-\d+)?',stem)
                if match: speaker = 'medisco:' + match[0]
            # Synthetic template family is training only; never use it as a human speaker ID.
            group = str(x.get('session_id') or video or speaker or x.get('group_id') or '')
            if syn: group = 'synthetic'
            if source=='fleurs':group='fleurs_unknown_speakers'  # per-clip IDs do not establish speaker independence
            if not group: raise ValueError('Missing recording/speaker/session group: ' + rel)
            declared = next((v for v in Path(rel).parts if v in ('train','dev','test')), 'unknown')
            clean=x.get('clean_text',text)
            changes=critical_proxy(text,clean)
            review_flags=[f'clean_changed_{k}' for k,v in changes.items() if v]
            if normalize(text)!=normalize(clean):review_flags.append('clean_lexical_change')
            rows.append(dict(id=rel, audio_file=rel, ref=text.strip(), source=source, synthetic=syn,
                training_only=syn or source=='fleurs',source_detail=x.get('source','unknown'),
                clean_text=clean,label_review_flags=review_flags,reference_policy='source_raw_without_language_tags',
                video_id=video, speaker_id=speaker, session_id=str(x.get('session_id') or ''),
                patient_id=str(x.get('patient_id') or ''), group_id=group, declared_split=declared,
                category=x.get('clinical_category',x.get('topic','unknown')),
                declared_seconds=float(x.get('duration',x.get('duration_seconds',0)) or 0),
                reference_verified=x.get('reference_verified') is True))
    if not rows: raise ValueError('Empty dataset')
    return sorted(rows,key=lambda r:r['id'])

def inspect_audio(row, root):
    import soundfile as sf
    r = dict(row); issues = []
    if any(f in row.get('label_review_flags',[]) for f in ('clean_changed_number','clean_changed_unit','clean_changed_negation')):
        issues.append('label_critical_disagreement_requires_audio_review')
    try:
        path = Path(root)/r['audio_file']
        x, sr = sf.read(path, dtype='float32', always_2d=True)
        r.update(sr=sr, channels=x.shape[1], seconds=len(x)/sr,
                 audio_sha256=file_hash(path), pcm_sha256=hashlib.sha256(x.tobytes()).hexdigest())
        r['peak'] = float(np.max(np.abs(x))) if len(x) else 0.
        r['rms'] = float(np.sqrt(np.mean(x*x))) if len(x) else 0.
        r['clip_fraction'] = float(np.mean(np.abs(x)>=.999)) if len(x) else 0.
        if not np.isfinite(x).all(): issues.append('nonfinite')
        if not 0.2 <= r['seconds'] <= 30.: issues.append('duration_out_of_window')
        if r.get('declared_seconds',0)>0 and abs(r['seconds']-r['declared_seconds'])>max(.5,.1*r['declared_seconds']):
            issues.append('duration_metadata_mismatch_requires_review')
        if r['rms'] < 1e-5 and r['ref']: issues.append('silent_with_label')
        if not r['ref']: issues.append('empty_label_requires_nonspeech_verification')
        r['warnings'] = []
        wps = len(normalize(r['ref']).split())/max(r['seconds'],.001)
        if wps < .6 or wps > 4.5: r['warnings'].append('unusual_label_rate')
        if r['clip_fraction'] > .001: r['warnings'].append('clipping')
        words = normalize(r['ref']).split()
        if len(words)>10 and len(set(words))/len(words)<.25: r['warnings'].append('repetition')
    except Exception as e: issues.append('decode_error:' + type(e).__name__)
    r['reject_reasons'] = issues
    return r

def grouped_split(rows, seed=13):
    """Connected components prevent shared recordings/speakers/patients/audio in different splits."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    seen = {}
    for i,r in enumerate(rows):
        for key in ('group_id','video_id','speaker_id','session_id','patient_id','pcm_sha256'):
            val=r.get(key)
            if not val: continue
            k=(key,val)
            if k in seen: parent[find(i)] = find(seen[k])
            seen[k]=i
    members=collections.defaultdict(list)
    for i,r in enumerate(rows): members[find(i)].append(r)
    out=[]
    for group in members.values():
        identities=sorted({(key,str(r[key])) for r in group for key in ('group_id','video_id','speaker_id','session_id','patient_id') if r.get(key)})
        gid=digest(identities)  # stable when clips within the same known source are added/removed
        u=int(digest([seed,gid])[:12],16)/16**12
        split='train' if any(r.get('training_only',r['synthetic']) for r in group) else 'test' if u<.1 else 'dev' if u<.2 else 'train'
        out.extend(dict(r,split=split,cluster_id=gid) for r in group)
    # Ensure each sufficiently large non-synthetic source has at least one group per split.
    # This is metadata-only and deterministic; never use model scores to choose a split.
    strata=collections.defaultdict(dict)
    for r in out:
        if not r.get('training_only',r['synthetic']):strata[r['source'] if 'source' in r else 'unknown'][r['cluster_id']]=r['split']
    for source,assign in strata.items():
        if len(assign)<3:continue
        for needed in ('train','dev','test'):
            counts=collections.Counter(assign.values())
            if counts[needed]:continue
            candidates=sorted(k for k,v in assign.items() if counts[v]>1)
            chosen=candidates[0];assign[chosen]=needed
        for r in out:
            if r['cluster_id'] in assign:r['split']=assign[r['cluster_id']]
    # Exact duplicates kept once per partition, train never learns held-out long references.
    refs={s:{normalize(r['ref']) for r in out if r['split']==s and len(normalize(r['ref']).split())>=8} for s in ('dev','test')}
    kept=[]; dropped=[]; pcm_seen=set()
    for r in sorted(out,key=lambda r:({'test':0,'dev':1,'train':2}[r['split']],r['id'])):
        ref=normalize(r['ref']); reason=None
        if r['split']=='train' and ref in refs['dev']|refs['test']: reason='heldout_exact_text'
        if r['split']=='dev' and ref in refs['test']: reason='test_exact_text'
        pcm=(r.get('sr'),r.get('pcm_sha256'))
        if pcm in pcm_seen: reason='duplicate_pcm'
        pcm_seen.add(pcm)
        (dropped if reason else kept).append(dict(r,drop_reason=reason) if reason else r)
    for s in ('train','dev','test'):
        if not any(r['split']==s for r in kept): raise ValueError('Empty split; insufficient independent groups')
    return kept,dropped

def edit_counts(a,b):
    # deterministic S/D/I counts, O(len(a)*len(b)) time and O(len(b)) memory
    prev=[(j,0,0,j) for j in range(len(b)+1)]
    for i,x in enumerate(a,1):
        cur=[(i,0,i,0)]
        for j,y in enumerate(b,1):
            c,s,d,k=prev[j-1]; sub=(c+(x!=y),s+(x!=y),d,k)
            c,s,d,k=prev[j]; delete=(c+1,s,d+1,k)
            c,s,d,k=cur[-1]; insert=(c+1,s,d,k+1)
            cur.append(min((sub,delete,insert), key=lambda z:z[0]))
        prev=cur
    _,s,d,i=prev[-1]; return int(s),int(d),int(i)

def score(ref,hyp):
    a,b=normalize(ref),normalize(hyp)
    s,d,i=edit_counts(a.split(),b.split()); cs,cd,ci=edit_counts(list(a),list(b))
    return dict(s=s,d=d,i=i,n=len(a.split()),err=s+d+i,chars=len(a),char_err=cs+cd+ci)

def paired_cluster_ci(rows, a, b, B=2000, seed=13):
    if set(a)!=set(b) or set(a)!={r['id'] for r in rows}: raise ValueError('Prediction coverage mismatch')
    clusters=collections.defaultdict(lambda:np.zeros(3))
    for r in rows:
        sa,sb=score(r['ref'],a[r['id']]),score(r['ref'],b[r['id']])
        clusters[r['cluster_id']]+=np.array([sa['err'],sb['err'],sa['n']])
    v=np.array(list(clusters.values())); total=v.sum(0)
    if not total[2]: raise ValueError('No reference words')
    result={'delta_wer_pp':float(100*(total[0]-total[1])/total[2]),'clusters':len(v),'unit':'connected_source_group','ci95':None}
    if len(v)<2: return result
    rng=np.random.default_rng(seed); boot=[]
    for _ in range(B):
        t=v[rng.integers(len(v),size=len(v))].sum(0)
        if t[2]>0: boot.append(100*(t[0]-t[1])/t[2])
    result['ci95']=np.quantile(boot,[.025,.975]).tolist();return result

def corpus_metrics(rows, hyps):
    if set(hyps)!={r['id'] for r in rows} or any(not isinstance(h,str) for h in hyps.values()):
        raise ValueError('All planned clips must have predictions; failures cannot disappear')
    totals=collections.Counter()
    for r in rows: totals.update(score(r['ref'],hyps[r['id']]))
    return dict(totals,wer=100*totals['err']/totals['n'] if totals['n'] else None,
                cer=100*totals['char_err']/totals['chars'] if totals['chars'] else None,clips=len(rows))

def critical_proxy(ref,hyp):
    # Screening only: no semantic scope/association inference or correctness certification.
    patterns={'number':r'\b\d+(?:[.,]\d+)?(?:/\d+)?\b',
              'unit':r'\b(?:mg|ml|mcg|kg|mmhg|gram|miligram|mililiter|persen|unit)\b',
              'negation':r'\b(?:tidak|tanpa|bukan|disangkal|negatif|belum)\b'}
    return {k:collections.Counter(re.findall(p,normalize(ref))) != collections.Counter(re.findall(p,normalize(hyp))) for k,p in patterns.items()}

def release_status(evidence):
    required=('independent_real_clinical_test','human_gold_references','critical_error_review',
              'silent_noise_test','long_form_test','latency_sla','subgroup_review','data_rights_review')
    missing=[k for k in required if evidence.get(k) is not True]
    return {'status':'NOT_READY' if missing else 'CANDIDATE_FOR_CONTROLLED_PILOT', 'missing':missing,
            'note':'Not automatic enterprise certification; clinician signs each RME draft.'}

def term_metrics(rows,hyps,terms):
    """Exact lexical occurrence scoring, not entity linking or semantic correctness."""
    terms=sorted({normalize(t) for t in terms if normalize(t)})
    if not terms:return {'status':'NOT_MEASURED','reason':'No clinician-approved lexicon supplied'}
    tp=fp=fn=0
    for r in rows:
        a,b=normalize(r['ref']),normalize(hyps[r['id']])
        for t in terms:
            p=r'(?<!\w)'+re.escape(t)+r'(?!\w)'
            na,nb=len(re.findall(p,a)),len(re.findall(p,b))
            tp+=min(na,nb);fp+=max(0,nb-na);fn+=max(0,na-nb)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,'lexicon_size':len(terms),
            'note':'Lexical matches only; overlapping aliases must be curated before use.'}


def quarantine_conflicts(rows):
    groups=collections.defaultdict(list)
    for r in rows:
        if r.get('pcm_sha256'):groups[(r.get('sr'),r['pcm_sha256'])].append(r)
    for group in groups.values():
        if len({normalize(r['ref']) for r in group})>1:
            for r in group:
                reason='identical_audio_conflicting_references'
                if reason not in r['reject_reasons']:r['reject_reasons'].append(reason)
    return rows
