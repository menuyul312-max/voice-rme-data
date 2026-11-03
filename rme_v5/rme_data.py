"""Versioned data refinement; annotations never inferred from an LLM."""
import collections,csv,json,math
from pathlib import Path
from rme_core import digest,normalize,critical_proxy,file_hash
from rme_decode import loop_flags

def apply_reviews(rows,path,allowed_split='train'):
    if not Path(path).exists():return rows,{'applied':0,'sha256':None}
    reviews=list(csv.DictReader(Path(path).open()));lookup={r['id']:r for r in rows};seen=set();edits={}
    for review in reviews:
        k=review['id']
        if k in seen or k not in lookup:raise ValueError('Duplicate/unknown review id: '+k)
        seen.add(k);r=lookup[k]
        if r['split']!=allowed_split:raise ValueError('Review cannot cross locked split: '+k)
        if review['audio_sha256']!=r['audio_sha256']:raise ValueError('Reviewed audio has changed')
        if review['adjudicated'].lower()!='true' or not review['reviewer_1'] or not review['reviewer_2'] or review['reviewer_1']==review['reviewer_2']:
            raise ValueError('Two independent reviewers and adjudication required')
        text=review['verified_verbatim'].strip()
        if not text:raise ValueError('Use separately verified nonspeech manifest for empty labels')
        edits[k]=text
    return [dict(r,ref=edits[r['id']],reference_verified=True,reference_policy='two_reviewer_adjudication') if r['id'] in edits else dict(r) for r in rows], {'applied':len(edits),'sha256':file_hash(path)}

def cap_sources(rows,max_per_group=300,synthetic_fraction=.05,seed=13):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['cluster_id']].append(r)
    kept=[];removed=[]
    for group in groups.values():
        ordered=sorted(group,key=lambda r:(not r.get('reference_verified',False),digest([seed,r['id']])))
        # Synthetic often shares a single group for leakage protection; cap separately below.
        cap=len(ordered) if max_per_group is None or all(r['synthetic'] for r in ordered) else max_per_group
        kept.extend(ordered[:cap]);removed.extend(dict(r,curation_reason='recording_cap') for r in ordered[cap:])
    real=[r for r in kept if not r['synthetic']];syn=sorted([r for r in kept if r['synthetic']],key=lambda r:digest([seed,r['id']]))
    budget=int(len(real)*synthetic_fraction/max(1e-9,1-synthetic_fraction))
    removed.extend(dict(r,curation_reason='synthetic_cap') for r in syn[budget:])
    return sorted(real+syn[:budget],key=lambda r:r['id']),removed

def independent_dev(rows,seed=13,selection_fraction=.7):
    groups=sorted({r['cluster_id'] for r in rows},key=lambda k:digest([seed,k]))
    if len(groups)<4:raise ValueError('Need at least four dev source groups')
    n=min(len(groups)-2,max(2,int(len(groups)*selection_fraction)))
    select=set(groups[:n]);return [r for r in rows if r['cluster_id'] in select],[r for r in rows if r['cluster_id'] not in select]

def balanced_sample(rows,n,seed=13):
    g=collections.defaultdict(list)
    for r in sorted(rows,key=lambda r:digest([seed,r['id']])):g[r['cluster_id']].append(r)
    out=[]
    while g and len(out)<n:
        for k in sorted(list(g)):
            out.append(g[k].pop())
            if not g[k]:del g[k]
            if len(out)>=n:break
    return out

def teacher_review(rows,predictions):
    """Disagreement prioritizes human listening. Teacher is neither oracle nor replacement label."""
    from rme_core import edit_counts
    result=[]
    for r in rows:
        record=predictions[r['id']];hyp=record['hyp'];rw=normalize(r['ref']).split();hw=normalize(hyp).split()
        wer=100*sum(edit_counts(rw,hw))/max(1,len(rw));critical=critical_proxy(r['ref'],hyp)
        priority=3 if record['decoding']['needs_review'] else 2 if any(critical.values()) or wer>50 else 1 if wer>20 else 0
        result.append({'id':r['id'],'audio_sha256':r['audio_sha256'],'source':r.get('source_detail',r['source']),
             'reference_dataset':r['ref'],'teacher_hyp':hyp,'disagreement_wer':wer,'priority':priority,**critical})
    return sorted(result,key=lambda r:(-r['priority'],-r['disagreement_wer'],r['id']))

def check_new_training(new_rows,locked_rows):
    """Additional corpus must already conform to audited schema; no implicit split creation."""
    reserved=[r for r in locked_rows if r['split']!='train'];keys=('group_id','video_id','speaker_id','session_id','patient_id','pcm_sha256')
    blocked={k:{r[k] for r in reserved if r.get(k)} for k in keys}
    seen={r['id'] for r in locked_rows};texts={normalize(r['ref']) for r in reserved if len(normalize(r['ref']).split())>=8}
    for r in new_rows:
        if r['id'] in seen:raise ValueError('Duplicate additional id')
        seen.add(r['id'])
        if r.get('split')!='train' or not r.get('reference_verified') or not r.get('speaker_id') or not r.get('session_id'):
            raise ValueError('New corpus requires train-only, verified reference, speaker/session IDs')
        if r.get('commercial_use_reviewed') is not True:raise ValueError('New corpus use rights must be reviewed')
        if r.get('ref') and (r.get('adjudicated') is not True or not r.get('reviewer_1') or not r.get('reviewer_2') or r['reviewer_1']==r['reviewer_2']):
            raise ValueError('Additional speech requires two-reviewer adjudication')
        rel=Path(r['audio_file'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Additional audio path must stay inside ROOT')
        for k in keys:
            if r.get(k) and r[k] in blocked[k]:raise ValueError('Additional train overlaps heldout '+k)
        if normalize(r['ref']) in texts:raise ValueError('Additional reference overlaps heldout')
    return new_rows

def verified_negatives(path,root,locked_rows,max_count):
    if not Path(path).exists():return []
    from rme_core import inspect_audio
    items=json.loads(Path(path).read_text());out=[]
    for x in items:
        if x.get('human_verified_nonspeech') is not True or x.get('commercial_use_reviewed') is not True or x.get('ref')!='':
            raise ValueError('Negative must be human-verified real nonspeech with empty reference')
        if not x.get('reviewer_1') or not x.get('reviewer_2') or x['reviewer_1']==x['reviewer_2']:raise ValueError('Nonspeech requires two reviewers')
        rel=Path(x['audio_file'])
        if rel.is_absolute() or not (Path(root)/rel).resolve().is_relative_to(Path(root).resolve()):raise ValueError('Unsafe negative path')
        r=inspect_audio(dict(x,synthetic=False,reference_verified=True,split='train'),root)
        r['reject_reasons']=[v for v in r['reject_reasons'] if v!='empty_label_requires_nonspeech_verification']
        if r['reject_reasons']:raise ValueError('Invalid negative audio')
        out.append(r)
    check_new_training(out,locked_rows)
    return sorted(out,key=lambda r:digest(r['id']))[:max_count]

def near_duplicate_candidates(rows,threshold=.85,num_perm=64,max_pairs=10000):
    from datasketch import MinHash,MinHashLSH
    lsh=MinHashLSH(threshold=threshold,num_perm=num_perm);byid={r['id']:r for r in rows};pairs=[]
    for r in sorted(rows,key=lambda r:r['id']):
        words=normalize(r['ref']).split()
        if len(words)<8:continue
        shingles={' '.join(words[i:i+4]) for i in range(len(words)-3)}
        h=MinHash(num_perm=num_perm,seed=13)
        for sh in sorted(shingles):h.update(sh.encode())
        for other in sorted(lsh.query(h)):
            if byid[other]['split']!=r['split'] and len(pairs)<max_pairs:
                pairs.append({'id_a':other,'id_b':r['id'],'split_a':byid[other]['split'],'split_b':r['split'],
                              'status':'candidate_text_overlap_requires_review'})
        lsh.insert(r['id'],h)
    return {'pairs':pairs,'limit_reached':len(pairs)>=max_pairs,'method':'approximate word4gram MinHash LSH; not proof of duplicate audio'}
