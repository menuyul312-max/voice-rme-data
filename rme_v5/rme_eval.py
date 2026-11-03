"""Comparable ASR, clinical proxy and group uncertainty; no certification from proxy scores."""
import collections
from rme_core import corpus_metrics,critical_proxy,paired_cluster_ci,normalize,edit_counts
from rme_decode import loop_flags

def evaluate_run(rows,run):
    hyps={k:v['hyp'] for k,v in run['records'].items()}
    m=corpus_metrics(rows,hyps)
    m['critical_proxy']={k:sum(critical_proxy(r['ref'],hyps[r['id']])[k] for r in rows) for k in ('number','unit','negation')}
    m['unresolved_decoding']=sum(v['decoding']['needs_review'] for v in run['records'].values())
    m['length_cap_primary']=sum(v['token_cap_hit'] for v in run['records'].values())
    m['loop_flags_final']=sum(bool(loop_flags(h)) for h in hyps.values())
    m['retry_count']=sum(bool(v['decoding']['primary_flags']) for v in run['records'].values())
    m['empty_ref_false_activations']=sum(bool(normalize(hyps[r['id']])) for r in rows if not normalize(r['ref']))
    m['verified_reference_clips']=sum(bool(r.get('reference_verified')) for r in rows)
    m['timing']=run['timing'];return m

def compare(rows,candidate,control,B=2000):
    cm,bm=evaluate_run(rows,candidate),evaluate_run(rows,control)
    ch={k:v['hyp'] for k,v in candidate['records'].items()};bh={k:v['hyp'] for k,v in control['records'].items()}
    paired=paired_cluster_ci(rows,ch,bh,B=B)
    passed=cm['wer']<bm['wer'] and cm['cer']<=bm['cer'] and cm['unresolved_decoding']==0 and all(cm['critical_proxy'][k]<=bm['critical_proxy'][k] for k in cm['critical_proxy'])
    return {'passed_internal_screen':passed,'candidate':cm,'control':bm,'paired':paired,
            'independent_clinical_evidence':False,'reference_warning':'Proxies against unverified labels do not establish clinical correctness'}

def paired_edit_analysis(ref,hyp):
    s,d,i=edit_counts(normalize(ref).split(),normalize(hyp).split())
    return {'substitution':s,'deletion':d,'insertion':i}

def wilson_upper(k,n,z=1.96):
    if not n:return None
    from math import sqrt
    p=k/n;return (p+z*z/(2*n)+z*sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)

def clinician_review(path,planned_ids):
    import csv
    from pathlib import Path
    if not Path(path).exists():return {'status':'NOT_MEASURED'}
    rows=list(csv.DictReader(Path(path).open()))
    if len(rows)!=len(planned_ids) or {r['id'] for r in rows}!=set(planned_ids):raise ValueError('Clinical review needs exact unique coverage')
    errors=opportunities=0
    for r in rows:
        if r['adjudicated'].lower()!='true' or not r['reviewer_1'] or not r['reviewer_2'] or r['reviewer_1']==r['reviewer_2']:raise ValueError('Incomplete clinician adjudication')
        e,n=int(r['critical_errors']),int(r['critical_opportunities'])
        if not 0<=e<=n:raise ValueError('Invalid critical error counts')
        errors+=e;opportunities+=n
    return {'status':'HUMAN_REVIEWED','critical_errors':errors,'critical_opportunities':opportunities,
            'rate':errors/opportunities if opportunities else None,
            'note':'Opportunity definition/severity must be preregistered. No independence assumption or certification inferred.'}
