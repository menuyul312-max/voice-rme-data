"""Prioritize held-out clinician adjudication; this output MUST NOT become training data."""
import argparse
import csv
import json
from pathlib import Path
from benchmark_real import normalize, build_term_list, word_error_counts
from local_med_llm import CLINICAL_ONTOLOGY


def build(input_csv, output_csv):
    terms=build_term_list(CLINICAL_ONTOLOGY)
    records=[]
    with open(input_csv, newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            ref=normalize(row['current_reference'])
            hyp=normalize(row['hyp_rme_v2|beam_fallback'])
            misses=[t for t in terms if f' {t} ' in f' {ref} ' and f' {t} ' not in f' {hyp} ']
            extras=[t for t in terms if f' {t} ' in f' {hyp} ' and f' {t} ' not in f' {ref} ']
            s,d,i=word_error_counts(ref.split(), hyp.split())
            records.append(dict(chunk_id=row['chunk_id'],audio_file=row['audio_file'],
                split_policy='HOLDOUT_DO_NOT_TRAIN',missing_terms='; '.join(misses),extra_terms='; '.join(extras),
                caption_wer_percent=round(100*(s+d+i)/max(len(ref.split()),1),2),
                clinical_disagreements=len(misses)+len(extras),
                corrected=bool(row.get('corrected_text','').strip())))
    records.sort(key=lambda r:(-r['clinical_disagreements'],-r['caption_wer_percent'],r['chunk_id']))
    with open(output_csv,'w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return {'sample_clips':len(records),'corrected_clips':sum(r['corrected'] for r in records),
            'clips_with_clinical_disagreement':sum(r['clinical_disagreements']>0 for r in records),
            'note':'Reference captions are unverified. Queue is adjudication only, never a training set.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input_csv');parser.add_argument('output_csv')
    args=parser.parse_args();print(json.dumps(build(args.input_csv,args.output_csv),indent=2))
