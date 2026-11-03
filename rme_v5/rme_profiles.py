"""Predeclared ablations; actual parameter counts are recorded per run."""
import re
PROFILES={
 'decoder_qv':{'r':16,'lr':1e-5,'dropout':.05,'encoder_top':0,'projections':['q_proj','v_proj']},
 'top8_qv':{'r':16,'lr':5e-6,'dropout':.05,'encoder_top':8,'projections':['q_proj','v_proj']},
 'all_attention':{'r':16,'lr':1e-5,'dropout':.05,'encoder_top':None,'projections':['q_proj','k_proj','v_proj','out_proj']}}

def lora_targets(model,profile):
    p=PROFILES[profile];depth=model.config.encoder_layers
    top=depth if p['encoder_top'] is None else min(depth,p['encoder_top'])
    targets=[]
    for name,module in model.named_modules():
        if name.rsplit('.',1)[-1] not in p['projections']:continue
        if '.decoder.layers.' in name:targets.append(name);continue
        match=re.search(r'encoder.layers.(\d+)\.',name)
        if match and int(match[1])>=depth-top:targets.append(name)
    if not targets:raise ValueError('No matching adaptation modules')
    return targets

def parameter_report(model):
    total=sum(p.numel() for p in model.parameters());train=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not total or not train:raise ValueError('No trainable parameters')
    return {'total_parameters_including_adapters':total,'trainable_parameters':train,
            'trainable_percent':100*train/total,'frozen_percent':100*(total-train)/total,
            'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad]}
