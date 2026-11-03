"""GPU training helpers; imported only after Colab dependencies are installed."""
import math, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from rme_core import SR, BASE, corpus_metrics

def load_wave(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    x,sr=sf.read(path,dtype='float32',always_2d=True)
    if not np.isfinite(x).all() or not 0 < len(x)/sr <= 30.: raise ValueError('Invalid audio: '+str(path))
    x=x.mean(axis=1)
    if sr!=SR:
        g=math.gcd(sr,SR);x=resample_poly(x,SR//g,sr//g)
    return np.asarray(x,dtype=np.float32)

class Clips:
    def __init__(self,rows,root,augment=False): self.rows,self.root,self.augment=rows,Path(root),augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        return {'path':str(self.root/r['audio_file']),'text':r['ref'],'augment':self.augment}

@dataclass
class Collator:
    processor: object
    rng: object = None
    _worker_seed: object = None
    def __call__(self,items):
        import torch
        worker_seed=torch.initial_seed() % 2**32
        if self.rng is None or self._worker_seed!=worker_seed:
            self.rng=np.random.default_rng(worker_seed);self._worker_seed=worker_seed
        waves=[]; ids=[]
        for it in items:
            x=load_wave(it['path'])
            if it['augment']:
                if self.rng.random()<.3:
                    rms=np.sqrt(np.mean(x*x))
                    x=x+self.rng.normal(0,rms*10**(-self.rng.uniform(20,35)/20),len(x)).astype(np.float32)
                if self.rng.random()<.3: x=x*10**(self.rng.uniform(-3,3)/20)
                peak=np.max(np.abs(x))
                if peak>.99:x=x*(.99/peak)
            waves.append(x)
            seq=self.processor.tokenizer(it['text']).input_ids
            if seq[0]==self.processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>'): seq=seq[1:]
            if len(seq)>448: raise ValueError('Label >448 tokens: segment and realign; never truncate')
            ids.append({'input_ids':seq})
        batch=self.processor.feature_extractor(waves,sampling_rate=SR,return_tensors='pt',return_attention_mask=True)
        labels=self.processor.tokenizer.pad(ids,return_tensors='pt')
        batch['labels']=labels['input_ids'].masked_fill(labels.attention_mask.ne(1),-100)
        return dict(batch)

def build_model(revision,r=16,profile="decoder_qv",model_id=BASE):
    import torch
    from transformers import WhisperForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    model=WhisperForConditionalGeneration.from_pretrained(model_id,revision=revision,torch_dtype=torch.float32,attn_implementation='sdpa')
    model.config.use_cache=False
    model.config.apply_spec_augment=False  # mild waveform augmentation only; ablate on dev
    model.generation_config.language='indonesian';model.generation_config.task='transcribe'
    model.generation_config.forced_decoder_ids=None
    model.generation_config.max_length=448
    model.generation_config.use_cache=True
    from rme_decode import DECODE
    for key,value in DECODE.items():setattr(model.generation_config,key,value)
    from rme_profiles import lora_targets,PROFILES
    return get_peft_model(model,LoraConfig(r=r,lora_alpha=2*r,lora_dropout=PROFILES[profile]['dropout'],
        target_modules=lora_targets(model,profile),bias='none'))

def make_trainer(model,proc,train,dev,root,out,config):
    import torch
    from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments,EarlyStoppingCallback
    def metrics(pred):
        p=pred.predictions[0] if isinstance(pred.predictions,tuple) else pred.predictions
        p=np.where(p<0,proc.tokenizer.pad_token_id,p)
        labels=np.where(pred.label_ids<0,proc.tokenizer.pad_token_id,pred.label_ids)
        refs=proc.batch_decode(labels,skip_special_tokens=True)
        hyps=proc.batch_decode(p,skip_special_tokens=True)
        m=corpus_metrics([{'id':str(i),'ref':t} for i,t in enumerate(refs)],{str(i):t for i,t in enumerate(hyps)})
        from rme_decode import loop_flags
        return {'wer':m['wer'],'cer':m['cer'],'loop_rate':sum(bool(loop_flags(h)) for h in hyps)/len(hyps)}
    steps=math.ceil(len(train)/(config['batch']*config['accum']))
    every=max(1,min(125,math.ceil(steps/4)))
    args=Seq2SeqTrainingArguments(output_dir=str(out),
        per_device_train_batch_size=config['batch'],per_device_eval_batch_size=max(1,config['batch']//2),
        gradient_accumulation_steps=config['accum'],num_train_epochs=config['epochs'],
        learning_rate=config['lr'],lr_scheduler_type='cosine',warmup_ratio=.05,weight_decay=.01,
        max_grad_norm=1.,bf16=config['bf16'],fp16=torch.cuda.is_available() and not config['bf16'],
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        eval_strategy='steps',save_strategy='steps',eval_steps=every,save_steps=every,
        save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='wer',greater_is_better=False,
        predict_with_generate=True,generation_max_length=448,generation_num_beams=3,
        eval_accumulation_steps=8,logging_steps=25,report_to='none',
        dataloader_num_workers=2,dataloader_pin_memory=True,dataloader_persistent_workers=False,
        remove_unused_columns=False,label_names=['labels'],seed=config['seed'],data_seed=config['seed'],
        optim='adamw_torch',save_safetensors=True)
    return Seq2SeqTrainer(model=model,args=args,train_dataset=Clips(train,root,True),
        eval_dataset=Clips(dev,root,False),data_collator=Collator(proc),processing_class=proc,
        compute_metrics=metrics,callbacks=[EarlyStoppingCallback(early_stopping_patience=4)])

def smoke_gradients(trainer):
    import torch
    batch=trainer.data_collator([trainer.train_dataset[0]])
    batch={k:v.to(trainer.args.device) for k,v in batch.items()}
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    trainer.model.train();trainer.model.zero_grad(set_to_none=True)
    with trainer.compute_loss_context_manager():loss=trainer.model(**batch).loss
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite smoke loss')
    loss.backward()
    grads=[p.grad for n,p in trainer.model.named_parameters() if p.requires_grad and 'lora_' in n]
    if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
        raise RuntimeError('Missing/nonfinite LoRA gradient')
    if not any(torch.count_nonzero(g)>0 for g in grads): raise RuntimeError('All LoRA gradients zero')
    trainer.model.zero_grad(set_to_none=True)
    return float(loss.detach())
