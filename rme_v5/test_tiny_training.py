"""Actual torch/PEFT/Whisper forward/backward test; no full model download or GPU."""
import contextlib,types
import numpy as np
import pytest
import torch
from transformers import WhisperConfig,WhisperForConditionalGeneration
from rme_profiles import parameter_report
from rme_train import build_model

@pytest.mark.parametrize('profile',['decoder_qv','top8_qv','all_attention'])
def test_real_tiny_whisper_lora_gradients(monkeypatch,profile):
    torch.set_num_threads(2);torch.manual_seed(13)
    cfg=WhisperConfig(vocab_size=64,num_mel_bins=8,d_model=16,encoder_layers=3,decoder_layers=2,
         encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,
         max_source_positions=12,max_target_positions=20,pad_token_id=0,bos_token_id=1,eos_token_id=2,decoder_start_token_id=1,
         suppress_tokens=[],begin_suppress_tokens=[])
    monkeypatch.setattr(WhisperForConditionalGeneration,'from_pretrained',lambda *a,**kw:WhisperForConditionalGeneration(cfg))
    model=build_model('tiny',r=4,profile=profile)
    report=parameter_report(model)
    assert 0<report['trainable_percent']<100
    assert all('lora_' in n for n in report['trainable_names'])
    if profile=='decoder_qv':assert not any('.encoder.layers.' in n for n in report['trainable_names'])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    inputs=torch.randn(2,8,24);labels=torch.tensor([[3,4,2],[5,6,2]])
    loss=model(input_features=inputs,attention_mask=torch.ones(2,24,dtype=torch.long),labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward();grads=[p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g)>0 for g in grads)
    original={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3);opt.step()
    for name,p in model.named_parameters():
        if name in original:assert torch.equal(p,original[name])
    model.eval();model.config.use_cache=True
    merged=model.merge_and_unload()
    assert not any('lora_' in n for n,p in merged.named_parameters())
    assert torch.isfinite(merged(input_features=inputs,labels=labels).loss)

def test_actual_trainer_step_and_generative_metrics(tmp_path,monkeypatch):
    import soundfile as sf
    from rme_train import make_trainer
    from transformers import EarlyStoppingCallback
    torch.set_num_threads(2)
    cfg=WhisperConfig(vocab_size=64,num_mel_bins=8,d_model=16,encoder_layers=2,decoder_layers=2,
         encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,
         max_source_positions=12,max_target_positions=20,pad_token_id=0,bos_token_id=1,eos_token_id=2,
         decoder_start_token_id=1,suppress_tokens=[],begin_suppress_tokens=[])
    monkeypatch.setattr(WhisperForConditionalGeneration,'from_pretrained',lambda *a,**kw:WhisperForConditionalGeneration(cfg))
    model=build_model('tiny',r=4)
    model.generation_config.language=None;model.generation_config.task=None
    class Tokens:
        pad_token_id=0
        def __call__(self,text):return types.SimpleNamespace(input_ids=[1,3,2])
        def convert_tokens_to_ids(self,text):return 1
        def pad(self,ids,return_tensors):
            tensor=torch.tensor([x['input_ids'] for x in ids]);return types.SimpleNamespace(input_ids=tensor,attention_mask=torch.ones_like(tensor))
        def batch_decode(self,seq,**kw):return ['nyeri dada']*len(seq)
    class LabelBatch(dict):
        __getattr__=dict.__getitem__
    tok=Tokens();original_pad=tok.pad
    tok.pad=lambda *a,**kw:LabelBatch(vars(original_pad(*a,**kw)))
    class Features:
        def __call__(self,waves,**kw):return {'input_features':torch.zeros(len(waves),8,24),'attention_mask':torch.ones(len(waves),24,dtype=torch.long)}
    class Processor:
        tokenizer=tok;feature_extractor=Features();pad_token_id=0;eos_token_id=2;model_input_names=['input_features']
        def batch_decode(self,*a,**kw):return self.tokenizer.batch_decode(*a,**kw)
    sf.write(tmp_path/'a.wav',np.ones(8000,dtype=np.float32)*.1,16000)
    rows=[{'id':'a','ref':'nyeri dada','audio_file':'a.wav'}]
    trainer=make_trainer(model,Processor(),rows,rows,tmp_path,tmp_path/'out',
       {'batch':1,'accum':1,'epochs':1,'lr':1e-4,'bf16':False,'seed':13})
    trainer.args.max_steps=1;trainer.args.eval_strategy='no';trainer.args.save_strategy='no'
    trainer.args.load_best_model_at_end=False;trainer.args.dataloader_num_workers=0
    trainer.remove_callback(EarlyStoppingCallback)
    result=trainer.train();assert np.isfinite(result.training_loss)
    trainer.args.generation_max_length=10
    report=trainer.evaluate()
    assert report['eval_wer']==0. and report['eval_cer']==0.
