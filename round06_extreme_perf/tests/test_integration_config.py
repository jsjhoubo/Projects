from dataclasses import replace
import json
import pytest
from perfkit.infra.config import ModelConfig,TrainConfig,EvalConfig,WorkloadConfig

def test_model_defaults():
    c=ModelConfig().validate()
    assert c.hidden_size%c.num_heads==0
    assert ModelConfig.from_json(c.as_json())==c

def test_user_configuration_roundtrip():
    raw={'vocab_size':128,'hidden_size':64,'num_heads':4,'num_layers':3,
         'intermediate_size':192,'max_position_embeddings':256,
         'dropout':.1,'rms_norm_eps':1e-5,'activation':'silu',
         'tie_embeddings':False,'positional':'rope'}
    c=ModelConfig.from_dict(raw)
    assert json.loads(c.as_json())==raw

@pytest.mark.parametrize('field,value',[('hidden_size',65),('num_layers',0),
    ('vocab_size',3),('dropout',1.0),('max_position_embeddings',0),
    ('activation','other'),('positional','unknown')])
def test_model_rejects_invalid(field,value):
    c=replace(ModelConfig(),**{field:value})
    with pytest.raises(ValueError):c.validate()

@pytest.mark.parametrize('field,value',[('micro_batch',0),('accumulation',0),
    ('learning_rate',0),('max_norm',-1),('label_smoothing',1),
    ('mixed_precision','fp8')])
def test_training_rejects_invalid(field,value):
    with pytest.raises(ValueError):replace(TrainConfig(),**{field:value}).validate()

def test_eval_and_workload_configs():
    assert EvalConfig().validate().val_batch_size>0
    assert WorkloadConfig().validate().min_length>=2
    with pytest.raises(ValueError):replace(EvalConfig(),temperature=0).validate()
    with pytest.raises(ValueError):replace(WorkloadConfig(),skew=2).validate()
