"""Validation of researcher-controlled experiment configurations.

Defaults should be small enough to run with CPU-only PyTorch. The configuration
object deliberately permits several independent attention backends for
controlled experiments; training code is responsible for selecting one.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 384
    hidden_size: int = 96
    num_heads: int = 4
    num_layers: int = 2
    intermediate_size: int = 384
    max_position_embeddings: int = 256
    dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    activation: str = 'gelu'
    tie_embeddings: bool = True
    positional: str = 'learned'

    def validate(self):
        if self.vocab_size < 8: raise ValueError('vocab_size < 8')
        if self.hidden_size <= 0 or self.hidden_size % self.num_heads:
            raise ValueError('head dimension must be integral')
        if self.num_layers < 1: raise ValueError('num_layers must be positive')
        if self.max_position_embeddings < 1: raise ValueError('invalid maximum sequence length')
        if not 0 <= self.dropout < 1: raise ValueError('dropout outside [0, 1)')
        if self.activation not in ('gelu', 'silu', 'relu'): raise ValueError('unknown activation')
        if self.positional not in ('learned', 'rope'): raise ValueError('unknown positional mode')
        return self

    @classmethod
    def from_dict(cls, obj):
        return cls(**obj).validate()

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def as_json(self):
        return json.dumps(asdict(self), sort_keys=True)

@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    micro_batch: int = 4
    accumulation: int = 4
    max_norm: float = 1.0
    label_smoothing: float = 0.0
    mixed_precision: str = 'none'
    diagnostics: bool = True

    def validate(self):
        if self.learning_rate <= 0: raise ValueError('learning rate')
        if self.micro_batch < 1 or self.accumulation < 1: raise ValueError('batch')
        if self.max_norm < 0: raise ValueError('max_norm')
        if not 0 <= self.label_smoothing < 1: raise ValueError('smoothing')
        if self.mixed_precision not in ('none','bf16','fp16'):
            raise ValueError('precision')
        return self

@dataclass(frozen=True)
class EvalConfig:
    val_batch_size: int = 8
    max_new_tokens: int = 24
    top_k: int = 0
    temperature: float = 1.0
    def validate(self):
        if self.val_batch_size < 1 or self.max_new_tokens < 0: raise ValueError('size')
        if self.top_k < 0 or self.temperature <= 0: raise ValueError('sampling')
        return self

@dataclass(frozen=True)
class WorkloadConfig:
    seed: int = 42
    train_count: int = 256
    val_count: int = 32
    min_length: int = 16
    max_length: int = 64
    vocab_size: int = 384
    skew: float = 0.0
    def validate(self):
        if min(self.train_count,self.val_count) < 0: raise ValueError('count')
        if not 2 <= self.min_length <= self.max_length: raise ValueError('length')
        if self.vocab_size < 8 or not 0 <= self.skew <= 1: raise ValueError('distribution')
        return self
