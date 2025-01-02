#!/usr/bin/python3
# Author: Suzanna Sia

from transformers.models.starcoder2.modeling_starcoder2 import (
    Starcoder2ForCausalLM, 
    Starcoder2Model,
    Starcoder2DecoderLayer,
    Starcoder2Attention,

)

from torch import nn
from src.model_hacks.attention_mixin import AttentionMaskMixin

class Starcoder2ForCausalLMHack(Starcoder2ForCausalLM):
    """Inheritance hack to manipulate attention layer"""

    def __init__(self, config):
        super().__init__(config)
        self.model = Starcoder2ModelHack(config)
        self.post_init()


class Starcoder2ModelHack(Starcoder2Model):
    """Inheritance hack to manipulate attention layer"""

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Starcoder2DecoderLayerHack(config, layer_idx) \
                    for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

class Starcoder2DecoderLayerHack(Starcoder2DecoderLayer):
    """Inheritance hack to manipulate attention layer"""

    def __init__(self, config=None, layer_idx=0):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = Starcoder2AttentionHack(config, layer_idx)

class Starcoder2AttentionHack(Starcoder2Attention, AttentionMaskMixin):
    """Inheritance hack with AttentionMaskMixin to manipulate attention layer"""

    def __init__(self, config, layer_idx):
        super().__init__(config=config, layer_idx=layer_idx)
        AttentionMaskMixin.__init__(self) 

    def forward(self, *args, **kwargs):
        kwargs['attention_mask'] = self._modify_mask(kwargs['attention_mask'])
        return super().forward(*args, **kwargs)

