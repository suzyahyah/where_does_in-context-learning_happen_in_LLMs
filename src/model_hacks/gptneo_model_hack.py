#!/usr/bin/python3
# Author: Suzanna Sia
#
from transformers import GPTNeoForCausalLM, GPTNeoModel
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock, GPTNeoAttention, GPTNeoSelfAttention
from torch import nn

from src.model_hacks.attention_mixin import AttentionMaskMixin

class GPTNeoForCausalLMHack(GPTNeoForCausalLM):
    """Inheritance hack to manipulate attention layer"""

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTNeoModelHack(config)
        self.post_init()

class GPTNeoModelHack(GPTNeoModel):
    """Inheritance hack to manipulate attention layer"""

    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([GPTNeoBlockHack(config, layer_id=i) for i in range(config.num_layers)])
        self.post_init()


class GPTNeoBlockHack(GPTNeoBlock):
    """Inheritance hack to manipulate attention layer"""

    def __init__(self, config, layer_id=None):
        super().__init__(config, layer_id)
        self.attn = GPTNeoAttentionHack(config, layer_id)


class GPTNeoAttentionHack(GPTNeoAttention):
    """Inheritance hack to manipulate attention layer"""

    def __init__(self, config, layer_id=0):
        super().__init__(config, layer_id)
        if self.attention_type in ('global', 'local'):
            self.attention = GPTNeoSelfAttentionHack(config, self.attention_type, layer_id)
        else:
            raise NotImplementedError(f"Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: {config.attention_layers}. Select attn layer types from ['global', 'local'] only.")


class GPTNeoSelfAttentionHack(GPTNeoSelfAttention, AttentionMaskMixin):
    """Inheritance hack with AttentionMaskMixin to manipulate attention layer"""

    def __init__(self, config, attention_type, layer_id):

        super().__init__(config, attention_type, layer_id=layer_id)
        AttentionMaskMixin.__init__(self)

    def forward(self, *args, **kwargs):
        kwargs['attention_mask'] = self._modify_mask(kwargs['attention_mask'])
        return super().forward(*args, **kwargs)
