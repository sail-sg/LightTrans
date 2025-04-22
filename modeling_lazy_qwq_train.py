import os
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

compiled_flex_attn = torch.compile(flex_attention)

# from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM


def sliding_window_causal_with_stream(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = (q_idx - kv_idx) <= 1024 
    stream_mask = (q_idx >= 128) & (kv_idx <= 128)
    # return causal_mask
    return (causal_mask & window_mask) | stream_mask
# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2 model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from functools import partial

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from flash_attn import flash_attn_func, flash_attn_with_kvcache

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

# from torch.nn.attention.flex_attention import flex_attention

logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-7B-beta"
_CONFIG_FOR_DOC = "Qwen2Config"


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2Config] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2RotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_single_rope(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed

# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
        self.K_Cache = None
        self.V_Cache = None
        self.answer_K_Cache = None
        self.answer_V_Cache = None
        self.max_len = 2048
        self.log_ratio = math.log(0.7)
        self.prefix_lens = None
        self.layer_idx = layer_idx
        self.last_layer = (config.num_hidden_layers == self.layer_idx + 1)
        self.softmax_scale = 1 / (self.head_dim ** 0.5)
        self.range_indices = None
        self.sink_len = 256
        self.kv_len = 2048
        self.stream_k_cache = None
        self.stream_v_cache = None
        self.coc_k_cache = None
        self.coc_v_cache = None
        self.buffer_size = 0
        self.flex_attn = None
        # self.lazy_list = [31, 27, 18, 9, 15, 19, 16, 17, 12, 8, 6, 10, 7, 5, 11, 14]
        self.lazy_list = [38, 51, 35, 26, 23, 43, 37, 24, 21, 28, 30, 22, 32, 33, 13, 20, 31, 27, 18, 9, 15, 19, 16, 17, 12, 8, 6, 10, 7, 5, 11, 14]
        # self.lazy_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
        # self.lazy_list = [38]
    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_lens=None,
        flex_attn=None,
        tree_mask=None,
        exec_type="training",
    ):
        
        kv_cache = None
        if self.layer_idx not in self.lazy_list:
            y = self.vanilla_training(hidden_states, position_embeddings)
        elif exec_type == "sliding_training":
            y = self.sliding_training(hidden_states, position_embeddings, flex_attn)

        return y, kv_cache
    
    def sliding_training(
            self,
            hidden_states,
            position_embeddings,
            flex_attn_mask,
            ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
        global_out = flash_attn_func(query_states, key_states, value_states, causal=True)

        mha_query = query_states.transpose(1, 2).reshape(bsz, self.num_heads, q_len, self.head_dim)
        # q_1 = mha_query[:, :32]
        # q_2 = mha_query[:, 32: 40]     
        mha_key = key_states.transpose(1, 2).reshape(bsz, self.num_key_value_heads, 1, q_len, self.head_dim).expand(-1, -1, self.num_key_value_groups, -1, -1).contiguous().view(bsz, self.num_heads, q_len, self.head_dim)
        mha_value = value_states.transpose(1, 2).reshape(bsz, self.num_key_value_heads, 1, q_len, self.head_dim).expand(-1, -1, self.num_key_value_groups, -1, -1).contiguous().view(bsz, self.num_heads, q_len, self.head_dim)
        # k_1, k_2 = mha_key[:, :32], mha_key[:, 32: 40]
        # v_1, v_2 = mha_value[:, :32], mha_value[:, 32: 40]

        # s_1 = compiled_flex_attn(q_1.contiguous(), k_1.contiguous(), v_1.contiguous(), block_mask=flex_attn_mask)#.transpose(1, 2).reshape(bsz, q_len, -1)
        # s_2 = compiled_flex_attn(q_2.contiguous(), k_2.contiguous(), v_2.contiguous(), block_mask=flex_attn_mask)#.transpose(1, 2).reshape(bsz, q_len, -1)
        # sliding_out = torch.cat([s_1, s_2], dim=1).transpose(1, 2).reshape(bsz, q_len, -1)
        sliding_out = compiled_flex_attn(mha_query, mha_key, mha_value, block_mask=flex_attn_mask).transpose(1, 2).reshape(bsz, q_len, -1)
        sliding_out = self.o_proj(sliding_out)
        return sliding_out
    
    def vanilla_training(
            self,
            hidden_states,
            position_embeddings,
            ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True).view(bsz, q_len, -1)
        
        attn_output = self.o_proj(attn_output)

        return attn_output
    

QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
    "flash_attention_2": Qwen2Attention,
    "sdpa": Qwen2Attention,
}


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.last_layer = (config.num_hidden_layers == self.layer_idx + 1)
        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings,  # will become mandatory in v4.46
        cache_lens=None,
        flex_attn=None,
        exec_type=None,
        tree_mask=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:


        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, kv_cache = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache_lens=cache_lens,
            flex_attn=flex_attn,
            exec_type=exec_type,
            tree_mask=tree_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, kv_cache)

        return outputs

class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()



    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    

    def forward(
        self,
        input_ids,
        position_ids=None,
        inputs_embeds=None,
        cache_lens=None,
        flex_attn=None,
        exec_type=None,
        tree_mask=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:


        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(1))[None, :].to(input_ids.device)
            if cache_lens is not None:
                # print(f"cache_lens: {cache_lens}")
                # print(f"position_ids: {position_ids}")
                if tree_mask is None:
                    position_ids = position_ids + cache_lens[:, None]
                else:
                    position_ids = tree_mask.sum(dim=-1) - 1 + cache_lens[:, None]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)


        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    cache_lens,
                    flex_attn,
                    exec_type,
                    tree_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings,
                    cache_lens,
                    flex_attn,
                    exec_type,
                    tree_mask,
                )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        if exec_type == "glide_training":
            kv_cache = layer_outputs[1]
        else:
            kv_cache = None
        # add hidden states from the last decoder layer

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=kv_cache,
            hidden_states=None,
            attentions=None,
        )

class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.eod = 151645
        # Initialize weights and apply final processing
        self.post_init()
        self.block_mask = create_block_mask(sliding_window_causal_with_stream, None, None, 1024, 1024)
        self.flag = True

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_max_gen_len(self, max_gen_len):
        for layer in self.model.layers:
            layer.self_attn.max_len = max_gen_len

    def set_stream_len(self, sink_len, kv_len):
        for layer in self.model.layers:
            layer.self_attn.sink_len = sink_len
            layer.self_attn.kv_len = kv_len
            layer.self_attn.max_len = sink_len + kv_len
    
    def set_log_ratio(self, log_ratio):
        for layer in self.model.layers:
            layer.self_attn.log_ratio = log_ratio
    
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def compute_loss(self, hidden_states, labels):
        logits = self.lm_head(hidden_states).float()
        # from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return loss


    def forward(
        self,
        input_ids,
        attention_mask= None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        cache_lens=None,
        exec_type="sliding_training",
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        

        # if exec_type == "sliding_training":
        self.block_mask = create_block_mask(sliding_window_causal_with_stream, None, None, input_ids.size(1), input_ids.size(1))
        flex_attn = self.block_mask
        # else:
        #     flex_attn = None
        if self.flag:
            print('===you are using lazy qwq====')
            self.flag = False
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            cache_lens=cache_lens,
            flex_attn=flex_attn,
            exec_type=exec_type,
        )

        hidden_states = outputs.last_hidden_state

        loss = None
        if labels is not None:
            loss = self.compute_loss(hidden_states[:, :], labels[:, :])
        else:
            logits = self.lm_head(hidden_states[:, -128:, :]).float()


        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
