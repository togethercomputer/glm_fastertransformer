# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import print_function

import os
import re
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

def get_torch_default_comm():
    r"""
    The NCCL communicator is needed so that Fast MoE can perform customized
    communication operators in the C code. However, it is not a publicly
    available variable. Therefore, a hacking class of the `ProcessGroupNCCL`
    in Fast MoE's C code takes the `_default_pg` and tries to dig the
    communicator out from the object. As PyTorch's private interface varies from
    time to time, different hacking techniques are tried one-by-one to be
    compatible with various versions of PyTorch.
    """
    try:
        comm = dist.distributed_c10d._get_default_group()
        return comm
    except Exception as _:
        pass
    try:
        comm = dist.distributed_c10d._default_pg
        if comm is not None:
            return comm
    except Exception as _:
        pass
    raise RuntimeError("Unsupported PyTorch version")


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-65504):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # sorted_indices_to_remove[..., :] = 1
        # sorted_indices_to_remove[..., 2] = 0

        # print(sorted_logits[cumulative_probs < top_p])
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    
    return logits

class GlmWeights(object):
    def __init__(self, head_num, size_per_head, layer_num, vocab_size, max_seq_len, tensor_para_size, pipeline_para_size, dtype):
        assert(head_num % tensor_para_size == 0)
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.layers_per_device = layer_num // pipeline_para_size

        local_head_num = head_num // tensor_para_size
        global_head_num = head_num
        local_hidden_units = local_head_num * size_per_head
        global_hidden_units = global_head_num * size_per_head
        local_inter_size = local_hidden_units * 8 // 3

        self.local_head_num = local_head_num
        self.global_head_num = global_head_num
        self.local_hidden_units = local_hidden_units
        self.global_hidden_units = global_hidden_units
        self.local_inter_size = local_inter_size
        self.dtype = dtype

        self.w = []
        self.weight = []
        self.scale = []

        # Transformer blocks
        self.w.extend([torch.zeros(3 * local_hidden_units, dtype = torch.float16)] * layer_num)                                   # attention.query_key_value.bias
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] * layer_num)                                   # attention.dense.bias
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] * layer_num)                                   # input_layernorm.bias
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] * layer_num)                                   # input_layernorm.weight
        self.w.extend([torch.zeros(local_inter_size, dtype = torch.float16)] * layer_num)                                   # mlp.dense_h_to_4h.bias.1
        self.w.extend([torch.zeros(local_inter_size, dtype = torch.float16)] * layer_num)                                   # mlp.dense_h_to_4h.bias.2
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] * layer_num)                                   # mlp.dense_4h_to_h.bias
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] * layer_num)                                   # post_attention_layernorm.bias
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] * layer_num)                                   # post_attention_layernorm.weight
        
        if dtype in ['fp16', 'int8']:
            w_type = torch.int8 if dtype == 'int8' else torch.float16
            self.weight.extend([torch.zeros(global_hidden_units * 3 * local_hidden_units, dtype = w_type)] * layer_num)             # attention.query_key_value.weight
            self.weight.extend([torch.zeros(local_hidden_units * global_hidden_units, dtype = w_type)] * layer_num)                                   # attention.dense.weight
            self.weight.extend([torch.zeros(global_hidden_units * local_inter_size, dtype = w_type)] * layer_num)                                   # mlp.dense_h_to_4h.weight.1
            self.weight.extend([torch.zeros(global_hidden_units * local_inter_size, dtype = w_type)] * layer_num)                                   # mlp.dense_h_to_4h.weight.2
            self.weight.extend([torch.zeros(local_inter_size * global_hidden_units, dtype = w_type)] * layer_num)             

        else:
            self.weight.extend([torch.zeros(global_hidden_units * 3 * local_hidden_units // 2, dtype = torch.int8)] * layer_num)             # attention.query_key_value.weight
            self.weight.extend([torch.zeros(local_hidden_units * global_hidden_units // 2, dtype = torch.int8)] * layer_num)                                   # attention.dense.weight
            self.weight.extend([torch.zeros(global_hidden_units * local_inter_size // 2, dtype = torch.int8)] * layer_num)                                   # mlp.dense_h_to_4h.weight.1
            self.weight.extend([torch.zeros(global_hidden_units * local_inter_size // 2, dtype = torch.int8)] * layer_num)                                   # mlp.dense_h_to_4h.weight.2
            self.weight.extend([torch.zeros(local_inter_size * global_hidden_units // 2, dtype = torch.int8)] * layer_num)                                   # mlp.dense_4h_to_h.weight
        
        # scale
        if dtype in ['int8', 'int4']:
            self.scale.extend([torch.zeros(3 * local_hidden_units, dtype = torch.float16)] * layer_num)
            self.scale.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] * layer_num)
            self.scale.extend([torch.zeros(local_inter_size, dtype = torch.float16)] * layer_num)
            self.scale.extend([torch.zeros(local_inter_size, dtype = torch.float16)] * layer_num)
            self.scale.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] * layer_num)
        
        # After Transformer blocks
        self.w.append(torch.zeros(global_hidden_units, dtype = torch.float16))   # layernorm_gamma final_layernorm.weight
        self.w.append(torch.zeros(global_hidden_units, dtype = torch.float16))   # layernorm_beta  final_layernorm.bias
        self.w.append(torch.zeros(vocab_size * global_hidden_units // tensor_para_size, dtype = torch.float16))   # embedding_table model.wte


    def __getitem__(self, idx):
        return self.w[idx]

    def __setitem__(self, idx, val):
        self.w[idx] = val

    def __len__(self):
        return len(self.w)

    def _map(self, func):
        for w in [self.w, self.weight, self.scale]:
            for i in range(len(w)):
                if isinstance(w[i], list):
                    for j in range(len(w[i])):
                        w[i][j] = func(w[i][j])
                else:
                    w[i] = func(w[i])

    def load(self, ckpt_path, tensor_para_rank, pipeline_para_rank):
        if not os.path.exists(ckpt_path):
            return False

        checkpoint_name = os.path.join(ckpt_path, 'mp_rank_{:02d}_model_states.pt'.format(tensor_para_rank))

        module = torch.load(checkpoint_name, map_location='cpu')['module']

        num_attention_heads = 96
        tensor_model_parallel_size = self.tensor_para_size
        layer_num = self.layer_num

        w = []
        weight = []
        scale = []

        num_splits = 3

        hidden_dim, local_dim = module['transformer.layers.0.attention.query_key_value.weight'].T.shape
        local_dim = local_dim // num_splits
        head_num = num_attention_heads
        size_per_head = hidden_dim // head_num
        if self.dtype == 'int4':
            size_per_head *= 2
        head_num = head_num // tensor_model_parallel_size
        if self.dtype in ['int8', 'int4']:
            scale.extend([module[f'transformer.layers.{i}.attention.query_key_value.weight_scale'].reshape(head_num, num_splits, size_per_head).permute(1, 0, 2).reshape(3, local_dim) for i in range(layer_num)])
            weight.extend([module[f'transformer.layers.{i}.attention.query_key_value.weight'].T.reshape(hidden_dim, head_num, num_splits, size_per_head).permute(0, 2, 1, 3).reshape(hidden_dim, 3 * local_dim).T for i in range(layer_num)])
        else:
            weight.extend([module[f'transformer.layers.{i}.attention.query_key_value.weight'].T.reshape(hidden_dim, head_num, num_splits, size_per_head).permute(0, 2, 1, 3).reshape(hidden_dim, 3 * local_dim) for i in range(layer_num)])

        local_dim = module['transformer.layers.0.attention.query_key_value.bias'].shape[0] // num_splits
        head_num = num_attention_heads // tensor_model_parallel_size
        size_per_head = local_dim // head_num
        w.extend([module[f'transformer.layers.{i}.attention.query_key_value.bias'].reshape(head_num, num_splits, size_per_head).permute(1, 0, 2).reshape(3, local_dim) for i in range(layer_num)])

        if self.dtype in ['int8', 'int4']:
            scale.extend([module[f'transformer.layers.{i}.attention.dense.weight_scale'] for i in range(layer_num)])
            weight.extend([module[f'transformer.layers.{i}.attention.dense.weight'] for i in range(layer_num)])
        else:
            weight.extend([module[f'transformer.layers.{i}.attention.dense.weight'].T for i in range(layer_num)])
        
        w.extend([module[f'transformer.layers.{i}.attention.dense.bias'] for i in range(layer_num)])
        w.extend([module[f'transformer.layers.{i}.input_layernorm.bias'] for i in range(layer_num)])
        w.extend([module[f'transformer.layers.{i}.input_layernorm.weight'] for i in range(layer_num)])


        local_dim = int(module['transformer.layers.0.mlp.dense_h_to_4h.weight'].shape[0] / 2)
        
        if self.dtype in ['int8', 'int4']:
            scale.extend([module[f'transformer.layers.{i}.mlp.dense_h_to_4h.weight_scale'][:local_dim] for i in range(layer_num)])
            weight.extend([module[f'transformer.layers.{i}.mlp.dense_h_to_4h.weight'][:local_dim,:] for i in range(layer_num)])
        else:
            weight.extend([module[f'transformer.layers.{i}.mlp.dense_h_to_4h.weight'][:local_dim,:].T for i in range(layer_num)])
        
        w.extend([module[f'transformer.layers.{i}.mlp.dense_h_to_4h.bias'][:local_dim] for i in range(layer_num)])
        
        if self.dtype in ['int8', 'int4']:
            scale.extend([module[f'transformer.layers.{i}.mlp.dense_h_to_4h.weight_scale'][local_dim:] for i in range(layer_num)])
            weight.extend([module[f'transformer.layers.{i}.mlp.dense_h_to_4h.weight'][local_dim:,:] for i in range(layer_num)])
        else:
            weight.extend([module[f'transformer.layers.{i}.mlp.dense_h_to_4h.weight'][local_dim:,:].T for i in range(layer_num)])
        
        w.extend([module[f'transformer.layers.{i}.mlp.dense_h_to_4h.bias'][local_dim:] for i in range(layer_num)])

        
        if self.dtype in ['int8', 'int4']:
            scale.extend([module[f'transformer.layers.{i}.mlp.dense_4h_to_h.weight_scale'] for i in range(layer_num)])
            weight.extend([module[f'transformer.layers.{i}.mlp.dense_4h_to_h.weight'] for i in range(layer_num)])
        else:
            weight.extend([module[f'transformer.layers.{i}.mlp.dense_4h_to_h.weight'].T for i in range(layer_num)])
        
        w.extend([module[f'transformer.layers.{i}.mlp.dense_4h_to_h.bias'] for i in range(layer_num)])
        w.extend([module[f'transformer.layers.{i}.post_attention_layernorm.bias'] for i in range(layer_num)])
        w.extend([module[f'transformer.layers.{i}.post_attention_layernorm.weight'] for i in range(layer_num)])

        w.append(module[f'transformer.final_layernorm.weight'])
        w.append(module[f'transformer.final_layernorm.bias'])
        w.append(module[f'transformer.word_embeddings.weight'])

        # Reshape
        def w_reshape(w,self_w):
            for i in range(len(w)):
                if w[i].nelement() > 0:
                    try:
                        self_w[i] = w[i].reshape(self_w[i].shape)
                    except:
                        raise RuntimeError("shape error")

        w_reshape(w, self.w)
        w_reshape(weight, self.weight)

        if self.dtype in ['int8', 'int4']:
            w_reshape(scale, self.scale)
        return True


class Glm(nn.Module):
    def __init__(self,
                 head_num, size_per_head,
                 vocab_size, rotary_embedding_dim, start_id, end_id, layer_num,
                 max_seq_len,
                 tensor_para_size, pipeline_para_size,
                 lib_path,
                 world_size,
                 rank,
                 tokenizer,
                 dtype="fp16"):
        super().__init__()
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.vocab_size = vocab_size
        self.rotary_embedding_dim = rotary_embedding_dim
        self.start_id = start_id
        self.end_id = end_id
        self.layer_num = layer_num
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.max_seq_len = max_seq_len
        self.use_sparse_gemm = False
        self.build_model = False

        self.tokenizer = tokenizer

        self.dtype = dtype
        self.dtype_id = {"fp32": 0, "fp16": 1, "int8": 2, "int4": 3}[dtype]

        assert dtype in ['fp16','int8','int4'], 'unsupport data_type'

        assert torch.cuda.is_available(), "CUDA is required for this model."

        assert head_num % tensor_para_size == 0, "head_num must be a multiple of tensor_para_size."
        assert layer_num % pipeline_para_size == 0, "layer_num must be a multiple of pipeline_para_size."

        # Load the C++ model.
        sys.path.append(os.path.abspath(lib_path))
        import libth_glm
        self.Glm = libth_glm.Glm

        # Prepare weights
        self.weights = GlmWeights(head_num, size_per_head, layer_num, vocab_size,
                                  max_seq_len, tensor_para_size, pipeline_para_size, dtype)

        # Prepare for tensor/pipeline parallel
        self.rank = rank
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        assert world_size == tensor_para_size * pipeline_para_size, "tensor_para_size * pipeline_para_size must be equal to world_size."

        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size

    def load(self, ckpt_path):
        is_load = self.weights.load(ckpt_path, tensor_para_rank=self.tensor_para_rank,
                                    pipeline_para_rank=self.pipeline_para_rank)
        return is_load

    def half(self):
        self.weights._map(lambda w: w.half())

    def bfloat16(self):
        self.weights._map(lambda w: w.bfloat16())

    def sparse(self):
        if not self.use_sparse_gemm:
            self.use_sparse_gemm = True

    def cuda(self):
        self.weights._map(lambda w: w.cuda(self.device))
        if self.build_model:
            del self.model
            self.build_model = False

        self.model = self.Glm(get_torch_default_comm(), self.rank, self.head_num, self.size_per_head, self.head_num * self.size_per_head * 8 // 3,
                                                            self.layer_num, self.vocab_size, self.rotary_embedding_dim, self.start_id, self.end_id,
                                                            self.tensor_para_size, self.pipeline_para_size, self.dtype_id, self.weights.w, self.weights.weight, self.weights.scale)
        self.build_model = True

    def init_model(self,
                output_len,
                beam_width=1,
                top_k=1,
                top_p=0.0,
                beam_search_diversity_rate=0.0,
                temperature=1.0,
                len_penalty=1.0,
                repetition_penalty=1.0,
                random_seed=0):
        if not self.build_model:
            self.cuda()
        self.model.init_model(output_len,
                                beam_width,
                                top_k,
                                top_p,
                                beam_search_diversity_rate,
                                temperature,
                                len_penalty,
                                repetition_penalty,
                                random_seed)

    def forward(self,
                start_ids,
                start_lengths,
                mask_positions,
                output_len,
                beam_width,
                strategy,
                regix=None):

        input_len = start_ids.size(1)
        assert input_len > 0, "input len must be larger than zero. For an unconditional case, use start_id as the first token."
        
        # Inputs to device
        start_ids = start_ids.cuda(self.device)
        start_lengths = start_lengths.cuda(self.device)
        mask_positions = mask_positions.cuda(self.device)
        
        # outputs: output_ids, output_lengths, output_cum_log_probs (optional)
        # model.forward is a deprecated interface where FT completes the entire inference process.

        # outputs = self.model.forward(start_ids,
        #                              start_lengths,
        #                              mask_positions,
        #                              return_cum_log_probs)
        
        # output_ids, output_lengths = outputs
        # return output_ids

        batch_size = start_ids.shape[0]
        max_len = input_len + output_len
        
        output_ids = torch.zeros([max_len,batch_size,1],dtype=torch.int32, device="cuda")
        output_ids_buf = torch.zeros([max_len,batch_size,1],dtype=torch.int32, device="cuda")
        logits_buf = torch.zeros([batch_size,1,self.vocab_size],dtype=torch.float32, device="cuda")
        parent_ids = torch.zeros([max_len,batch_size,1],dtype=torch.int32, device="cuda")
        sequence_lengths = torch.zeros([batch_size,1],dtype=torch.int32, device="cuda")
        cum_log_probs = torch.zeros([batch_size,1],dtype=torch.float32, device="cuda")

        k_cache_shape = [self.layer_num,
                                batch_size // beam_width,
                                beam_width,
                                self.head_num // self.tensor_para_size,
                                self.size_per_head // 8,
                                max_len,
                                8] # 16 / sizeof(fp16)
        v_cache_shape = [self.layer_num,
                                batch_size // beam_width,
                                beam_width,
                                self.head_num // self.tensor_para_size,
                                max_len,
                                self.size_per_head]
        key_cache = torch.zeros(k_cache_shape,dtype=torch.float16, device="cuda")
        value_cache = torch.zeros(v_cache_shape,dtype=torch.float16, device="cuda")
        
        self.model.encode(start_ids,
                            start_lengths,
                            mask_positions,
                            output_ids_buf,
                            logits_buf,
                            output_ids,
                            parent_ids,
                            sequence_lengths,
                            cum_log_probs,
                            key_cache,
                            value_cache,
                            0)

        history = [[]] * start_ids.shape[0]

        tokens = start_ids.reshape(batch_size // beam_width, beam_width, -1)
        for i in range(input_len, max_len):
            self.model.decode(key_cache.contiguous(), value_cache.contiguous(), i)
            logits = logits_buf.reshape(batch_size // beam_width, beam_width, -1)
            tokens, key_cache, value_cache = strategy.forward(logits, tokens, key_cache, value_cache)
            for j in range(batch_size):
                pred = tokens[j // beam_width][j % beam_width][-1]
                output_ids_buf[i][j][0] += pred
                sequence_lengths[j][0] += 1
                if regix:
                    history[j].append(int(pred.cpu().detach()))
                    detokenized = self.tokenizer.detokenize(history[j])
                    if regix.match(detokenized):
                        strategy._is_done[j // beam_width] = True
            if strategy.is_done:
                break

        return strategy.finalize(tokens)

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor
