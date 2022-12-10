import sys
import ctypes
sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
from torch.nn.utils.rnn import pad_sequence
import random
import os
import re
import sys
import argparse
import datetime
import torch

torch.manual_seed(42)

from utils.glm import Glm
sys.setdlopenflags(sys.getdlopenflags() ^ ctypes.RTLD_GLOBAL)

from icetk_glm_130B import _IceTokenizer
tokenizer = _IceTokenizer()

from utils.strategies import BaseStrategy, BeamSearchStrategy

torch.set_printoptions(precision=20)

parser = argparse.ArgumentParser()
parser.add_argument('--layer_num', type=int, default=70,
                    help='number of layers')
parser.add_argument('--head_num', type=int, default=96,
                    help='head number')
parser.add_argument('--size_per_head', type=int, default=128,
                    help='size per head')
parser.add_argument('--vocab_size', type=int, default=150528,
                    help='vocab size')
parser.add_argument('--rotary_embedding_dim', type=int, default=64,
                    help='vocab size')
parser.add_argument('--tensor_para_size', type=int, default=8,
                    help='tensor parallel size')
parser.add_argument('--pipeline_para_size', type=int, default=1,
                    help='pipeline parallel size')
parser.add_argument('--ckpt_path', type=str,
                    help='path to the checkpoint file.')
parser.add_argument('--lib_path', type=str, default='./lib',
                    help='path to the fastertransformer lib folder.')
parser.add_argument('--start_id', type=int, default=150004,
                    help='start token id.')
parser.add_argument('--end_id', type=int, default=150001,
                    help='end token id.')
parser.add_argument('--max_seq_len', type=int, default=1024,
                    help='max sequence length for position embedding table.')
parser.add_argument('--data_type', type=str, choices=['fp16', 'int8', 'int4'], default='fp16')
parser.add_argument('--return_cum_log_probs', type=int, default=0, choices=[0, 1, 2],
                    help='Whether to compute the cumulative log probsbility of sentences.'
                            ' 0: do not return the cumulative log probs '
                            ' 1: return the cumulative log probs of generated sequences'
                            ' 2: return the cumulative log probs of sequences')
parser.add_argument('--world_size', type=int, default=8)
parser.add_argument('--local_rank', type=int, default=None,
                    help='local rank passed from distributed launcher.')

args = parser.parse_args()

layer_num = args.layer_num
head_num = args.head_num
size_per_head = args.size_per_head
vocab_size = args.vocab_size
rotary_embedding_dim = args.rotary_embedding_dim
tensor_para_size = args.tensor_para_size
pipeline_para_size = args.pipeline_para_size
start_id = args.start_id
end_id = args.end_id
max_seq_len = args.max_seq_len

if args.world_size > 1:
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=args.world_size,
        rank=args.local_rank,
        timeout=datetime.timedelta(days=30))
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
else:
    rank = 0
    world_size = 1

if rank == 0:
    print("\n=============== Arguments ===============")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    print("=========================================\n")

# Prepare model.
glm = Glm(head_num, size_per_head, vocab_size, rotary_embedding_dim, start_id, end_id,
                  layer_num, max_seq_len, tensor_para_size, pipeline_para_size,
                  lib_path=args.lib_path, world_size=args.world_size, rank=args.local_rank, tokenizer=tokenizer, dtype = args.data_type)
if not glm.load(ckpt_path=args.ckpt_path):
    print("[WARNING] Checkpoint file not found. Model loading is skipped.")

# most of the params in init_model are not used, just make FT happy
glm.init_model(512,# output_len,
                1, # beam_width
                1, # top_k,
                0, # top_p,
                0., # beam_search_diversity_rate,
                1.0, # temperature,
                1., # len_penalty,
                1., #repetition_penalty,
                42, # random_seed
                )

import json
from flask import Flask, request, jsonify, make_response
import torch.distributed as dist
from threading import Semaphore

gpu_sem = Semaphore(1)
app = Flask('glm-130b')

def tokenize(contexts, pad = True):
    def encode(raw_text):
        # add MASK
        generation_mask = "[MASK]" if "[MASK]" in raw_text else "[gMASK]"
        use_gmask = "[MASK]" not in raw_text

        mask_pattern = r"\[g?MASK\]"
        text_list = re.split(mask_pattern, raw_text)
        pattern_list = re.compile(mask_pattern).findall(raw_text)
        seq = []
        for i in range(len(pattern_list)):
            pattern = pattern_list[i]
            sub_text = text_list[i]
            seq.extend(tokenizer.tokenize(sub_text))
            seq.append(tokenizer.get_command(pattern))

        seq.extend(tokenizer.tokenize(text_list[-1]))

        if 'MASK]' not in raw_text:
            seq += [tokenizer.get_command(generation_mask)]
            raw_text += ' ' + generation_mask
        if not raw_text.endswith('MASK]'):
            seq = seq + [tokenizer.get_command('eos')]
        seq = seq + [tokenizer.get_command('sop')]
        return torch.IntTensor(seq), -1 if use_gmask else seq.index(tokenizer.get_command(generation_mask))

    def get_ids(contexts):
        start_ids, mask_positions = zip(*[encode(c) for c in contexts])
        start_lengths = torch.IntTensor([len(ids) for ids in start_ids])
        if pad:
            start_ids = pad_sequence(start_ids, batch_first=True, padding_value=0)
        return start_ids, start_lengths, torch.IntTensor(mask_positions)
    
    return get_ids(contexts)


def get_tokenize(config):
    config = json.loads(config)

    contexts = config["text"].splitlines()

    start_ids, start_lengths, mask_positions = tokenize(contexts)

    return make_response(jsonify({
        "start_ids": start_ids.tolist(),
        "start_lengths": start_lengths.tolist(),
        "mask_positions": mask_positions.tolist()
    }),200)

def get_generate(config):
    contexts = config["prompt"]

    if isinstance(contexts, str):
        contexts = [contexts]

    if config.get("num_beams") and config.get("sampling_strategy") == "BeamSearchStrategy":
        beam_width = config.get("num_beams")
    else:
        config["num_beams"] = 1
        beam_width = 1

    start_ids, _, _ = tokenize(contexts, pad = False)

    context_idx = sorted(range(len(contexts)), key=lambda k: len(start_ids[k]), reverse=True)

    new_contexts = []
    for i in range(len(contexts)):
        for _ in range(beam_width):
            new_contexts.append(contexts[context_idx[i]])
    contexts = new_contexts

    start_ids, start_lengths, mask_positions = tokenize(contexts)

    args = {}
    for i in ["seed", "max_tokens", "min_tokens", "sampling_strategy", "num_beams", "length_penalty", "no_repeat_ngram_size", "temperature", "top_k", "top_p", "regix"]:
        if config.get(i) != None:
            args[i] = config.get(i)

    with gpu_sem:
        res = predict(start_ids, start_lengths, mask_positions, **args)

    res_new = [None] * len(res)
    for i in range(len(res)):
        res_new[context_idx[i]] = res[i]
    res = res_new

    return make_response(jsonify({"text": res}),200)

if __name__ == "__main__":

    def get_res(tokens_batch, start_lengths):
        res = []
        if tokens_batch is not None:
            for i, tokens in enumerate(tokens_batch):
                res.append([])
                try:
                    for beam_id in range(len(tokens)):
                        token = tokens[beam_id].tolist()
                        token = token[start_lengths[0]:] # exclude context input from the output
                        if 20002 in token:
                            token = token[:token.index(20002)]
                        if 150005 in token:
                            token = token[:token.index(150005)]
                        res[-1].append(tokenizer.detokenize(token))
                except:
                    pass
        return res

    def predict(start_ids, start_lengths, mask_positions, seed=42, max_tokens=64, min_tokens=0, sampling_strategy='BaseStrategy', 
    num_beams=1, length_penalty=0.9, no_repeat_ngram_size=3, 
    temperature=1, top_k=5, top_p=0, regix=None):

        if start_ids.size(1) + max_tokens > max_seq_len:
            return ["length too long"]

        if torch.distributed.get_rank() == 0:
            dist.broadcast_object_list([start_ids, start_lengths, mask_positions, seed, max_tokens, min_tokens, sampling_strategy, num_beams, length_penalty, no_repeat_ngram_size, temperature, top_k, top_p, regix], src=0)

        try:
            if regix:
                regix = re.compile(regix)
        except:
            regix = None
            
        end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]
        batch_size = start_ids.shape[0]

        if sampling_strategy == "BaseStrategy":
            strategy = BaseStrategy(batch_size=batch_size, temperature=temperature, top_k=top_k, top_p=top_p,
                                    end_tokens=end_tokens)
        elif sampling_strategy == "BeamSearchStrategy":
            strategy = BeamSearchStrategy(
                batch_size=batch_size // num_beams,
                num_beams=num_beams,
                length_penalty=length_penalty,
                consider_end=True,
                end_tokens=end_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                min_tokens=min_tokens,
            )
        else:
            return [f"unknown strategy {sampling_strategy}"]

        # with the same seed, no need to synchronize random numbers between processes
        torch.manual_seed(seed)

        tokens_batch = glm(start_ids,
                        start_lengths,
                        mask_positions,
                        max_tokens,
                        num_beams,
                        strategy,
                        regix)

        res = get_res(tokens_batch, start_lengths)
        if torch.distributed.get_rank() == 0:
            print(tokens_batch)
            print(res)
        return res

    # from https://github.com/hanyullai/GLM-130B/commit/a0ad56b76650eee679123fcc26bb92d2b3b49cb2
    if torch.distributed.get_rank() == 0:
        get_generate({"prompt": "test"})
    else:
        while True:
            info = [None, None, None, None, None, None, None, None, None, None, None, None, None, None]
            dist.broadcast_object_list(info, src=0)

            start_ids, start_lengths, mask_positions, seed, max_tokens, min_tokens, sampling_strategy, num_beams, length_penalty, no_repeat_ngram_size, temperature, top_k, top_p, regix = info

            predict(start_ids, start_lengths, mask_positions, seed, max_tokens, min_tokens, sampling_strategy, 
                num_beams, length_penalty, no_repeat_ngram_size, 
                temperature, top_k, top_p, regix)