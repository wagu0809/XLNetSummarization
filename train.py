#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division


import argparse
import os
from others.logging import init_logger
from train_abstractive import validate_abs, train_abs, test_abs, test_text_abs

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-encoder", default='xlnet', type=str, choices=['bert', 'xlnet'])
    parser.add_argument("-decoder", default='xlnet', type=str, choices=['xlnet', 'gpt2', 'transformer'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-data_path", default='./data/xlnet/cnndm')
    parser.add_argument("-model_path", default='./model/')
    parser.add_argument("-result_path", default='./results/cnndm')
    parser.add_argument("-temp_dir", default='./temp')
    parser.add_argument('-log_file', default='./logs/abs_xlnet_cnndm_train.log')
    # parser.add_argument('-log_file', default='./logs/abs_test_xlnet.log')

    parser.add_argument("-batch_size", default=1, type=int)
    parser.add_argument("-test_batch_size", default=100, type=int)

    parser.add_argument("-max_pos", default=128, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-lr_enc", default=2e-3, type=float)  # learning rate for encoder
    parser.add_argument("-lr_dec", default=2e-1, type=float)  # learning rate for decoder
    parser.add_argument("-use_pre_emb", type=str2bool, nargs='?', const=True, default=True)  # use embeddings of pretrained model

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=True)
    # TODO: after changing all finetune_bert to finetune_encoder, delete it
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-finetune_encoder", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.1, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=12, type=int)
    parser.add_argument("-dec_ff_size", default=3072, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)
    parser.add_argument("-mem_len", default=128, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha", default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps_bert", default=20000, type=int)
    parser.add_argument("-warmup_steps_enc", default=10000, type=int)
    parser.add_argument("-warmup_steps_dec", default=5000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=10000, type=int)
    parser.add_argument("-accum_count", default=10, type=int)
    parser.add_argument("-report_every", default=50, type=int)
    parser.add_argument("-train_steps", default=200000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-device_id', default=0, type=int)
    parser.add_argument('-device', default='cuda', type=str)

    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if args.mode == 'train':
        train_abs(args, device_id)
    elif args.mode == 'validate':
        validate_abs(args, device_id)
    if args.mode == 'test':
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        test_abs(args, device_id, cp, step)
    elif args.mode == 'test_text':
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
            test_text_abs(args, device_id, cp, step)
