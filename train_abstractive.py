#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import glob
import os
import random
import time

import torch
from transformers import XLNetTokenizer, AlbertTokenizer

# import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.loss import abs_loss
from models.model_builder import AbsSummarizer
from models.predictor import build_predictor
from models.trainer import build_trainer
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def get_symbol_and_tokenizer(encoder, temp_dir):
    if encoder == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True, cache_dir=temp_dir)
        vocab = tokenizer.get_vocab()
        # '<unk>': 0, '<s>': 1, '</s>': 2, '<cls>': 3, '<sep>': 4, '<pad>': 5, '<mask>': 6, '<eod>': 7, '<eop>': 8,
        symbols = {'UNK': vocab['<unk>'], 'BOS': vocab['<s>'], 'CLS': vocab['<cls>'], 'SEP': vocab['<sep>'], 'EOD': ['<eod>'],
                   'EOP': vocab['<eop>'], 'PAD': vocab['<pad>'], 'EOS': vocab['</s>']}
    else:
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=temp_dir)
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True, cache_dir=temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
                   'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    return symbols, tokenizer


def validate_abs(args, device_id):
    timestep = 0
    if args.test_all:
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            if args.test_start_from != -1 and step < args.test_start_from:
                xent_lst.append((1e6, cp))
                continue
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if i - max_step > 10:
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:5]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test_abs(args, device_id, cp, step)
    else:
        while True:
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if cp_files:
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if not os.path.getsize(cp) > 0:
                    time.sleep(60)
                    continue
                if time_of_cp > timestep:
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    test_abs(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if cp_files:
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if time_of_cp > timestep:
                    continue
            else:
                time.sleep(300)


def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if pt != '':
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if k in model_flags:
            setattr(args, k, opt[k])
    print(args)

    symbols, tokenizer = get_symbol_and_tokenizer(args.encoder, args.temp_dir)
    model = AbsSummarizer(args, device, checkpoint, symbols=symbols)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False, tokenizer=tokenizer)
    valid_loss = abs_loss(model.generator, symbols, model.vocab_size, train=False, device=device)

    trainer = build_trainer(args, device_id, model, None, valid_loss)
    stats = trainer.validate(valid_iter, step)
    return stats.xent()


def test_abs(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if pt != '':
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if k in model_flags:
            setattr(args, k, opt[k])
    print(args)
    symbols, tokenizer = get_symbol_and_tokenizer(args.encoder, args.temp_dir)
    model = AbsSummarizer(args, device, checkpoint, symbols=symbols)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True, tokenizer=tokenizer)

    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, step)


def test_text_abs(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if pt != '':
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)
    symbols, tokenizer = get_symbol_and_tokenizer(args.encoder, args.temp_dir)
    model = AbsSummarizer(args, device, checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)

    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, step)


def train_abs(args, device_id):
    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    if args.load_from_extractive != '':
        logger.info('Loading bert from extractive model %s' % args.load_from_extractive)
        bert_from_extractive = torch.load(args.load_from_extractive, map_location=lambda storage, loc: storage)
        bert_from_extractive = bert_from_extractive['model']
    else:
        bert_from_extractive = None
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    symbols, tokenizer = get_symbol_and_tokenizer(args.encoder, args.temp_dir)

    model = AbsSummarizer(args, device, checkpoint, bert_from_extractive, symbols=symbols)
    if args.sep_optim:
        optim_enc = model_builder.build_optim_enc(args, model, checkpoint)
        optim_dec = model_builder.build_optim_dec(args, model, checkpoint)
        optim = [optim_enc, optim_dec]
    else:
        optim = [model_builder.build_optim(args, model, checkpoint)]

    logger.info(model)


    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                      shuffle=True, is_test=False, tokenizer=tokenizer)

    train_loss = abs_loss(model.generator, symbols, model.vocab_size, device, train=True,
                          label_smoothing=args.label_smoothing)

    trainer = build_trainer(args, device_id, model, optim, train_loss)

    trainer.train(train_iter_fct, args.train_steps)
