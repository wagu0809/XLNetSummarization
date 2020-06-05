import copy
import math
import torch
import torch.nn as nn
# from pytorch_transformers import BertModel, BertConfig
from transformers import XLNetModel, XLNetConfig, BertModel, BertConfig, AlbertModel, AlbertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    return optim


def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def build_optim_enc(args, model, checkpoint):
    """ Build optimizer for encoder """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_enc, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_enc)
    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('encoder.model')]

    optim.set_parameters(params)

    return optim


def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


#  查阅transformers的文档修改其返回值
class XLNet(nn.Module):
    def __init__(self, args, large, temp_dir, finetune=False, symbols=None):
        super(XLNet, self).__init__()

        self.args = args
        self.symbols = symbols
        self.device = "cpu" if args.visible_gpus == '-1' else "cuda"
        self.model = XLNetModel.from_pretrained('xlnet-base-cased', cache_dir=temp_dir)
        self.model.mem_len = self.args.mem_len
        self.model.config.output_hidden_states = True

        self.finetune = finetune

    def forward(self, src, segs, mask):
        # TODO: 查阅transformers的文档修改其返回值
        src_length = src.shape[1]
        if src_length <= self.args.max_pos:
            chunk_num = 1
            whole_src = copy.deepcopy(src)
            whole_att = copy.deepcopy(mask)
        else:
            chunk_num = math.ceil(src_length / self.args.max_pos)
            final_length = self.args.max_pos * chunk_num
            pads_needed = final_length - src_length
            if pads_needed > 0:
                pads = torch.full((pads_needed,), self.symbols['PAD'], dtype=torch.long, device=self.device).unsqueeze(0)
                whole_src = torch.cat((pads, src), dim=-1)
                pads_att = torch.full((pads_needed,), 0, dtype=torch.long, device=self.device).unsqueeze(0)
                whole_att = torch.cat((pads_att, mask), dim=-1)
            else:
                whole_src = copy.deepcopy(src)
                whole_att = copy.deepcopy(mask)
        _mems = None
        all_hidden_states = []
        inputs = dict()
        for i in range(chunk_num):

            start_id = self.args.max_pos * i
            input_id = whole_src[:, start_id: start_id + self.args.max_pos]
            attention_mask = whole_att[:, start_id: start_id + self.args.max_pos]
            inputs["input_ids"] = input_id
            inputs["attention_mask"] = attention_mask
            if _mems is not None:
                inputs['mems'] = _mems

            if self.finetune:
                top_vec, _mems  = self.model(**inputs)
            else:
                self.eval()
                with torch.no_grad():
                    top_vec, _mems  = self.model(**inputs)

            all_hidden_states.append(top_vec.to('cpu'))
        top_vec = sum(all_hidden_states)
        top_vec = top_vec.to('cuda')
        return top_vec


class Bert(nn.Module):
    def __init__(self, args, large, temp_dir, finetune=False, symbols=None):
        super(Bert, self).__init__()
        if large:
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
            # self.model = AlbertModel.from_pretrained('albert-base-v2', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
            # self.model = AlbertModel.from_pretrained('albert-base-v2', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if self.finetune:
            top_vec, _ = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
        return top_vec


def pre_models(model_type):
    models = {'bert': Bert,
              'xlnet': XLNet}
    return models[model_type]


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, from_extractive=None, symbols=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.symbols = symbols

        # TODO: 根据args.encoder是bert还是xlnet进行区分， 构建encoder
        self.pre_model = pre_models(args.encoder)  # 选出bert或者xlnet的类
        self.encoder = self.pre_model(args, args.large, args.temp_dir, args.finetune_encoder, self.symbols)  # encoder is bert or xlnet
        # self.decoder = XLNet(args.large, args.temp_dir, args.finetune_encoder)  # decoder is xlnet

        if args.max_pos > 512:
            if args.encoder == 'bert':
                my_pos_embeddings = nn.Embedding(args.max_pos, self.encoder.model.config.hidden_size)
                my_pos_embeddings.weight.data[:512] = self.encoder.model.embeddings.position_embeddings.weight.data
                my_pos_embeddings.weight.data[512:] = self.encoder.model.embeddings.position_embeddings.weight.data[-1][
                                                      None, :].repeat(args.max_pos - 512, 1)
                self.encoder.model.embeddings.position_embeddings = my_pos_embeddings


        if from_extractive is not None:
            self.encoder.model.load_state_dict(
                dict([(n[11:], p) for n, p in from_extractive.items() if n.startswith(args.encoder + '.model')]), strict=True)

        self.vocab_size = self.encoder.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.model.config.hidden_size, padding_idx=0)
        if self.args.share_emb:
            tgt_embeddings.weight = copy.deepcopy(self.encoder.model.word_embedding.weight)

        # TODO: create decoder, options: TransformerDecoder, XLNet, GPT-2
        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        # TODO： create generator, options: GPT-2, XLNet
        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if args.use_pre_emb:
                if args.encoder == 'bert':
                    tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.model.config.hidden_size, padding_idx=0)
                    tgt_embeddings.weight = copy.deepcopy(self.encoder.model.embeddings.word_embeddings.weight)
                if args.encoder == 'xlnet':
                    tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.model.config.d_model, padding_idx=0)
                    tgt_embeddings.weight = copy.deepcopy(self.encoder.model.word_embedding.weight)

                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.encoder(src, segs, mask_src)
        # if self.args.encoder == 'xlnet':
        #     src = src[:, :self.args.max_pos]
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
