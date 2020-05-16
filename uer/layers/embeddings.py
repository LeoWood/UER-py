# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm


class BertEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg):
        word_emb = self.word_embedding(src)

        pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))

        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb


class CscibertEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(CscibertEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)

        ## pos_embedding 嵌入词性标注特征(使用pkuseg词性标注，共计39种词性标签)
        self.add_pos = args.add_pos
        self.pos_embedding = nn.Embedding(39, args.emb_size)
        ## term_embedding 嵌入术语特征
        self.term_embedding = nn.Embedding(3, args.emb_size)

        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg):
        ## src 包含三个元素 word_index,pos_label,term_label
        # assert type(src) == tuple
        if self.add_pos:
            word_emb = self.word_embedding(src[0])
            pos_emb = self.pos_embedding(src[1])
            term_emb = self.term_embedding(src[2])
        else:
            word_emb = self.word_embedding(src)
            pos_emb = 0
            term_emb = 0

        position_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device,dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        seg_emb = self.segment_embedding(seg)

        if self.add_pos:
            emb = word_emb + pos_emb + term_emb + position_emb + seg_emb
        else:
            emb = word_emb + position_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb


class WordEmbedding(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(WordEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, _):
        emb = self.word_embedding(src)
        emb = self.dropout(self.layer_norm(emb))
        return emb