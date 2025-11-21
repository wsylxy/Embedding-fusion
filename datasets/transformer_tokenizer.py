from torch import Tensor
from typing import List

import torch


class Tokenizer:
    def __init__(self, file_path: str, mode='fuse') -> None:
        self.soe = "SOE"
        self.eoe = "EOE"
        self.pad = "PAD"
        self.Q = "[Q]"
        self.D = "[D]"
        self.longest_len = 256
        self.mode = mode
        if self.mode == 'fuse':
            self.vocabs = [self.pad, self.soe]
        elif self.mode == 'transformer' or self.mode == 'fuse_maxsim':
            self.vocabs = [self.pad, self.soe, self.Q, self.D]
        file = open(file=file_path, mode='r', encoding='utf-8')
        for line in file:
            self.vocabs.append(line.strip('\n').split('\t')[0])
        file.close()
        self.vocabs.append(self.eoe)  # put end-of-expr in the end

        self.word2idx = {w: i for i, w in enumerate(self.vocabs)}
        self.idx2word = {i: w for i, w in enumerate(self.vocabs)}
        
        return

    def encode(self, expr: str) -> Tensor:
        tokens = []
        # print(expr)
        expr = expr.split(sep=' ')
        # if self.mode == 'transformer':
        if len(expr)>self.longest_len:
            expr = expr[:self.longest_len]
        for word in expr:
            try:
                tokens.append(self.word2idx[word])
            except:
                tokens.append(self.word2idx["spec"])

        tokens = torch.cat(
            tensors=(
                torch.tensor(data=[self.word2idx["SOE"]], dtype=torch.int64),
                torch.tensor(data=tokens, dtype=torch.int64),
                torch.tensor(data=[self.word2idx["EOE"]], dtype=torch.int64),
            ),
            dim=0,
        )

        return tokens

    def decode(self, tokens: Tensor) -> str:
        expr = []

        for token in tokens:
            expr.append(self.idx2word[token])
            if token == self.word2idx[self.eoe]:
                break

        expr = " ".join(expr)

        return expr
