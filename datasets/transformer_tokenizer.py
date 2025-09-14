from torch import Tensor
from typing import List

import torch


class Tokenizer:
    def __init__(self, file_path: str) -> None:
        self.soe = "SOE"
        self.eoe = "EOE"
        self.pad = "PAD"
        self.Q = "[Q]"
        self.D = "[D]"

        self.vocabs = [self.pad, self.soe]
        # self.vocabs = [self.pad, self.soe, self.Q, self.D]
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
        for word in expr.split(sep=' '):
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
