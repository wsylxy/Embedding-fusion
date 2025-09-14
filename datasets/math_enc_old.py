from typing import Dict, List
from torch import Tensor
import torch
from .registry import register_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from .transformer_tokenizer import Tokenizer



class Math_enc_dataset_old(Dataset):
    def __init__(
            self,
            file_path: str,
            tokenizer: Tokenizer,
            max_seq_len: int,
    ) -> None:
        super().__init__()
        self.exprs = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.maxline = 6
        file = open(file=file_path, mode='r', encoding='utf-8')
        for i, line in enumerate(file):
            if self.maxline and i>=self.maxline:
                break
            expr = line.strip().split(sep='\t')[:3]
            expr[0] = f"[Q] {expr[0]}"
            expr[1] = f"[D] {expr[1]}"
            expr[2] = f"[D] {expr[2]}"
            # expr[3] = f"[D] {expr[3]}"
            # expr[4] = f"[D] {expr[4]}"
            self.exprs.append(expr)
            # print(expr)
        file.close()

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, idx: int) -> List[str]:
        src = self.exprs[idx]
        src_tokens = [self.tokenizer.encode(expr=e) for e in src]
        # print(src_tokens)
        return src_tokens

    def collate_fn(
            self,
            batch: List[List[Tensor]],
    ) -> Dict[str, Tensor]:
        exprs = [expr for item in batch for expr in item]
        src = pad_sequence(
            sequences=exprs,
            batch_first=True,
            padding_value=self.tokenizer.word2idx["PAD"],
        )
        # https://gmongaras.medium.com/how-do-self-attention-masks-work-72ed9382510f
        # [batch_size, n_heads, 1, seq_len]
        src_mask = torch.eq(input=src, other=self.tokenizer.word2idx["PAD"]) \
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.bool)
        batch_enc = {"input_ids": src, "attention_mask": src_mask}
        return batch_enc

@register_dataset(name="math_enc_old")
def build_dataset(cfg) -> Dataset:
    tokenizer = Tokenizer(file_path=cfg.DATA.VOCAB_FILE)
    return Math_enc_dataset_old(
        file_path=cfg.DATA.MATH,
        tokenizer=tokenizer,
        max_seq_len=cfg.MODEL.MATH_ENC.MAX_SEQ_LEN,
    )

