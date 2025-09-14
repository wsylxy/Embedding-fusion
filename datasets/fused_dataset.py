from typing import Dict, List
from .transformer_tokenizer import Tokenizer
from torch import Tensor
import torch
import string
from .registry import register_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizer


class FuseData(Dataset):
    def __init__(
            self,
            file_path: str,
            tokenizer_text: BertTokenizer,
            tokenizer_formula: Tokenizer,
            max_text_len: int,
    ) -> None:
        super().__init__()
        self.exprs = []
        self.tokenizer_text = tokenizer_text
        self.tokenizer_formula = tokenizer_formula
        self.max_text_len = max_text_len
        print(file_path)
        file = open(file=file_path, mode='r', encoding='utf-8')
        for line in file:
            expr = line.strip().split(sep='\t')
            self.exprs.append(expr)
        file.close()

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, idx: int) -> List[str]:
        return self.exprs[idx]

    def collate_fn(
            self,
            batch: List[List[Tensor]],
    ) -> Dict[str, Tensor]:
        text, formula = [], []
        for i, line in enumerate(batch):
            for k, item in enumerate(line):
                if k%2 == 0:
                    if k == 0:
                        text.append(f"[unused0] {item}")
                    elif k == 2 or k == 4:
                        text.append(f"[unused1] {item}")
                else:
                    formula.append(item)
        print("text:", text)
        
        print("formula:", formula)
        batch_enc_text = self.tokenizer_text(
            text=text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        punct_ids = []
        for p in string.punctuation:
            punct_id = self.tokenizer_text(text=p, add_special_tokens=False)
            punct_ids.append(punct_id['input_ids'][0])

        input_ids = batch_enc_text['input_ids']
        print("text ids:", input_ids)
        print("text shape:", input_ids.shape)
        print("text mask:", batch_enc_text["attention_mask"])
        print("text mask shape:", batch_enc_text["attention_mask"].shape)
        punct_ids = torch.tensor(punct_ids, device=input_ids.device)
        punct_mask = torch.isin(input_ids, punct_ids)
        punct_mask = (~punct_mask).to(dtype=torch.int64)
        batch_enc_text['punct_mask'] = punct_mask

        src_formula = [self.tokenizer_formula.encode(expr) for expr in formula]
        # print("formula ids:", src_formula)
        src_formula = pad_sequence(
            sequences=src_formula,
            batch_first=True,
            padding_value=self.tokenizer_formula.word2idx["PAD"],
        )
        print("formula ids:", src_formula)
        print("formula shape:", src_formula.shape)
        # [batch_size, n_heads, 1, seq_len]
        src_mask = torch.eq(input=src_formula, other=self.tokenizer_formula.word2idx["PAD"]) \
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.bool)
        print("formula mask:", src_mask)
        print("formula mask shape:", src_mask.shape)
        batch_enc_formula = {"src": src_formula, "src_mask": src_mask}

        

        return {'text': batch_enc_text,
                'formula': batch_enc_formula,
                }



