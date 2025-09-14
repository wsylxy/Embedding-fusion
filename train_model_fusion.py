#!/usr/bin/env python3


import argparse
from config import get_config, DEVICE
from criterions.registry import build_criterion
from datasets.registry import build_dataset
from lr_scheduler import build_scheduler
from models.registry import build_model
from optimizer import build_optimizer
from torch.utils.data import DataLoader
from train_fusion import train_model
from transformers import BertTokenizer, BertConfig, BertModel
from transformer import Transformer
from bert import Bert
from datasets.fused_dataset import FuseData
from datasets.transformer_tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(prog='Math IR')
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        metavar="FILE",
        help='path to config file',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        metavar="FILE",
        help='path to dataset config file',
    )
    args, unparsed = parser.parse_known_args()
    cfg = get_config(args=args)

    # tokenizer
    """text tokenizer"""
    tokenizer_text = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.CKPT.BERT.TOKENIZER
    )
    tokenizer_text.add_special_tokens({
        'additional_special_tokens': ['[unused0]', '[unused1]']
    })
    """formula tokenizer"""
    tokenizer_formula = Tokenizer(file_path=cfg.DATA.VOCAB_FILE)

    # dataset
    dataset = FuseData(
        file_path=cfg.DATA.MATH,
        tokenizer_text=tokenizer_text,
        tokenizer_formula=tokenizer_formula,
        max_text_len=cfg.MODEL.BERT.MAX_SEQ_LEN,
    )

    # dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.LOADER.TRAIN.BATCH_SIZE,
        shuffle=cfg.LOADER.TRAIN.SHUFFLE,
        num_workers=cfg.LOADER.TRAIN.NUM_WORKERS,
        collate_fn=dataset.collate_fn,
        pin_memory=cfg.LOADER.TRAIN.PIN_MEMORY,
    )
    # for batch in dataloader:
    #     input_ids = batch["input_ids"]
    #     decoded = dataset.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    #     print(input_ids.shape)
    #     print(decoded)
    #     print(batch["attention_mask"])
    # return

    # model
    config_text = BertConfig.from_json_file(json_file=cfg.CKPT.BERT.CFG)
    model_text = Bert(
        config=config_text,
        add_pooling_layer=cfg.MODEL.BERT.ADD_POOLING,
        reduce_dim=cfg.MODEL.BERT.REDUCE_DIM,
        dim=cfg.MODEL.BERT.DIM,
    )


    model_formula = Transformer(
        vocab_size=len(tokenizer_formula.vocabs),
        dim=cfg.MODEL.MATH_ENC.DIM,
        n_layers=cfg.MODEL.MATH_ENC.N_LAYERS,
        n_heads=cfg.MODEL.MATH_ENC.N_HEADS,
        n_kv_heads=cfg.MODEL.MATH_ENC.N_KV_HEADS,
        base=cfg.MODEL.MATH_ENC.BASE,
        max_seq_len=cfg.MODEL.MATH_ENC.MAX_SEQ_LEN,
        multiple_of=cfg.MODEL.MATH_ENC.MULTIPLE_OF,
        ffn_dim_multiplier=cfg.MODEL.MATH_ENC.FFN_DIM_MULTIPLIER,
        norm_eps=cfg.MODEL.MATH_ENC.NORM_EPS,
    )

    # model_formula = build_model(cfg=cfg)

    # 冻结 BERT
    for param in model_text.parameters():
        param.requires_grad = False

    # 冻结 Transformer
    for param in model_formula.parameters():
        param.requires_grad = False

    model_fuse = build_model(cfg=cfg)
    # optimizer
    optimizer = build_optimizer(cfg=cfg, model=model_fuse)

    # lr scheduler
    lr_scheduler = build_scheduler(cfg=cfg, optimizer=optimizer)

    # criterion
    criterion = build_criterion(cfg=cfg)

    train_model(
        model_text=model_text,
        ckpt_text=cfg.CKPT.BERT.MODEL,
        model_formula=model_formula,
        ckpt_formula=cfg.CKPT.LAST,
        model_fuse=model_fuse,
        ckpt_fuse=cfg.CKPT.FUSE,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        postprocess=cfg.POSTPROCESS.NAME,
        n_exprs=cfg.DATA.N_EXPRS,
        criterion=criterion,
        max_norm=cfg.TRAIN.MAX_NORM,
        device=DEVICE,
        n_epochs=cfg.TRAIN.N_EPOCHS,
        dataloader=dataloader,
        save_every_n_iters=cfg.TRAIN.SAVE_N_ITERS,
    )

    return


if __name__ == '__main__':
    main()
