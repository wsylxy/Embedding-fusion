# Mathematical semantic fusion for MIR
This code contains the training code of MaRF (Mathematical Reresentative-level Fusion) framework

## Training of formula encoder
To train the formula encoder

Edit path of training dataset and in ```cfgs/datasets/math_enc_dataset.yaml```, the training dataset needs to be preprocessed formula dataset for contrastive learning. 

(Due to data lost, the preprocessed dataset and checkpoint is not currently available, we are trying to recover it)

Here we provide the formula preprocessing code in ```preprocess/formula_process.py```, which can be used to process MathML formulas, the format of dataset needs to be in contrastive learning data format [QUERY_FORMULA] [POS_FORMULA] [NEG_FORMULA1] [NEG_FORMULA2] [NEG_FORMULA3]

Edit "MODEL.NAME" to be "math_enc_old" in ```cfgs/models/math_enc_dataset.yaml```
```
python train_model.py --cfg cfgs/models/mir_enc.yaml --dataset cfgs/datasets/math_enc_dataset.yaml
```

## Training of fusion model
To train the fusion model

Edit path of training dataset and in ```cfgs/datasets/math_enc_dataset.yaml```, the training dataset is contrastive learning dataset containing both context and formulas, the format needs to be [CTX1] [FOM1] [CTX2] [FOM2] [CTX3] [FOM3] .

Edit "MODEL.NAME" to be "fuse_text" in ```cfgs/models/math_enc_dataset.yaml```

Edit "CKPT.LAST" to be path of pretrained formula encoder in ```cfgs/models/mir_enc.yaml```

Edit "CKPT.BERT.MODEL" to be path of pretrained BERT context encoder in ```cfgs/models/mir_enc.yaml```

```
python train_model_fusion.py --cfg cfgs/models/mir_enc.yaml --dataset cfgs/datasets/math_enc_dataset.yaml
```
