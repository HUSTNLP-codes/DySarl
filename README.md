# DySarl: Dynamic Structure-Aware Representation Learning for Multimodal Knowledge Graph Reasoning

This is the released codes of the following paper:

Kangzheng Liu, Feng Zhao, Yu Yang, and Guandong Xu. DySarl: Dynamic Structure-Aware Representation Learning for Multimodal Knowledge Graph Reasoning. MM 2024.

![DySarl](https://github.com/Liudaxian1/FIG/blob/main/DySarl.png)

## Environment

```shell
python==3.10.9
torch==2.2.1+cu118
dgl==2.1.0+cu118
numpy==1.26.4
```

## Instruction

- `src`: Python scripts.
- `src_data`: Source triplet data of MKGs.
- `data`: Triplet data of MKGs processed by `src/process_datasets.py`.
- `pre_train`: Pre-trained auxiliary modal (visual and linguistic) features of MKGs. One can also obtain these pre-trained auxiliary modal features from [OTKGE](https://github.com/Lion-ZS/OTKGE) (Thanks to the data preprocessing).
- `results`: Model files to replicate the reported results in our paper.


## Training Command

```shell
cd src
CUDA_VISIBLE_DEVICES=0 python main.py --model two --dataset WN9IMG --bias learn --s-delta-ind --n-head 2 --rank 100
```

```shell
cd src
CUDA_VISIBLE_DEVICES=1 python main.py --model two --dataset FBIMG --bias learn --s-delta-ind --n-head 2 --rank 100
```

## Testing Command

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model two --dataset WN9IMG --bias learn --s-delta-ind --n-head 2 --rank 100 --test
```

```shell
CUDA_VISIBLE_DEVICES=1 python main.py --model two --dataset FBIMG --bias learn --s-delta-ind --n-head 2 --rank 100 --test
```

## Acknowledgements

The source codes take [ReTIN](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12186) as the backbone to implement our proposed method. Please cite both our work and [ReTIN](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12186) if you would like to use the source codes.
