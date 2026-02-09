# DHPRec
![model](./asset/fracwork.svg)
## Overview
Sequential Recommender Systems aim to capture dynamic user preferences by mining historical dependencies. However, existing methods typically adopt a truncation strategy that focuses mainly on the most recent interactions. From a cognitive psychology perspective, real-world user behavior is jointly driven by immediate intent and accumulated historical habits. The truncation strategy ignores information in the dimension of historical preferences and is limited entirely to recent information. A straightforward solution is to feed the user’s
entire historical sequence into the model. However, as the sequence length increases, the attention weights for each item are diluted. The dilution confronts two critical challenges: noise interference amplified and the inability to focus on specific regularities caused by attention dilution. To address these issues, we propose \textbf{DHPRec} (\underline{D}enoised \underline{H}istorical \underline{P}atterns for Sequential \underline{Rec}ommendation), a novel framework designed to bridge immediate intent and specific long-term patterns. DHPRec introduces a Frequency-Domain Feature Refinement mechanism to filter high-frequency noise and extract stable long-term representations. Furthermore, we design a Pattern-based Anchor-Guided Fusion mechanism,which extracts pattern units representing long-term regularities and utilizes an anchor to effectively extract the global specific historical preferences. Finally, a learnable gating mechanism is employed to balance the contribution of the immediate intent and the specific historical preferences. Extensive experiments on four real-world datasets demonstrate that DHPRec significantly outperforms state-of-the-art baselines, achieving an average relative improvement of 8.1\% across all metrics, with gains reaching up to 14.6\% in NDCG@10 on the Scientific dataset.
## Dependency
Please install required packages via `pip install -r requirement.txt`

## Quick Start

### Data Download and Preprocess

Download the following files into `./dataset/{dataset}`:

- Musical_Instruments:
[train](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Musical_Instruments.train.csv.gz),
[valid](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Musical_Instruments.valid.csv.gz),
[test](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Musical_Instruments.test.csv.gz),
[meta](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Musical_Instruments.jsonl.gz)

- Video_Games:
[train](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Video_Games.train.csv.gz),
[valid](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Video_Games.valid.csv.gz),
[test](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Video_Games.test.csv.gz),
[meta](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz)

- Industrial_and_Scientific:
[train](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Industrial_and_Scientific.train.csv.gz),
[valid](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Industrial_and_Scientific.valid.csv.gz),
[test](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Industrial_and_Scientific.test.csv.gz),
[meta](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Industrial_and_Scientific.jsonl.gz)

- Baby_Products:
[train](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Baby_Products.train.csv.gz),
[valid](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Baby_Products.valid.csv.gz),
[test](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Baby_Products.test.csv.gz),
[meta](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Baby_Products.jsonl.gz)

Run the command:
```
bash run_preprocess.sh
```
### Best configurations for each dataset

```
python main.py \
    --dataset=Musical_Instruments \
    --lr=0.001 \
    --neg_num=24000 \
    --text_types title brand features categories description \
    --mask_ratio=0.5 \
    --cl_weight=0.4 \
    --mlm_weight=0.6 \
    --data_path=./dataset \
    --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
    --code_level=20 \
    --n_codes_per_lel=256 \
    --max_his_len=50 \
    --batch_size=100 \
    --dropout_prob=0.3 \
    --dropout_prob_cross=0.3 \
    --n_layers=2 \
    --n_heads=4 \
    --embedding_size=128 \
    --hidden_size=512\
    --early_stop=100\
    --log_dir="./logs"\
    --device=cuda:1




python main.py \
    --dataset=Video_Games \
    --lr=0.001 \
    --neg_num=25000 \
    --text_types title brand features categories description \
    --mask_ratio=0.5 \
    --cl_weight=0.5 \
    --mlm_weight=0.3 \
    --data_path=./dataset \
    --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
    --code_level=20 \
    --n_codes_per_lel=256 \
    --max_his_len=50 \
    --batch_size=90 \
    --dropout_prob=0.2 \
    --dropout_prob_cross=0.1 \
    --n_layers=2 \
    --n_heads=2 \
    --embedding_size=128 \
    --hidden_size=512\
    --early_stop=100\
    --log_dir="./logs/Video_Games/傅里叶_分层"\
    --resume="/home/yejinxuan/yejinxuan/DHPRec/myckpt/Video_Games/Jan-11-2026_02-18-e18370_mlm0.3_cl0.5_maskratio0.5_drop0.2_dpcross0.1/best_model.pth"\
    --device=cuda:1


python main.py \
    --dataset=Industrial_and_Scientific \
    --lr=0.0005 \
    --neg_num=25000 \
    --text_types title brand features categories description \
    --mask_ratio=0.5 \
    --cl_weight=0.4 \
    --mlm_weight=0.2 \
    --data_path=./dataset \
    --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
    --code_level=20 \
    --n_codes_per_lel=256 \
    --max_his_len=50 \
    --batch_size=100 \
    --dropout_prob=0.4 \
    --dropout_prob_cross=0.1 \
    --n_layers=2 \
    --n_heads=2 \
    --embedding_size=128 \
    --hidden_size=512\
    --early_stop=100\
    --log_dir="./logs/Industrial_and_Scientific/分析_时间为12小时"\
    --device=cuda:1
python main.py \
    --dataset=Baby_Products \
    --lr=0.0005 \
    --neg_num=25000 \
    --text_types title brand features categories description \
    --mask_ratio=0.5 \
    --cl_weight=0.5 \
    --mlm_weight=0.3 \
    --data_path=./dataset \
    --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
    --code_level=20 \
    --n_codes_per_lel=256 \
    --max_his_len=50 \
    --batch_size=100 \
    --dropout_prob=0.2 \
    --dropout_prob_cross=0.2 \
    --n_layers=2 \
    --n_heads=2 \
    --embedding_size=128 \
    --hidden_size=512\
    --early_stop=100\
    --log_dir="./logs/Baby_Products/分析_时间为12小时"\
    --device=cuda:1

```






