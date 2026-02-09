# python preprocess.py --dataset Musical_Instruments --his_len 50

# python encode_emb.py --dataset Musical_Instruments --text_types title brand features categories description --gpu_id 0

# cd vq
# python generate_faiss_multi_emb.py --config Musical_Instruments.yaml

# python preprocess.py --dataset Industrial_and_Scientific --his_len 50

# python encode_emb.py --dataset Industrial_and_Scientific --text_types title brand features categories description --gpu_id 1

# cd vq
# python generate_faiss_multi_emb.py --config Industrial_and_Scientific.yaml


    

python preprocess.py --dataset Baby_Products --his_len 50

python encode_emb.py --dataset Baby_Products --text_types title brand features categories description --gpu_id 1

cd vq
python generate_faiss_multi_emb.py --config Baby_Products.yaml


# python preprocess.py --dataset Video_Games --his_len 50

# python encode_emb.py --dataset Video_Games --text_types title brand features categories description --gpu_id 0

# cd vq
# python generate_faiss_multi_emb.py --config Video_Games.yaml