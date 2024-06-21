export CUDA_VISIBLE_DEVICES=3
python main.py -dataset=pems04 -model=GCNmamba -mamba_features=307 -seq_len=144
