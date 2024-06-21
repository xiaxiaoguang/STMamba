export CUDA_VISIBLE_DEVICES=2
python main.py -dataset=pems04 -model=STGmamba -mamba_features=307 -seq_len=144
