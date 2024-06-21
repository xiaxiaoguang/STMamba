# Comprehend Mamba in Spatial-Temporal Graph Forecasting task

Detrailed Results of our training is in checkpoints folder, the iMamba 3/6 is the final results of iMamba, because each of them need more epoch(600 comparing to 200) to converge,but because of the high efficiency, its total traning time is similar with Graphmamba/EmbedMamba.

## Requirements
- PyTorch==1.11.0
- Python==3.8.10
- numpy==1.22.4
- pandas==2.0.3
- einops==0.7.0
- mamba-ssm
- causal-conv1d
- triton
- argparse
- dataclasses
- typing
- time
- math


## Model Training/Testing

```bash
# STGMamba
bash run.sh
```

```bash
# GCNMamba
bash run2.sh
```

```bash
# iMamba
bash run3.sh
```




## Citation

If you find this repository useful in your own research, please cite our work.

