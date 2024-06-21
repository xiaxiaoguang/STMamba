import torch.utils.data as utils
import numpy as np
import pickle
import os
from forecasting_dataset import ForecastingDataset

def PrepareDataset(speed_matrix, BATCH_SIZE=48, seq_len=12, 
                   pred_len=12, train_propotion=0.7, valid_propotion=0.1):
    """ Prepare Train & Test datasets and dataloaders

    Convert traffic/weather/volume matrix to train and test dataset.

    Args:
        speed_matrix: The whole spatial-temporal dataset matrix. 
        (It doesn't necessarily means speed, but can also be flow or weather matrix). 
        seq_len: The length of input sequence.
        pred_len: The length of prediction sequence, match the seq_len for model compatibility.
    Return:
        Train_dataloader
        Test_dataloader
    """
    f1 = f"dataset/data_{seq_len}_{pred_len}.npy"
    f2 = f"dataset/index_{seq_len}_{pred_len}.npy"
    max_speed = speed_matrix.max().max()
    if os.path.isfile(f1):
        data = np.load(f1)
        with open(f2,'rb') as f:
            index = pickle.load(f)
    else :
        time_len = speed_matrix.shape[0]
        min_speed = speed_matrix.min().min()
        speed_matrix = (speed_matrix - min_speed)/(max_speed - min_speed)   
        speed_labels = []
        for i in range(time_len - seq_len - pred_len):
            speed_labels.append((i,i + seq_len,i + seq_len + pred_len))
        speed_labels = np.asarray(speed_labels)
        num_data = speed_labels.shape[0]
        index = {}
        a = int(num_data*train_propotion)
        b = int(num_data*(train_propotion + valid_propotion))
        index['train'] = speed_labels[:a]
        index['valid'] = speed_labels[a:b]
        index['test']  = speed_labels[-b:]
        data=np.array(speed_matrix)
        np.save(f1,speed_matrix)
        with open(f2,'wb') as f:
            pickle.dump(index,f)


    train_dataset = ForecastingDataset(data, index,mode='train')
    valid_dataset = ForecastingDataset(data, index,mode='valid')
    test_dataset  = ForecastingDataset(data, index,mode='test')

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader, max_speed
