import torch.utils.data as utils
import numpy as np
import torch

def ini_PrepareDataset(speed_matrix, BATCH_SIZE=48, seq_len=12, pred_len=12, train_propotion=0.7, valid_propotion=0.1):
    """ Prepare Train & Test datasets and dataloaders

    Convert traffic/weather/volume matrix to train and test dataset.

    Args:
        speed_matrix: The whole spatial-temporal dataset matrix. (It doesn't necessarily means speed, but can also be flow or weather matrix). 
        seq_len: The length of input sequence.
        pred_len: The length of prediction sequence, match the seq_len for model compatibility.
    Return:
        Train_dataloader
        Test_dataloader
    """
    time_len = speed_matrix.shape[0]
    #max_speed = speed_matrix.max().max()
    #speed_matrix = speed_matrix / max_speed

    # MinMax Normalization Method.
    max_speed = speed_matrix.max().max()
    min_speed = speed_matrix.min().min()
    speed_matrix = (speed_matrix - min_speed)/(max_speed - min_speed)    

    speed_sequences, speed_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        speed_sequences.append(speed_matrix.iloc[i:i + seq_len].values)
        speed_labels.append(speed_matrix.iloc[i + seq_len:i + seq_len + pred_len].values)
    speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)

    # Reshape labels to have the same second dimension as the sequences
    speed_labels = speed_labels.reshape(speed_labels.shape[0], seq_len, -1)

    # shuffle & split the dataset to training and testing sets
    sample_size = speed_sequences.shape[0]
    index = np.arange(sample_size, dtype=int)
    np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
    valid_data, valid_label = speed_sequences[train_index:valid_index], speed_labels[train_index:valid_index]
    test_data, test_label = speed_sequences[valid_index:], speed_labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader, max_speed



def PrepareDataset(speed_matrix, BATCH_SIZE=48, seq_len=12, pred_len=12, train_propotion=0.7, valid_propotion=0.1):
    """ Prepare Train & Test datasets and dataloaders

    Convert traffic/weather/volume matrix to train and test dataset.

    Args:
        speed_matrix: The whole spatial-temporal dataset matrix. (It doesn't necessarily means speed, but can also be flow or weather matrix). 
        seq_len: The length of input sequence.
        pred_len: The length of prediction sequence, match the seq_len for model compatibility.
    Return:
        Train_dataloader
        Test_dataloader
    """
    time_len = speed_matrix.shape[0]
    
    # MinMax Normalization Method.
    max_speed = speed_matrix.max().max()
    min_speed = speed_matrix.min().min()
    speed_matrix = (speed_matrix - min_speed) / (max_speed - min_speed)

    train_speed_sequences, train_speed_labels = [], []
    valid_speed_sequences, valid_speed_labels = [], []
    test_speed_sequences, test_speed_labels = [], []

    # Prepare training dataset with new sequence and prediction lengths
    for i in range(int(np.floor(time_len * train_propotion)) - 2 * seq_len - 2 * pred_len + 1):
            train_speed_sequences.append(speed_matrix.iloc[i:i + seq_len * 2].values)
            train_speed_labels.append(speed_matrix.iloc[i + 1:i + 1 + 2 * pred_len].values)
    train_speed_sequences, train_speed_labels = np.asarray(train_speed_sequences), np.asarray(train_speed_labels)

    # Prepare validation and test datasets with original sequence and prediction lengths
    for i in range(time_len - seq_len - pred_len):
        if i >= int(np.floor(time_len * train_propotion)):
            if i < int(np.floor(time_len * (train_propotion + valid_propotion))):
                valid_speed_sequences.append(speed_matrix.iloc[i:i + seq_len].values)
                valid_speed_labels.append(speed_matrix.iloc[i + seq_len:i + seq_len + pred_len].values)
            else:
                test_speed_sequences.append(speed_matrix.iloc[i:i + seq_len].values)
                test_speed_labels.append(speed_matrix.iloc[i + seq_len:i + seq_len + pred_len].values)
    
    valid_speed_sequences, valid_speed_labels = np.asarray(valid_speed_sequences), np.asarray(valid_speed_labels)
    test_speed_sequences, test_speed_labels = np.asarray(test_speed_sequences), np.asarray(test_speed_labels)

    # Reshape labels to have the same second dimension as the sequences for training
    train_speed_labels = train_speed_labels.reshape(train_speed_labels.shape[0], seq_len * 2, -1)

    # Shuffle the training dataset
    train_index = np.arange(train_speed_sequences.shape[0])
    np.random.shuffle(train_index)

    train_speed_sequences = train_speed_sequences[train_index]
    train_speed_labels = train_speed_labels[train_index]

    train_data, train_label = torch.Tensor(train_speed_sequences), torch.Tensor(train_speed_labels)
    valid_data, valid_label = torch.Tensor(valid_speed_sequences), torch.Tensor(valid_speed_labels)
    test_data, test_label = torch.Tensor(test_speed_sequences), torch.Tensor(test_speed_labels)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader, max_speed
