from this import d
import time
import numpy as np
import pickle
import math
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from STGMamba import *
from torch.autograd import Variable
import torch.nn.functional as F
from STGEmbMamba import STGEmbMamba

def TrainSTG_Mamba(train_dataloader, valid_dataloader, test_dataloader, 
                   num_epochs=1, model_args=None,
                   max_speed = 1):
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    patch_size = 12
    in_channel = 1
    embed_dim = 96
    norm_layer = None
    num_feats = 1
    num_len = model_args.seq_len
    pred_len = 12
    fea_size = 96
    d_model = 96
    d_state = 64
    d_conv = 4
    expand = 2
    n_layers = 4
    STmamba = STGEmbMamba(patch_size, in_channel, num_len , pred_len,
                             embed_dim,norm_layer, num_feats, fea_size, 
                             d_model, d_state, d_conv, expand, n_layers).cuda()
    STmamba.cuda()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    learning_rate = 1e-4
    optimizer = optim.AdamW(STmamba.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    use_gpu = torch.cuda.is_available()

    interval = 50
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    losses_epoch = []

    cur_time = time.time()
    pre_time = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        trained_number = 0

        valid_dataloader_iter = iter(valid_dataloader)

        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            STmamba.zero_grad()

            labels = torch.squeeze(labels)
            pred = STmamba(inputs)

            loss_train = loss_MSE(pred, labels)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            scheduler.step()

            losses_train.append(loss_train.cpu().detach().numpy())

            try:
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)

            if use_gpu:
                inputs_val, labels_val = inputs_val.cuda(), labels_val.cuda()

            labels_val = torch.squeeze(labels_val)

            pred = STmamba(inputs_val)
            loss_valid = loss_MSE(pred, labels_val)
            losses_valid.append(loss_valid.cpu().detach().numpy())

            trained_number += 1

            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(np.mean(losses_train[-interval:]), decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(np.mean(losses_valid[-interval:]), decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format(
                    trained_number * batch_size,
                    loss_interval_train,
                    loss_interval_valid,
                    np.around([cur_time - pre_time], decimals=8)))
                with open("record.txt",'a') as f:
                    print('EPOCH {} Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format(
                    epoch,
                    trained_number * batch_size,
                    loss_interval_train,
                    loss_interval_valid,
                    np.around([cur_time - pre_time], decimals=8)),file=f)
                pre_time = cur_time

        loss_epoch = np.mean(losses_valid[-interval:])
        losses_epoch.append(loss_epoch)
        TestSTG_Mamba(STmamba,test_dataloader,max_speed,epoch)


    return STmamba, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]




def TestSTG_Mamba(STmamba, test_dataloader, max_speed,epoch):
    inputs, labels = next(iter(test_dataloader))

    [batch_size, step_size, fea_size] = inputs.size()

    pre_time = time.time()

    use_gpu = torch.cuda.is_available()

    losses_mse = []
    losses_l1 = []
    MAEs = []
    MAPEs = []
    MSEs = []
    RMSEs = []
    VARs = []

    for data in test_dataloader:
        inputs, labels = data

        if inputs.shape[0] != batch_size:
            continue

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        pred = STmamba(inputs)
        labels = torch.squeeze(labels)

        loss_mse = F.mse_loss(pred, labels)
        loss_l1 = F.l1_loss(pred, labels)
        MAE = torch.mean(torch.abs(pred - torch.squeeze(labels)))
        MAPE = torch.mean(torch.abs(pred - torch.squeeze(labels)) / torch.abs(torch.squeeze(labels)))
        non_zero_labels = torch.abs(labels) > 0
        if torch.any(non_zero_labels):
            MAPE_values = torch.abs(pred - torch.squeeze(labels)) / torch.abs(torch.squeeze(labels))
            MAPE = torch.mean(MAPE_values[non_zero_labels])
            MAPEs.append(MAPE.item())

        MSE = torch.mean((torch.squeeze(labels) - pred)**2)
        RMSE = math.sqrt(torch.mean((torch.squeeze(labels) - pred)**2))
        VAR = 1-(torch.var(torch.squeeze(labels)-pred))/torch.var(torch.squeeze(labels))

        losses_mse.append(loss_mse.item())
        losses_l1.append(loss_l1.item())
        MAEs.append(MAE.item())
        MAPEs.append(MAPE.item())
        MSEs.append(MSE.item())
        RMSEs.append(RMSE)
        VARs.append(VAR.item())

        # tested_batch += 1

        # if tested_batch % 100 == 0:
        #     cur_time = time.time()
        #     breakpoint()
        #     print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format(
        #         tested_batch * batch_size,
        #         np.around([loss_l1.data[0]], decimals=8),
        #         np.around([loss_mse.data[0]], decimals=8),
        #         np.around([cur_time - pre_time], decimals=8)))
        #     pre_time = cur_time

    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    MAEs = np.array(MAEs)
    MAPEs = np.array(MAPEs)
    MSEs = np.array(MSEs)
    RMSEs = np.array(RMSEs)
    VARs = np.array(VARs)

    # mean_l1 = np.mean(losses_l1) * max_speed
    # std_l1 = np.std(losses_l1) * max_speed
    # mean_mse = np.mean(losses_mse) * max_speed
    MAE_ = np.mean(MAEs) * max_speed
    std_MAE_ = np.std(MAEs) * max_speed 
    
    #std_MAE measures the consistency & stability of the model's 
    # performance across different test sets or iterations. 
    # Usually if (std_MAE/MSE)<=10%., means the trained model is good.
    cur_time = time.time()

    MAPE_ = np.mean(MAPEs) * 100
    MSE_ = np.mean(MSEs) * (max_speed ** 2)
    RMSE_ = np.mean(RMSEs) * max_speed
    VAR_ = np.mean(VARs)
    results = [MAE_, std_MAE_, MAPE_, MSE_, RMSE_, VAR_]

    print('Tested: MAE: {}, std_MAE: {}, MAPE: {}, MSE: {}, RMSE: {}, VAR: {} ,TIME: {}'.format(MAE_, std_MAE_, MAPE_, MSE_, RMSE_, VAR_,cur_time-pre_time))
    with open("record.txt",'a') as f:
        print('Tested: MAE: {}, std_MAE: {}, MAPE: {}, MSE: {}, RMSE: {}, VAR: {}'.format(MAE_, std_MAE_, MAPE_, MSE_, RMSE_, VAR_),file=f)

    with open(f'checkpoints/mamba_{epoch%10}.pkl', 'wb') as file:
        pickle.dump(STmamba, file)
    return results
