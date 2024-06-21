from this import d
import time
import numpy as np
import math
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from STGMamba import *
from torch.autograd import Variable
from mamba_ssm import Mamba2, Mamba
import torch.nn.functional as F
from seq_mamba import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from embedding import OldPatchEmbedding
from STGEmbMamba import STGEmbMamba

import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def TrainSTG_Mamba(train_dataloader, valid_dataloader, A, K=3, num_epochs=1, mamba_features=307, test_dataloader=None, max_value=None):
    inputs, labels = next(iter(train_dataloader))
    # inputs = F.pad(inputs, (0, 5), "constant", 0)
    # labels = F.pad(labels, (0, 5), "constant", 0)
    [batch_size, step_size, fea_size] = inputs.size()

    ################################## EmbManba ############################
    # patch_size = 1
    # in_channel = 1
    # embed_dim = 96
    # norm_layer = None
    # num_feats = 1
    # fea_size = 96
    # d_model = 96
    # d_state = 64
    # d_conv = 4
    # expand = 2
    # n_layers = 1
    # kfgn_mamba = STGEmbMamba(patch_size, in_channel, embed_dim, norm_layer, num_feats, fea_size, d_model, d_state, d_conv, expand, n_layers).cuda()
    ################################### EmbManba ############################

    ################################### one layer mamba #####################
    # kfgn_mamba = Mamba(
    #         # This module uses roughly 3 * expand * d_model^2 parameters
    #         d_model=fea_size, # Model dimension d_model
    #         d_state=64,  # SSM state expansion factor, typically 64 or 128
    #         d_conv=4,    # Local convolution width
    #         expand=2,    # Block expansion factor
    #         # headdim=39,
    #     )
    ################################### one layer mamba #####################

    ################################# seq mamba ##############################

    config = MambaConfig(
        d_model=1024,
        n_layer=4,
        d_intermediate=128,
        vocab_size=1000,  # This will be adjusted inside the model
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        pad_vocab_size_multiple=1,
        feature_size=307,  # New field for the feature size
        tie_embeddings=False,  # Not used, but should be set
        adjacency_matrix=torch.tensor(A)
    )

        # Create the model
    kfgn_mamba = MambaLMHeadModel(config)




    ################################# seq mamba ##############################


    kfgn_mamba.cuda()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    learning_rate = 1e-4
    optimizer = optim.AdamW(kfgn_mamba.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    use_gpu = torch.cuda.is_available()

    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    losses_epoch = []

    cur_time = time.time()
    pre_time = time.time()

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        trained_number = 0

        valid_dataloader_iter = iter(valid_dataloader)

        for data in train_dataloader:
            inputs, labels = data
            
            # inputs = F.pad(inputs, (0, 5), "constant", 0)
            # labels = F.pad(labels, (0, 5), "constant", 0)

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            kfgn_mamba.zero_grad()

            labels = torch.squeeze(labels)
            pred = kfgn_mamba(inputs,12)

            loss_train = loss_MSE(pred, labels)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            scheduler.step()

            losses_train.append(loss_train.cpu().detach().numpy())

            try:
                inputs_val, labels_val = next(valid_dataloader_iter)
                # inputs_val = F.pad(inputs_val, (0, 5), "constant", 0)
                # labels_val = F.pad(labels_val, (0, 5), "constant", 0)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
                # inputs_val = F.pad(inputs_val, (0, 5), "constant", 0)
                # labels_val = F.pad(labels_val, (0, 5), "constant", 0)

            if use_gpu:
                inputs_val, labels_val = inputs_val.cuda(), labels_val.cuda()

            labels_val = torch.squeeze(labels_val)

            pred = kfgn_mamba(inputs_val,12)
            loss_valid = loss_MSE(pred, labels_val)
            losses_valid.append(loss_valid.cpu().detach().numpy())

            trained_number += 1

            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(np.mean(losses_train[-interval:]), decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(np.mean(losses_valid[-interval:]), decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}\n'.format(
                    trained_number * batch_size,
                    loss_interval_train,
                    loss_interval_valid,
                    np.around([cur_time - pre_time], decimals=8)))
                print("Test results:\n")
                pre_time = cur_time

        loss_epoch = np.mean(losses_valid[-interval:])
        losses_epoch.append(loss_epoch)
    

    return kfgn_mamba, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]




def TestSTG_Mamba(kfgn_mamba, A, test_dataloader, max_speed):
    inputs, labels = next(iter(test_dataloader))
    # inputs = F.pad(inputs, (0, 5), "constant", 0)
    # labels = F.pad(labels, (0, 5), "constant", 0)

    [batch_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()

    use_gpu = torch.cuda.is_available()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    tested_batch = 0

    losses_mse = []
    losses_l1 = []
    MAEs = []
    MAPEs = []
    MSEs = []
    RMSEs = []
    VARs = []

    #predictions = []
    #ground_truths = []

    for data in test_dataloader:
        inputs, labels = data
        # inputs = F.pad(inputs, (0, 5), "constant", 0)
        # labels = F.pad(labels, (0, 5), "constant", 0)

        if inputs.shape[0] != batch_size:
            continue

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        pred = kfgn_mamba(inputs,12)
        labels = torch.squeeze(labels)

        loss_mse = F.mse_loss(pred, labels)
        loss_l1 = F.l1_loss(pred, labels)
        MAE = torch.mean(torch.abs(pred - torch.squeeze(labels)))
        MAPE = torch.mean(torch.abs(pred - torch.squeeze(labels)) / torch.abs(torch.squeeze(labels)))
        # Calculate MAPE only for non-zero labels
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

        #predictions.append(pd.DataFrame(pred.cpu().data.numpy()))
        #ground_truths.append(pd.DataFrame(labels.cpu().data.numpy()))

        tested_batch += 1

        if tested_batch % 100 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format(
                tested_batch * batch_size,
                np.around([loss_l1.data[0]], decimals=8),
                np.around([loss_mse.data[0]], decimals=8),
                np.around([cur_time - pre_time], decimals=8)))
            pre_time = cur_time

    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    MAEs = np.array(MAEs)
    MAPEs = np.array(MAPEs)
    MSEs = np.array(MSEs)
    RMSEs = np.array(RMSEs)
    VARs = np.array(VARs)

    mean_l1 = np.mean(losses_l1) * max_speed
    std_l1 = np.std(losses_l1) * max_speed
    mean_mse = np.mean(losses_mse) * max_speed
    MAE_ = np.mean(MAEs) * max_speed
    std_MAE_ = np.std(MAEs) * max_speed #std_MAE measures the consistency & stability of the model's performance across different test sets or iterations. Usually if (std_MAE/MSE)<=10%., means the trained model is good.
    MAPE_ = np.mean(MAPEs) * 100
    MSE_ = np.mean(MSEs) * (max_speed ** 2)
    RMSE_ = np.mean(RMSEs) * max_speed
    VAR_ = np.mean(VARs)
    results = [MAE_, std_MAE_, MAPE_, MSE_, RMSE_, VAR_]

    print('Tested: MAE: {}, std_MAE: {}, MAPE: {}, MSE: {}, RMSE: {}, VAR: {}'.format(MAE_, std_MAE_, MAPE_, MSE_, RMSE_, VAR_))
    return results
