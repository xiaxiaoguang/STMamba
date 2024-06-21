import argparse
from prepare import *
from train_rnn import *
from train_STGmamba2 import *
from train_STGmamba3 import *
from train_STGmamba4 import *


parser = argparse.ArgumentParser(description='Train & Test STG_Mamba for traffic/weather/flow forecasting')

# choose dataset
parser.add_argument('-dataset', type=str, default='know_air', help='which dataset to run [options: know_air, pems04, hz_metro]')
# choose model
parser.add_argument('-model', type=str, default='STGmamba', help='which model to train & test [options: STGmamba, lstm]')
# choose number of node features. For PEMS04 dataset, you should set mamba_features=307; For Know_Air dataset, mamba_features=184; For HZ_Metro, mamba_features=80
parser.add_argument('-mamba_features', type=int, default=307, help='number of features for the STGmamba model [options: 307,184,80]')

parser.add_argument('-seq_len',type=int,default=12,help="time series sequence length")
args = parser.parse_args()

###### loading data #######
    
if args.dataset =='know_air':
    print("\nLoading KnowAir Dataset...")
    speed_matrix = pd.read_csv('Know_Air/knowair_temperature.csv',sep=',')
    A = np.load('Know_Air/knowair_adj_mat.npy')
    

elif args.dataset == 'pems04':
    print("\nLoading PEMS04 data...")
    speed_matrix = pd.read_csv('PEMS04/pems04_flow.csv',sep=',')
    A = np.load('PEMS04/pems04_adj.npy')

elif args.dataset == 'hz_metro':
    print("\nLoading HZ-Metro data...")
    speed_matrix = pd.read_csv('HZ_Metro/hzmetro_flow.csv',sep=',')
    A = np.load('HZ_Metro/hzmetro_adj.npy')


print("\nPreparing train/test data...")
#train_dataloader, valid_dataloader, test_dataloader, max_value_speed = PrepareDataset(speed_matrix, BATCH_SIZE=64)
train_dataloader, valid_dataloader, test_dataloader, max_value = PrepareDataset(speed_matrix, seq_len=args.seq_len , BATCH_SIZE=32)

# models you want to use
if args.model == 'STGmamba':
    print("\nTraining STGmamba model...")
    STGmamba, STGmamba_loss = TrainSTG_Mamba(train_dataloader, valid_dataloader,test_dataloader,
                                              num_epochs=200, 
                                             model_args=args,max_speed=max_value)

# models you want to use
if args.model == 'GCNmamba':
    print("\nTraining GCNmamba model...")
    STGmamba, STGmamba_loss = TrainGCNMamba(train_dataloader, valid_dataloader,test_dataloader,
                                              num_epochs=200, 
                                             model_args=args,max_speed=max_value)

# models you want to use
if args.model == 'imamba':
    print("\nTraining imamba model...")
    STGmamba, STGmamba_loss = TrainiMamba(train_dataloader, valid_dataloader,test_dataloader,
                                              num_epochs=200, model_args=args,max_speed=max_value)

