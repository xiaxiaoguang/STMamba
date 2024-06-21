import argparse
from prepare import *
from train_STGmamba import *
from train_rnn import *


parser = argparse.ArgumentParser(description='Train & Test STG_Mamba for traffic/weather/flow forecasting')

# choose dataset
parser.add_argument('-dataset', type=str, default='know_air', help='which dataset to run [options: know_air, pems04, hz_metro]')
# choose model
parser.add_argument('-model', type=str, default='STGmamba', help='which model to train & test [options: STGmamba, lstm]')
# choose number of node features. For PEMS04 dataset, you should set mamba_features=307; For Know_Air dataset, mamba_features=184; For HZ_Metro, mamba_features=80
parser.add_argument('-mamba_features', type=int, default=307, help='number of features for the STGmamba model [options: 307,184,80]')

args = parser.parse_args()

###### loading data #######
    
if args.dataset =='know_air':
    print("\nLoading KnowAir Dataset...")
    speed_matrix = pd.read_csv('./Know_Air/knowair_temperature.csv',sep=',')
    A = np.load('./Know_Air/knowair_adj_mat.npy')

elif args.dataset == 'pems04':
    print("\nLoading PEMS04 data...")
    speed_matrix = pd.read_csv('./PEMS04/pems04_flow.csv',sep=',')
    A = np.load('./PEMS04/pems04_adj.npy')
    print(A)

elif args.dataset == 'hz_metro':
    print("\nLoading HZ-Metro data...")
    speed_matrix = pd.read_csv('./HZ_Metro_Dataset/hzmetro_flow.csv',sep=',')
    A = np.load('./HZ_Metro_Dataset/hzmetro_adj.npy')


print("\nPreparing train/test data...")
#train_dataloader, valid_dataloader, test_dataloader, max_value_speed = PrepareDataset(speed_matrix, BATCH_SIZE=64)
train_dataloader, valid_dataloader, test_dataloader, max_value = ini_PrepareDataset(speed_matrix, BATCH_SIZE=48)

# models you want to use
if args.model == 'STGmamba':
    # print("\nTraining STGmamba model for 1 epoch...")
    # STGmamba, STGmamba_loss = TrainSTG_Mamba(train_dataloader, valid_dataloader, A, K=3, num_epochs=1, mamba_features=args.mamba_features,test_dataloader=test_dataloader, max_value=max_value)
    # print("\nTesting STGmamba model for 1 epoch...")
    # results = TestSTG_Mamba(STGmamba, A, test_dataloader, max_value)

    print("\nTraining STGmamba model...")
    STGmamba, STGmamba_loss = TrainSTG_Mamba(train_dataloader, valid_dataloader, A, K=3, num_epochs=250, mamba_features=args.mamba_features,test_dataloader=test_dataloader, max_value=max_value)
    print("\nTesting STGmamba model...")
    results = TestSTG_Mamba(STGmamba, A, test_dataloader, max_value)


elif args.model == 'lstm':
    print("\nTraining lstm model...")
    lstm, lstm_loss = TrainLSTM(train_dataloader, valid_dataloader, num_epochs=200)
    print("\nTesting lstm model...")
    results = TestLSTM(lstm, test_dataloader, max_value)

