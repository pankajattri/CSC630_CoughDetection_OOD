import torch
import torchaudio
import time
import pandas as pd
from models.disc import Disc
from models.FENet import FENet
from torch_datasets.AudioDataset import AudioDataset
from torch_datasets.mydataset import testdataset2
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import joblib
import tqdm as tq

import warnings
# supresses torchaudio warnings. Should not be used in development
warnings.filterwarnings("ignore")

# for reproducibility
torch.manual_seed(42)

# to put tensors on GPU if available
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# where the raw data is
raw_data_dir = 'data/raw'

# what window length to use
window_length = 1.5 # seconds
print('\nTraining using window length of {} seconds...'.format(window_length))

# what sampling frequency to resample windows to
sr = 16000 # Hz

# initialize loss function (negative log-likelihood function for
# Bernoulli distribution)
loss_func = torch.nn.BCEWithLogitsLoss(reduction = 'mean')

# initialize log operator for Logarithmic Mel-scale Spectrogram
log = torchaudio.transforms.AmplitudeToDB().to(device)

# initialize Mel-scale Spectrogram operator for Logarithmic Mel-scale Spectrogram
mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sr,
                                                n_fft = 1024,
                                                n_mels = 128,
                                                hop_length = 64).to(device)

# initialize discriminator network
FENet_param_path = 'parameters/FENet/FENet.pkl'
net = FENet(FENet_param_path).to(device)


# where to save net parameters in a .pt file
pt_filename = '{}s_{}Hz.pt'.format(str(window_length).replace('.','-'),
                                            sr)
param_path = 'parameters/disc/' + pt_filename

# initialize training and validation dataloaders
dataloaders = {}
dl_config = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
batch_size = 64
                             
for mode in ['train','val']:
    
    dataset = testdataset2(raw_data_dir,window_length,sr,mode,
                            only_speech = False)
    
    dataloaders[mode] = torch.utils.data.DataLoader(
                               dataset = dataset,
                               batch_size = batch_size,
                               shuffle = True,
                               num_workers=0)

# record the best validation loss across epochs
best_val_loss = 1e10

print("Train_loader size: ",len(dataloaders['train']))
print("Val_loader size: ",len(dataloaders['val']))

if __name__ == '__main__':
    
    start = time.time()
    '''
    for i,batch in tq.tqdm(enumerate(dataloaders['train'])):
        
        x,labels = batch
        x = x.to(device)
        #labels = labels.to(device)
        log_mel_spec = log(mel_spec(x))
        features = net(log_mel_spec)

        data_df = pd.DataFrame(features.detach().cpu().numpy())
        #classes_df = pd.DataFrame(labels.detach().cpu().numpy(),columns=['class'])
        classes_df = pd.DataFrame(labels,columns=['class'])
        #print(data_df.shape, classes_df.shape)
        
        full_data = pd.concat([data_df,classes_df],axis = 1)
        full_data.to_csv('/content/features.csv', mode='a',index='False')
        
    '''
    dtf = pd.read_csv('/content/features_train.csv',index_col = ['Unnamed: 0'])
    X_columns = dtf.columns[~dtf.columns.isin(['class'])]
    X_Train = dtf[X_columns]
    Y_Train = dtf['class']

    model_RF = RandomForestClassifier(max_samples=None,max_features=32,n_estimators=500) 

    print('Started fitting RF model...')
    model_RF.fit(X_Train,Y_Train)
    print('Model fitting complete')

    print('Saving RF model')
    joblib.dump(model_RF,'/content/model_RF.pkl')

 