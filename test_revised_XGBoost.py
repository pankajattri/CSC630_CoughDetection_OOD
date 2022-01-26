import torch
import torchaudio
import time
import pandas as pd
from models.disc import Disc
from models.FENet import FENet
from torch_datasets.AudioDataset import AudioDataset
from torch_datasets.mydataset import testdataset2
from sklearn.metrics import confusion_matrix, classification_report

import xgboost as xgb
import joblib
import tqdm as tq

import warnings

from pycm import *

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

dataloaders = {}
dl_config = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
batch_size = 64
                             
for mode in ['test']:
    
    dataset = testdataset2(raw_data_dir,window_length,sr,mode,
                            only_speech = False)
    
    dataloaders[mode] = torch.utils.data.DataLoader(
                               dataset = dataset,
                               batch_size = batch_size,
                               shuffle = True,
                               num_workers=0)

FENet_param_path = 'parameters/FENet/FENet.pkl'
net = FENet(FENet_param_path).to(device)

print("Test_loader size: ",len(dataloaders['test']))

if __name__ == '__main__':
    
    start = time.time()
    
    for i,batch in tq.tqdm(enumerate(dataloaders['test'])):
        
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
        full_data.to_csv('/content/features_test1.csv', mode='a',index='False')
        

    dtf = pd.read_csv('/content/features_test1.csv',index_col=None)
    dtf = dtf.dropna()
    dtf.to_csv('/content/features_test1.csv',index=False)
    dtf = pd.read_csv('/content/features_test1.csv',index_col=['Unnamed: 0'])
    print(dtf.shape)
    X_columns = dtf.columns[~dtf.columns.isin(['class'])]
    X_Val = dtf[X_columns]
    Y_Val = dtf['class']

    model = joblib.load('/content/XGB_classifier_model.pkl')
    preds = model.predict(X_Val)
    print(confusion_matrix(Y_Val,list(preds)))

    print(classification_report(Y_Val,list(preds)))

    cm = ConfusionMatrix(actual_vector=Y_Val, predict_vector=list(preds))
    print(cm)