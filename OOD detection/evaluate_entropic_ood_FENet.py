
import numpy as np

from tqdm import tqdm

import torchaudio

import copy
import time
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW,SGD
import torch.utils.data as Data

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from models.disc_entropy_ood_FENet import Disc
from torch_datasets.mydataset import testdataset2
from utils.utils import encode_onehot
from torch.utils.data import DataLoader
import tools
import tqdm as tq
import os
import torchnet as tnt

from torchmetrics import AUROC
from torch_datasets.getresp import getresp

import torch.nn as nn
import torch.nn.functional as F
import torch


class IsoMaxPlusLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()"""
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature        
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        #distances = torch.abs(self.distance_scale) * F.pairwise_distance(
        #    F.normalize(features).unsqueeze(2), F.normalize(self.prototypes).t().unsqueeze(0), p=2.0)       
        distances = torch.abs(self.distance_scale) * torch.cdist(
            F.normalize(features), F.normalize(self.prototypes), p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -distances
        # The temperature may be calibrated after training to improve uncertainty estimation.
        return logits / self.temperature


class IsoMaxPlusLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, entropic_scale=10.0):
        super(IsoMaxPlusLossSecondPart, self).__init__()
        self.entropic_scale = entropic_scale

    def forward(self, logits, targets, debug=False):
        #############################################################################
        #############################################################################
        """Probabilities and logarithms are calculated separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
        #############################################################################
        #############################################################################
        #print("************")
        #print(type(logits))
        #print(logits)
        distances = -logits
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets.long()]
        loss = -torch.log(probabilities_at_targets).mean()
        if not debug:
            return loss
        else:
            targets_one_hot = torch.eye(distances.size(1))[targets].long().cuda()
            intra_inter_distances = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())
            inter_intra_distances = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)
            intra_distances = intra_inter_distances[intra_inter_distances != float('Inf')]
            inter_distances = inter_intra_distances[inter_intra_distances != float('Inf')]
            return loss, 1.0, intra_distances, 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  #to accelerate


#####################
###data processing###
#####################
torch.manual_seed(40)

#raw data's path
raw_data_dir = '../data/raw'

# what window length to use
window_length = 1.5 # seconds
print('\nTraining using window length of {} seconds...'.format(window_length))

# what sampling frequency to resample windows to
sr = 16000 # Hz

# initialize loss function 
loss_func = nn.NLLLoss().cuda()

# initialize log operator for Logarithmic Mel-scale Spectrogram
log = torchaudio.transforms.AmplitudeToDB().to(device)

# initialize Mel-scale Spectrogram operator for Logarithmic Mel-scale Spectrogram
mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sr,
                                                n_fft = 1024,
                                                n_mels = 128,
                                                hop_length = 64).to(device)

# initialize discriminator network
FENet_param_path = '../parameters/FENet/FENet.pkl'
model = Disc(FENet_param_path).to(device)

# initialize optimizer
optimizer = torch.optim.AdamW(params = model.parameters(),
                             lr = 0.01)

##################################################################
#criterion = nn.CrossEntropyLoss()
criterion = IsoMaxPlusLossSecondPart()
##################################################################


# number of epoch to train and validate for
num_epochs = 30

# where to save model parameters in a .pt file
pt_filename = '{}s_{}Hz_{}epochs_with_confidence.pt'.format(str(window_length).replace('.','-'),
                                            sr,num_epochs)
param_path = '../parameters/disc/' + pt_filename

# initialize training and validation dataloaders
dataloaders = {}

batch_size = 128
                             

train_dataset = testdataset2(raw_data_dir,window_length,sr,'train',
                            only_speech = False)
    
train_dataloader = DataLoader(
                               dataset = train_dataset,
                               batch_size = batch_size,
                               shuffle = True,
                               pin_memory=True,
                               num_workers=0)

val_dataset = testdataset2(raw_data_dir,window_length,sr,'val',
                            only_speech = False)
    
val_dataloader = DataLoader(
                               dataset = val_dataset,
                               batch_size = batch_size,
                               shuffle = True,
                               pin_memory=True,
                               num_workers=0)

test_dataset = testdataset2(raw_data_dir,window_length,sr,'test',
                            only_speech = False)
    
test_dataloader = DataLoader(
                               dataset = test_dataset,
                               batch_size = batch_size,
                               shuffle = True,
                               pin_memory=True,
                               num_workers=0)

# record the best validation loss across epochs
best_val_loss = 1e10

print("train_loader size",len(train_dataloader))

cudnn.benchmark = True
best_acc = 0


####################################
###begin trainning and validating###
####################################


def train(epoch):
    print('Epoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tq.tqdm(train_dataloader)):
        inputs = Variable(inputs, volatile=True).cuda()
        targets = targets.cuda().type_as(inputs)
        
        log_mel_spec = log(mel_spec(inputs))
        
        
        optimizer.zero_grad()
        outputs = model(log_mel_spec)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        tools.progress_bar(batch_idx, len(train_dataloader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def val(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tq.tqdm(val_dataloader)):
            inputs = Variable(inputs, volatile=True).cuda()
            targets = targets.cuda().type_as(inputs)
            
            log_mel_spec = log(mel_spec(inputs))
            
            
            outputs = model(log_mel_spec)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            tools.progress_bar(batch_idx, len(val_dataloader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving...')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/ckpt.pth')
        best_acc = acc

def test():
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tq.tqdm(test_dataloader)):
            inputs = Variable(inputs, volatile=True).cuda()
            targets = targets.cuda().type_as(inputs)
            
            log_mel_spec = log(mel_spec(inputs))
            
            
            outputs = model(log_mel_spec)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    acc = 100.*correct/total
    return acc
    
def detect(inloader, oodloader):
    auroc = AUROC(pos_label=1)
    auroctnt = tnt.meter.AUCMeter()
    model.eval()
    with torch.no_grad():
        print('Started evaluating test data set')
        
        for _, (inputs, targets) in enumerate(tq.tqdm(inloader)):
            
            inputs = Variable(inputs, volatile=True).cuda()
            targets = targets.cuda().type_as(inputs)
            
            log_mel_spec = log(mel_spec(inputs))
            targets.fill_(1)
            targets = targets.int()
            #print(type(targets))
            #print(targets)
            
            outputs = model(log_mel_spec)
            #print(outputs)
            #probabilities = torch.nn.Softmax(dim=1)(outputs)
            #score = probabilities.max(dim=1)[0] # this is the maximum probability score 
            #entropies = -(probabilities * torch.log(probabilities)).sum(dim=1)
            #score = -entropies # this is the negative entropy score
            # the negative entropy score is the best option for the IsoMax loss
            # outputs are equal to logits, which in turn are equivalent to negative distances
            score = outputs.max(dim=1)[0] # this is the minimum distance score
            #print(score)
            # the minimum distance score is the best option for the IsoMax+ loss
            auroc.update(score, targets) 
            auroctnt.add(score, targets)           
        
        print('Started evaluating OOD data set')
        for _, (inputs) in enumerate(tq.tqdm(oodloader)):
            inputs = Variable(inputs, volatile=True).cuda()
            #targets = torch.zeros_like(inputs)
            targets = torch.empty(inputs.shape[0])
            targets = targets.cuda().type_as(inputs)
            
            log_mel_spec = log(mel_spec(inputs))
            
            targets.fill_(0)
            targets = targets.int()
            #print(targets.shape)
            outputs = model(log_mel_spec)
            #print(outputs.shape)
            #probabilities = torch.nn.Softmax(dim=1)(outputs)
            #score = probabilities.max(dim=1)[0] # this is the maximum probability score 
            #entropies = -(probabilities * torch.log(probabilities)).sum(dim=1)
            #score = -entropies # this is the negative entropy score
            # the negative entropy score is the best option for the IsoMax loss
            # outputs are equal to logits, which in turn are equivalent to negative distances
            score = outputs.max(dim=1)[0] # this is the minimum distance score for detection
            #print(score.shape)
            # the minimum distance score is the best option for the IsoMax+ loss
            auroc.update(score, targets)            
            auroctnt.add(score, targets)            
    return auroc.compute(), auroctnt.value()[0]

'''
total_epochs= 1
for epoch in range(total_epochs):
    print()
    for param_group in optimizer.param_groups:
        print("LEARNING RATE: ", param_group["lr"])
    train(epoch)
    val(epoch)

'''
checkpoint = torch.load('checkpoint/ckpt.pth')
model.load_state_dict(checkpoint['model'])
#test_acc = checkpoint['acc']
test_acc = test()

print()
print("###################################################")
print("Test Accuracy (%): {0:.4f}".format(test_acc))
print("###################################################")
print()

#dataroot = os.path.expanduser(os.path.join('data', 'Imagenet_resize'))
#oodset = torchvision.datasets.ImageFolder(dataroot, transform=transform_test)
#oodloader = torch.utils.data.DataLoader(oodset, batch_size=64, shuffle=False, num_workers=4)


ood_dataset=getresp(sr)
oodloader = torch.utils.data.DataLoader(dataset=ood_dataset,
                                         batch_size=8,
                                         shuffle=False,
                                        
                                         num_workers=0)


auroc = detect(test_dataloader, oodloader)
print()
print("#################################################################################################################")
print("Detection performance for RESP as Out-of-Distribution [AUROC] (%): {0:.4f}".format(100. * auroc[0].item()), auroc[1])
print("#################################################################################################################")
print()