import torch
from torchvision import models

class Disc(torch.nn.Module):
    
    def __init__(self):
        
        # constructor of torch.nn.Module
        
        super(Disc, self).__init__()
        
        # initialize feature extractor
        
        self.model = models.resnet18(pretrained=True)
        self.model.conv1=torch.nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=self.model.conv1.kernel_size[0], 
                      stride=self.model.conv1.stride[0], padding=self.model.conv1.padding[0])
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 1)
        # freeze parameters of feature extractor
        '''
        for param in self.model.parameters():
            param.requires_grad = False
        '''    
        # initialize logistic regression
        '''
        self.fc1 = torch.nn.Linear(
            in_features = self.feature_extractor.stage7[0].out_channels,
            out_features = 1)
        '''
    def forward(self, x):
        x = self.model(x)
        
        return x

if __name__ == '__main__':
    
    net = Disc()
    # minimum input size is 128 x 128
    x = torch.randn(8,1,128,128)
    y = net(x)