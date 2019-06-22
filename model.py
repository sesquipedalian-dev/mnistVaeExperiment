import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch 

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.fc1 = nn.Linear(20 * 7 * 7, 500)
        self.fc2 = nn.Linear(500, 250)
        
        self.fc31 = nn.Linear(250, 50)
        self.fc32 = nn.Linear(250, 50)
        
    def forward(self, x):
        # start 1 x 28 x 28
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x, pool_indices_1 = self.pool(x)
        # now 10 x 14 x 14
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x, pool_indices_2 = self.pool(x)
        # now 20 x 7 x 7)
        
        x = x.view(-1, 20 * 7 * 7)
        x = F.relu(self.fc1(x))
        # now 1 x 500
        
        x = F.relu(self.fc2(x))
        # now 1 x 250 
        
        # Split into means and std deviations
        mu = self.fc31(x)
        logvar = self.fc32(x)
        # now 2 * 1 x 50 
        
        return mu, logvar, pool_indices_1, pool_indices_2

class Decoder(nn.Module): 
    def __init__(self, device): 
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(50, 250)
        self.fc2 = nn.Linear(250, 500)
        self.fc3 = nn.Linear(500, 20 * 7 * 7)
        self.conv1 = nn.ConvTranspose2d(20, 20, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(20, 10, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(10, 10, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.ConvTranspose2d(10, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxUnpool2d(2, 2)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        
    @staticmethod
    def make_top_left_unpool_indices(shape, stride=(2, 2)):
        """
        Create an unpooling index that always puts the value in the top left of the 
        unpooling kernel.
        """
        out = np.zeros(shape) # TODO 
        for b in range(shape[0]): 
            for d in range(shape[1]): 
                for x in range(shape[2]): 
                    for y in range(shape[3]):
                        out[b, d, x, y] = x * shape[3] * 2 * stride[0] + y * stride[1]
        return torch.from_numpy(out).float()
    
    def forward(self, x): 
        # start 50
        pool_indices_1 = self.make_top_left_unpool_indices((x.shape[0], 10, 14, 14)).type(torch.LongTensor).to(self.device)
        pool_indices_2 = self.make_top_left_unpool_indices((x.shape[0], 20, 7, 7)).type(torch.LongTensor).to(self.device)
        
        x = self.fc1(x)
        # now 250
        
        x = F.relu(self.fc2(x))
        # now 500
        
        x = F.relu(self.fc3(x))
        # now 20 * 7 * 7
        
        x = x.view(-1, 20, 7, 7)
        # now 20 x 7 x 7
        
        x = self.pool(x, pool_indices_2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # now 10 x 14 x 14
        
        x = self.pool(x, pool_indices_1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # now 1 x 28 x 28
        
        # normalize to range -1 to 1 to match input data
        x = self.sigmoid(x)
        x = (x - .75) * 4
        
        return x
    
class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(device)
        self.device = device

    def reparametrize(self, mu, logvar, deviation):
        std = logvar.mul(0.5).exp_()
        
        if deviation is not None:
            if not isinstance(deviation, torch.Tensor):
                deviation = torch.FloatTensor([deviation]).repeat(50).to(self.device)
            eps = Variable(deviation)
            deviation.detach()
        else:
            if torch.cuda.is_available():
                deviation = torch.cuda.FloatTensor(std.size()).normal_()
            else:
                deviation = torch.FloatTensor(std.size()).normal_()
            eps = Variable(deviation)
            
        return eps.mul(std).add_(mu)
    
    def encode(self, x): 
        mu, logvar, _, _ = self.encoder(x)
        return mu, logvar
    
    def decode(self, x): 
        return self.decoder(x)
    
    def sample(self, mu, logvar, deviation=1.): 
        z = self.reparametrize(mu.to(self.device), logvar.to(self.device), deviation)
        z = z.to(self.device)
        decoded = self.decode(z)
        return decoded.cpu().detach()
    
    def forward(self, x):
        # encode
        mu, logvar, _, _ = self.encoder(x)
        
        # sample
        z = self.reparametrize(mu, logvar)
        
        result = self.decode(z)
        return result, mu, logvar
