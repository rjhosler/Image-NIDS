import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd
import time

import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils import data
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh, Dropout2d
from torchvision import datasets, transforms, utils
import bottleneck as bn

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

import os
import numpy as np
from itertools import chain
import sklearn

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cudnn.benchmark = True

def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.main = nn.Sequential(
            Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), Dropout2d(DROP), ReLU(inplace=True),
            Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), Dropout2d(DROP), ReLU(inplace=True),
            Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), Dropout2d(DROP), ReLU(inplace=True),
            Conv2d(DIM * 4, DIM * 8, 4, 2, 0, bias=False), BatchNorm2d(DIM * 8), Dropout2d(DROP), ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(DIM * 8, NLAT)
        )
        
    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x = self.fc(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(NLAT, DIM * 8), ReLU(),
        )
        
        self.main = nn.Sequential(
            ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 0, bias=False), BatchNorm2d(DIM * 4), Dropout2d(DROP), ReLU(inplace=True),
            ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), Dropout2d(DROP), ReLU(inplace=True),
            ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), Dropout2d(DROP), ReLU(inplace=True),
            ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, DIM * 8, 1, 1)
        x = self.main(x)
        return x
        
def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    
    dataset = datasets.ImageFolder('BOT-IOT/Images/Train', transform=transform)
    
    train_loader = data.DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=16, drop_last=True)

    encoder, decoder = Encoder(), Decoder()
    criterion = nn.MSELoss()

    encoder.train()
    decoder.train()

    if torch.cuda.is_available():
        encoder, decoder = encoder.cuda(), decoder.cuda()

    # Optimizers
    enc_optim = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    dec_optim = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    curr_iter = 0       
    
    for batch_idx, (images, _) in enumerate(train_loader, 1):
        if torch.cuda.is_available():
            images = images.cuda()
                
        if curr_iter == 0:
            init_x = images
            curr_iter += 1
        break
    
    start_time = time.time()
    for epoch in range(ITER):
        if (epoch+1) % 10 == 0 or epoch == 0:
            print("======== EPOCH %d ========" % (epoch+1))
        for batch_idx, (images, _) in enumerate(train_loader, 1):

            if torch.cuda.is_available():
                images = images.cuda()

            enc_optim.zero_grad()
            dec_optim.zero_grad()

            # ======== Train Generator ======== #

            batch_size = images.size()[0]

            z = encoder(images)
            x_recon = decoder(z)
            recon_loss = criterion(x_recon, images)

            # ======== MMD Kernel Loss ======== #

            z_fake = Variable(torch.randn(images.size()[0], NLAT) * SIGMA)
            if torch.cuda.is_available():
                z_fake = z_fake.cuda()

            z_real = encoder(images)

            mmd_loss = imq_kernel(z_real, z_fake, h_dim=NLAT)
            mmd_loss = mmd_loss / BATCH_SIZE

            total_loss = recon_loss + mmd_loss
            total_loss.backward()

            enc_optim.step()
            dec_optim.step()

            curr_iter += 1
            
            if curr_iter % 1000 == 0 or curr_iter == 2:
                print("Iter: %d, Reconstruction Loss: %.4f, MMD Loss %.4f, Seconds %.4f" %
                      (curr_iter, recon_loss.data.item(), mmd_loss.item(), time.time() - start_time))
            
            if (epoch+1) % (ITER // 5) == 0:
                print("======== SAVING MODELS ======== ")
                if not os.path.exists('Models/wae/'):
                    os.makedirs('Models/wae/')
                torch.save(encoder.state_dict(), 'Models/wae/encoder_bot%d.ckpt' % (epoch+1))
                torch.save(decoder.state_dict(), 'Models/wae/decoder_bot%d.ckpt' % (epoch+1))

# training hyperparameters
BATCH_SIZE = 1024
ITER = 50
IMAGE_SIZE = 32
NUM_CHANNELS = 3
DIM = 16
NLAT = 100

LEARNING_RATE = 1e-4
SIGMA = 1
DROP = 0.1

if __name__ == "__main__":
    main()