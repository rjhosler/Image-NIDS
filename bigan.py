import torch
import torch.nn as nn
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
from torch.optim.lr_scheduler import StepLR
import bottleneck as bn
import os

from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import os

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def log_odds(p):
    p = torch.clamp(p.mean(dim=0), 1e-7, 1-1e-7)
    return torch.log(p / (1 - p))


class MaxOut(nn.Module):
    def __init__(self, k=2):
        """ MaxOut nonlinearity.

        Args:
          k: Number of linear pieces in the MaxOut opeartion. Default: 2
        """
        super().__init__()

        self.k = k

    def forward(self, input):
        output_dim = input.size(1) // self.k
        input = input.view(input.size(0), output_dim, self.k, input.size(2), input.size(3))
        output, _ = input.max(dim=2)
        return output


class DeterministicConditional(nn.Module):
    def __init__(self, mapping, shift=None):
        """ A deterministic conditional mapping. Used as an encoder or a generator.
        Args:
          mapping: An nn.Sequential module that maps the input to the output deterministically.
          shift: A pixel-wise shift added to the output of mapping. Default: None
        """
        super().__init__()

        self.mapping = mapping
        self.shift = shift

    def set_shift(self, value):
        if self.shift is None:
            return
        assert list(self.shift.data.size()) == list(value.size())
        self.shift.data = value

    def forward(self, input):
        output = self.mapping(input)
        if self.shift is not None:
            output = output + self.shift
        return output


class GaussianConditional(nn.Module):
    def __init__(self, mapping, shift=None):
        """ A Gaussian conditional mapping. Used as an encoder or a generator.
        Args:
          mapping: An nn.Sequential module that maps the input to the parameters of the Gaussian.
          shift: A pixel-wise shift added to the output of mapping. Default: None
        """
        super().__init__()

        self.mapping = mapping
        self.shift = shift

    def set_shift(self, value):
        if self.shift is None:
            return
        assert list(self.shift.data.size()) == list(value.size())
        self.shift.data = value

    def forward(self, input):
        params = self.mapping(input)
        nlatent = params.size(1) // 2
        mu, log_sigma = params[:, :nlatent], params[:, nlatent:]
        sigma = log_sigma.exp()
        eps = torch.randn(mu.size()).to(input.device)
        output = mu + sigma * eps
        if self.shift is not None:
            output = output + self.shift
        return output


class JointCritic(nn.Module):
    def __init__(self, x_mapping, z_mapping, joint_mapping):
        """ A joint Wasserstein critic function.
        Args:
          x_mapping: An nn.Sequential module that processes x.
          z_mapping: An nn.Sequential module that processes z.
          joint_mapping: An nn.Sequential module that process the output of x_mapping and z_mapping.
        """
        super().__init__()

        self.x_net = x_mapping
        self.z_net = z_mapping
        self.joint_net = joint_mapping

    def forward(self, x, z):
        assert x.size(0) == z.size(0)
        x_out = self.x_net(x)
        z_out = self.z_net(z)
        joint_input = torch.cat((x_out, z_out), dim=1)
        output = self.joint_net(joint_input)
        return output


class WALI(nn.Module):
    def __init__(self, E, G, C):
        """ Adversarially learned inference (a.k.a. bi-directional GAN) with Wasserstein critic.
        Args:
          E: Encoder p(z|x).
          G: Generator p(x|z).
          C: Wasserstein critic function f(x, z).
        """
        super().__init__()

        self.E = E
        self.G = G
        self.C = C

    def get_encoder_parameters(self):
        return self.E.parameters()

    def get_generator_parameters(self):
        return self.G.parameters()

    def get_critic_parameters(self):
        return self.C.parameters()

    def encode(self, x):
        return self.E(x)

    def generate(self, z):
        return self.G(z)

    def reconstruct(self, x):
        return self.generate(self.encode(x))

    def criticize(self, x, z_hat, x_tilde, z):
        input_x = torch.cat((x, x_tilde), dim=0)
        input_z = torch.cat((z_hat, z), dim=0)
        output = self.C(input_x, input_z)
        data_preds, sample_preds = output[:x.size(0)], output[x.size(0):]
        return data_preds, sample_preds

    def calculate_grad_penalty(self, x, z_hat, x_tilde, z):
        bsize = x.size(0)
        eps = torch.rand(bsize, 1, 1, 1).to(x.device) # eps ~ Unif[0, 1]
        intp_x = eps * x + (1 - eps) * x_tilde
        intp_z = eps * z_hat + (1 - eps) * z
        intp_x.requires_grad = True
        intp_z.requires_grad = True
        C_intp_loss = self.C(intp_x, intp_z).sum()
        grads = autograd.grad(C_intp_loss, (intp_x, intp_z), retain_graph=True, create_graph=True)
        grads_x, grads_z = grads[0].view(bsize, -1), grads[1].view(bsize, -1)
        grads = torch.cat((grads_x, grads_z), dim=1)
        grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def forward(self, x, z, lamb=10):
        z_hat, x_tilde = self.encode(x), self.generate(z)
        data_preds, sample_preds = self.criticize(x, z_hat, x_tilde, z)
        EG_loss = torch.mean(data_preds - sample_preds)
        C_loss = -EG_loss + lamb * self.calculate_grad_penalty(x.data, z_hat.data, x_tilde.data, z.data)
        return C_loss, EG_loss
      
def create_encoder():
    mapping = nn.Sequential(
        Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        Conv2d(DIM * 4, DIM * 4, 4, 1, 0, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        Conv2d(DIM * 4, NLAT, 1, 1, 0))
    return DeterministicConditional(mapping)

def create_generator():
    mapping = nn.Sequential(
        ConvTranspose2d(NLAT, DIM * 4, 4, 1, 0, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh())
    return DeterministicConditional(mapping)

def create_critic():
    x_mapping = nn.Sequential(
        Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
        Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
        Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
        Conv2d(DIM * 4, DIM * 4, 4, 2, 0), LeakyReLU(LEAK))

    z_mapping = nn.Sequential(
        Conv2d(NLAT, 512, 1, 1, 0), LeakyReLU(LEAK),
        Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK))

    joint_mapping = nn.Sequential(
        Conv2d(DIM * 4 + 512, 1024, 1, 1, 0),  LeakyReLU(LEAK),
        Conv2d(1024, 1024, 1, 1, 0), LeakyReLU(LEAK),
        Conv2d(1024, 1, 1, 1, 0))

    return JointCritic(x_mapping, z_mapping, joint_mapping)


def create_WALI():
    E = create_encoder()
    G = create_generator()
    C = create_critic()
    wali = WALI(E, G, C)
    return wali

def main():
    start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    wali = create_WALI().to(device)

    optimizerEG = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()), 
        lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizerC = Adam(wali.get_critic_parameters(), 
        lr=LEARNING_RATE, betas=(BETA1, BETA2))

    EG_scheduler = StepLR(optimizerEG, step_size=5, gamma=0.5)
    C_scheduler = StepLR(optimizerC, step_size=5, gamma=0.5)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) 
    
    dataset = datasets.ImageFolder('BOT-IOT/Images/Train', transform=transform)
    
    loader = data.DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=16, drop_last=True)
    noise = torch.randn(64, NLAT, 1, 1, device=device)

    EG_losses, C_losses = [], []
    curr_iter = C_iter = EG_iter = 0
    C_update, EG_update = True, False

    print('Training starts...')

    for epoch in range(ITER): 
        print("--------------------------")
        print("======== EPOCH %d ========" % (epoch+1))
        print("--------------------------")
        for batch_idx, (x, _) in enumerate(loader, 1):
            x = x.to(device)

            if curr_iter == 0:
                init_x = x
                curr_iter += 1

            z = torch.randn(x.size(0), NLAT, 1, 1).to(device)
            C_loss, EG_loss = wali(x, z, lamb=LAMBDA)

            if C_update:
                optimizerC.zero_grad()
                C_loss.backward()
                C_losses.append(C_loss.item())
                optimizerC.step()
                C_iter += 1

                if C_iter == C_ITERS:
                    C_iter = 0
                    C_update, EG_update = False, True
                continue

            if EG_update:
                optimizerEG.zero_grad()
                EG_loss.backward()
                EG_losses.append(EG_loss.item())
                optimizerEG.step()
                EG_iter += 1

                if EG_iter == EG_ITERS:
                    EG_iter = 0
                    C_update, EG_update = True, False
                    curr_iter += 1
                else:
                    continue

            # print training statistics
            if curr_iter % 50 == 0:
                print("--- %s seconds ---" % (time.time() - start_time))
                print('[%d]\tW-distance: %.4f\tC-loss: %.4f'
                  % (curr_iter, EG_loss.item(), C_loss.item()))

        if (epoch+1) % (ITER // 5) == 0:
            print("-------------------------------")
            print("======== SAVING MODELS ========")
            print("-------------------------------")
            if not os.path.exists('Models/bigan/'):
                os.makedirs('Models/bigan/')
            torch.save(wali.state_dict(), 'Models/bigan/bot%d.ckpt' % (epoch+1))
        
        EG_scheduler.step()
        C_scheduler.step()        
            
    return EG_losses, C_losses
    
BATCH_SIZE = 1024
ITER = 50
NUM_CHANNELS = 3
DIM = 16 
NLAT = 100
LEAK = 0.1

C_ITERS = 5       # critic iterations
EG_ITERS = 1      # encoder / generator iterations
LAMBDA = 20       # strength of gradient penalty
LEARNING_RATE = 1e-4
BETA1 = 0.5
BETA2 = 0.9

if __name__ == "__main__":
    main()