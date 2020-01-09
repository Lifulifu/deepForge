from model import FrontEnd, D, Q, G, E, weights_init
from util import load_mnist
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader as DataLoader
import random
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
import numpy as np

def noise_sample(C, noise, v, bs):

    idx = np.random.randint(10, size=bs)
    c = np.zeros((bs, 10))
    c[range(bs),idx] = 1.0

    C.data.copy_(torch.Tensor(c))
    noise.data.uniform_(-1.0, 1.0)
    z = torch.cat([noise, C, v], 1).view(-1, 128, 1, 1)

    return z, idx


class Dataset():
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.len = data.shape[0]
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return self.len

def train(iterations=10000, batch_size=100, sample_interval=5, save_model_interval=100,
                            train_D_iters=1, train_G_iters=3, D_lr=0.0001,  G_lr=0.0001, betas=(0.5, 0.99), img_dir='./info_imgs', model_dir='./models'):

    imgs, digits, test_img, test_digits = load_mnist()
    dataset = Dataset(imgs, digits)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #dataset = Dataset(test_img, test_digits)
    #test_loader = DataLoader(dataset,batch_size=batch_size, shuffle=False)
    
    if torch.cuda.is_available:
        print(f"Using GPU {torch.cuda.current_device()}")
        device = "cuda"
    else: 
        print("Using CPU...")
        device = "cpu"

    generaotor, discriminator, front, qq, encoder = G(), D(), FrontEnd(), Q(), E()
    generaotor = generaotor.to(device).apply(weights_init)
    discriminator = discriminator.to(device).apply(weights_init)
    qq = qq.to(device).apply(weights_init)
    encoder = encoder.to(device).apply(weights_init)
    front = front.to(device).apply(weights_init)

    opt_G = torch.optim.Adam([
        {"params":generaotor.parameters()},
        {"params":qq.parameters()},
        {"params":encoder.parameters()}], lr=G_lr, betas=betas)

    opt_D = torch.optim.Adam([
        {"params":discriminator.parameters()},
        {"params":front.parameters()}], lr=D_lr, betas=betas)

    CELoss_D = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1,1,1,1,1,1,1])).to(device)
    CELoss_G = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1,1,1,1,1,1,20])).to(device)
    CELoss_Q = nn.CrossEntropyLoss().to(device)
    CosineLoss = nn.CosineEmbeddingLoss().to(device)

    real_x = torch.FloatTensor(batch_size, 1, 32, 32).to(device)
    trg = torch.LongTensor(batch_size).to(device)
    label = torch.FloatTensor(batch_size, 1).to(device)
    noise = torch.FloatTensor(batch_size, 54).to(device)
    c = torch.FloatTensor(batch_size, 10).to(device)
    v_target = torch.LongTensor(batch_size,64).to(device) # For Q
    
    real_x = Variable(real_x)
    noise = Variable(noise)
    c = Variable(c)
    trg = Variable(trg, requires_grad=False)
    label = Variable(label, requires_grad=False)
    v_target = Variable(v_target, requires_grad=False)


    for epoch in range(iterations):
        for step, [batch_x, batch_target] in enumerate(loader):

            bs = batch_x.size(0)

        # train D
        #==========
            # real
            opt_D.zero_grad()
            
            real_x.data.copy_(batch_x)
            trg.data.copy_(batch_target)

            fe1 = front(real_x)
            real_pred = discriminator(fe1)

            real_loss = CELoss_D(real_pred, trg)
            real_loss.backward()

            #fake
            real_x.data.copy_(batch_x)

            v = encoder(real_x)
            z, idx = noise_sample(c, noise, v, bs)

            fake_stroke = generaotor(z)
            fake_x = fake_stroke + real_x
            fake_x = fake_x.clamp(max=1,min=0) 

            fe2 = front(fake_x.detach())
            fake_pred = discriminator(fe2)
        
            trg.data.fill_(10)
            if epoch > 0:
                ignore_rate = 0.01
            else:
                ignore_rate = 1
                
            fake_loss = CELoss_D(fake_pred, trg) * ignore_rate
            fake_loss.backward()
            D_loss = real_loss + fake_loss 
            #D_loss.backward()
            opt_D.step()
        
        # train G, Q, E
        #===============
            #train G
            opt_G.zero_grad()
            fe = front(fake_x)
            fake_pred = discriminator(fe)


            trg.data.copy_(torch.LongTensor(idx))
            reconstruct_loss = CELoss_G(fake_pred, trg)

            # train Q
            c_out, v_out = qq(fe)

            class_ = torch.LongTensor(idx).to(device)
            target = Variable(class_)

            v_target.data.copy_(v)

            # GQ Loss
            q_c_loss = CELoss_Q(c_out, target)
            q_v_loss = CosineLoss(v_out, v_target, label.data.fill_(1))

            q_loss = q_c_loss + q_v_loss
            
            G_loss = reconstruct_loss + q_c_loss + q_v_loss 
            G_loss.backward()
            opt_G.step()

            # accuracy 


        print(f'Epoch: {epoch} | Dloss: {D_loss.data.cpu().numpy()} | QCloss: {q_c_loss.data.cpu().numpy()} | QVloss: {q_v_loss.data.cpu().numpy()} | reloss: {reconstruct_loss.data.cpu().numpy()}')
        save_image(torch.cat((fake_x , fake_stroke, real_x),dim=0).data, f'./{img_dir}/{epoch}.png', nrow=20)
        print(f"fake pred {np.argmax(fake_pred.data.cpu().numpy(),axis=1)}")
        #print(f"Qpred {np.argmax(c_out[1].data.cpu().numpy())}")
        #print(f"Origin {batch_target[1].data.cpu().numpy()} ToBe: {idx[0]}")
        #save_image(real_x.data, f'./{img_dir}/{epoch}_R.png', nrow=10)

        """
        D_lr=0.000001,
        G_lr=0.00005,
        """
if __name__ == "__main__":
    train( 
        D_lr=0.000005,
        G_lr=0.00005,
        betas=(0.6, 0.99),
        img_dir='./info_imgs', 
        model_dir='./models'
        )
        
          

            
            

    
    
    
