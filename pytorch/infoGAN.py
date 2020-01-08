from model import FrontEnd, D, Q, G, E
from util import load_mnist, onehot, reverse_onehot
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
                            train_D_iters=1, train_G_iters=3, D_lr=0.0001,  G_lr=0.0001, img_dir='./info_imgs', model_dir='./models'):

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
    generaotor = generaotor.to(device)
    discriminator = discriminator.to(device)
    qq = qq.to(device)
    encoder = encoder.to(device)
    front = front.to(device)

    opt_G = torch.optim.Adam([
        {"params":generaotor.parameters()},
        {"params":qq.parameters()},
        {"params":encoder.parameters()}], lr=G_lr)

    opt_D = torch.optim.Adam([
        {"params":discriminator.parameters()},
        {"params":front.parameters()}], lr=D_lr)

    lossfnD = nn.BCELoss().to(device)
    lossfnQ = nn.CrossEntropyLoss().to(device)
    lossfnQ_v = nn.CosineEmbeddingLoss().to(device)

    real_x = torch.FloatTensor(batch_size, 1, 32, 32).to(device)
    label = torch.FloatTensor(batch_size, 1).to(device)
    noise = torch.FloatTensor(batch_size, 54).to(device)
    c = torch.FloatTensor(batch_size, 10).to(device)
    v_target = torch.LongTensor(batch_size,64).to(device)
    
    real_x = Variable(real_x)
    label = Variable(label, requires_grad=False)
    noise = Variable(noise)
    c = Variable(c)
    v_target = Variable(v_target, requires_grad=False)

    idx = np.arange(10).repeat(10)
    one_hot = np.zeros((batch_size, 10)) 
    one_hot[range(batch_size), idx] = 1 # 0-9 0-9...
    fix_noise = torch.Tensor(100, 54).uniform_(-1, 1)

    for epoch in range(iterations):
        for step, [batch_x, batch_target] in enumerate(loader):

            #print("transpose")
            bs = batch_x.size(0)

        # train D
        #==========
            # real
            #print("train d real")
            opt_D.zero_grad()
            real_x.data.copy_(batch_x)
            label.data.fill_(1)

            fe1 = front(real_x)
            real_pred = discriminator(fe1)
            real_loss = lossfnD(real_pred,label)
            real_loss.backward()

            #fake
            #print("train d fake")
            real_x.data.copy_(batch_x)

            v = encoder(real_x)
            z, idx = noise_sample(c, noise, v, bs)

            fake_stroke = generaotor(z)
            fake_x = fake_stroke + real_x
            fake_x = fake_x.clamp(max=1,min=0)



            fe2 = front(fake_x.detach())
            fake_pred = discriminator(fe2)
            label.data.fill_(0)
            fake_loss = lossfnD(fake_pred, label)
            fake_loss.backward()

            D_loss = real_loss + fake_loss

            opt_D.step()
        
        # train G, Q, E
        #===============
            #print("train G")
            opt_G.zero_grad()
            fe = front(fake_x)
            fake_pred = discriminator(fe)
            label.data.fill_(1.0)

            reconstruct_loss = lossfnD(fake_pred, label)
            
            #print("train Q")
            c_out, v_out = qq(fe)

            class_ = torch.LongTensor(idx).to(device)
            target = Variable(class_)

            v_target.data.copy_(v)

            q_c_loss = lossfnQ(c_out, target)
            q_v_loss = lossfnQ_v(v_out, v_target, label.data.fill_(1))

            q_loss = q_c_loss + q_v_loss

            G_loss = reconstruct_loss * 0.001  + q_c_loss + q_v_loss
            G_loss.backward()

            opt_G.step()



        print(f'Epoch: {epoch} | Dloss: {D_loss.data.cpu().numpy()} | qloss: {q_loss.data.cpu().numpy()} | reloss: {reconstruct_loss.data.cpu().numpy()}')
        save_image(fake_x.data, f'./{img_dir}/{epoch}_F.png', nrow=10)
        print(f"Qpred {np.argmax(c_out[1].data.cpu().numpy())}")
        print(f"Origin {batch_target[1].data.cpu().numpy()} ToBe: {idx[0]}")
        #save_image(real_x.data, f'./{img_dir}/{epoch}_R.png', nrow=10)

        """
        if epoch % 3 == 0:

            noise.data.copy_(fix_noise)
            c.data.copy_(torch.Tensor(one_hot))

            z = torch.cat([noise, c, v], 1).view(-1, 128, 1, 1)
            x_save = generaotor(z)

            if not os.path.isdir(img_dir):
                os.makedirs(img_dir)

            save_image(x_save.data, f'./{img_dir}/{epoch}.png', nrow=10)
        """
if __name__ == "__main__":
    train( 
        D_lr=0.0001,
        G_lr=0.0001, 
        img_dir='./info_imgs', 
        model_dir='./models')
        
          

            
            

    
    
    
