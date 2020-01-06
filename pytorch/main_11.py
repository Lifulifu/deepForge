import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader as DataLoader
from util import load_mnist, onehot, reverse_onehot
import random
import matplotlib.pyplot as plt
import os

def target_generator(batch_size, device):
    valid_real = Variable(torch.cat((torch.ones((batch_size, 1)),torch.zeros((batch_size, 1))) ,dim=1), requires_grad = False) #(1,0)
    valid_num = Variable(torch.cat((torch.zeros((batch_size, 1)),torch.ones((batch_size, 1))) ,dim=1), requires_grad = False) #(0,1)
    valid = Variable(torch.ones((batch_size, 2)), requires_grad = False) #(1,1)
    fake = Variable(torch.zeros((batch_size, 2)), requires_grad = False) #(0,0)
    return valid_real.to(device), valid_num.to(device), valid.to(device), fake.to(device)

class encoder_block(nn.Module):
    def __init__(self, in_channel, out_channel, bn=True, dp=False, ps=0.25):
        super(encoder_block, self).__init__()
        self.bn, self.dp = bn, dp
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
        self.batchn = nn.BatchNorm2d(out_channel)
        self.dropout = nn.Dropout(ps)
    def forward(self, x):
        x = self.conv(x)
        if self.bn: x = self.batchn(x) 
        if self.dp: x = self.dropout(x)
        x = F.relu(x)
        
        return x

class decoder_block(nn.Module):
    def __init__(self, in_channel, out_channel, bn=True, dp=False, ps=0.25):
        super(decoder_block, self).__init__()
        self.bn, self.dp = bn, dp
        self.dconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batchn = nn.BatchNorm2d(out_channel)
        self.dropout = nn.Dropout(ps)
    def forward(self, x, encode_x):
        x = torch.cat([x, encode_x], axis=1)
        x = self.dconv(x)
        if self.bn: x = self.batchn(x) 
        if self.dp: x = self.dropout(x)
        x = F.relu(x)
        return x
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = encoder_block(1,16) #(16,16,16)
        self.down2 = encoder_block(16,32) #(32,8,8)
        self.down3 = encoder_block(32,64) #(64,4,4)
        self.bottomA = nn.Sequential(nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64)) #(64,4,4)
        self.linear1 = nn.Linear(64*4*4, 16)
        self.linear2 = nn.Linear(16+10+8, 32) #(concat 10-one-hot and latent map to normal shape )
        self.linear3 = nn.Linear(32, 8*4*4)
        self.linear4 = nn.Linear(8*4*4, 64*4*4)
        self.bottomB = nn.Sequential(nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64)) #(64,4,4)
        self.up3 = decoder_block(128,32) 
        self.up2 = decoder_block(64,16) 
        self.up1 = decoder_block(32,1)
        self.Bn1 = nn.BatchNorm1d(16+10+8)
        self.Bn2 = nn.BatchNorm1d(32)
        self.Bn3 = nn.BatchNorm1d(8*4*4)
        self.Bn4 = nn.BatchNorm1d(64*4*4)
    def forward(self, x, c, noise):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        h = self.bottomA(x3)

        h = h.view(-1,64*4*4) 
        h = self.linear1(F.relu(h))
        h = nn.Sigmoid()(h)
        h = torch.cat((h,c), dim=1)
        h = torch.cat((h,noise),dim=1)
        h = self.Bn1(h)
        h = self.linear2(F.relu(h))
        h = self.Bn2(h)
        h = self.linear3(F.relu(h))
        h = self.Bn3(h)
        h = self.linear4(F.relu(h))
        h = self.Bn4(h)
        h = h.view(-1,64,4,4)

        h = self.bottomB(F.relu(h))
        h = self.up3(h,x3)
        h = self.up2(h,x2)
        h = self.up1(h,x1)

        h = nn.Tanh()(h) * 0.5 + 0.5
        out = h + x 
        out = out.clamp(max=1,min=0)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = encoder_block(1,16) #(16,16,16)
        self.down2 = encoder_block(16,64) #(32,8,8)
        self.down3 = encoder_block(64,4) #(4,4,4)
        self.linear0 = nn.Linear(4*4*4, 16)
        self.linear1 = nn.Linear(4*4 + 10, 16) #concat with the one-hot condition
        self.linear2 = nn.Linear(16,16)
        self.linear3 = nn.Linear(16,11) 
        self.Bn1 = nn.BatchNorm1d(26)
        self.Bn2 = nn.BatchNorm1d(16)
        self.Bn3 = nn.BatchNorm1d(11)
    def forward(self, x, c):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = x.view(-1,4*4*4)
        x = self.linear0(F.relu(x))
        x = nn.Sigmoid()(x)
        x = torch.cat((x,c),dim=1)
        x = self.Bn1(x)
        x = self.linear1(F.relu(x))
        x = self.Bn2(x)
        x = self.linear2(F.relu(x))
        x = self.Bn3(x)
        x = self.linear3(F.relu(x))
        x = nn.Softmax(dim=1)(x) 
        return x

class Dataset():
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.len = data.shape[0]
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return self.len
    
def train(iterations=10000, batch_size=128, sample_interval=5, save_model_interval=100,
                            train_D_iters=1, train_G_iters=1, D_lr=0.0001,  G_lr=0.00001, img_dir='./imgs', model_dir='./models'):

    imgs, digits, test_img, test_digits = load_mnist()
    dataset = Dataset(imgs, digits)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset = Dataset(test_img, test_digits)
    test_loader = DataLoader(dataset,batch_size=batch_size, shuffle=False)
    
    if torch.cuda.is_available:
        print(f"Using GPU {torch.cuda.current_device()}")
        device = "cuda"
    else: 
        print("Using CPU...")
        device = "cpu"

    generator = Generator()
    generator = generator.float().to(device)
    discriminator = Discriminator()
    discriminator = discriminator.float().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=G_lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=D_lr)
    loss_fn = nn.BCELoss()
    
    
    for iters in range(iterations):
        #======#
        #  Dis #
        #======#
        for _ in range(train_D_iters):
            discriminator.train()
            generator.eval()
            D_loss = 0
            D_real_acc = torch.tensor([0,0]).float()
            D_fake_acc = torch.tensor([0,0]).float()
            for step, (batch_x, batch_y) in enumerate(loader):
                batch_x = torch.transpose(batch_x,1,3)
                batch_x = torch.transpose(batch_x,2,3)
                match_c = onehot(batch_y, 10)
                unmatch_c = onehot(batch_y, 10, exclusive=True)
                noise = torch.randn(batch_x.size(0),8 ,device=device)

                fake_x = generator(batch_x.float().to(device), unmatch_c.float().to(device), noise)
                real_x = batch_x
                #valid_real, valid_num, valid, fake = target_generator(batch_x.size(0), device)
                fake = torch.zeros(batch_x.size(0),11)
                fake[:,10] = 1
                valid = unmatch_c
                optimizer_D.zero_grad()

                real_pred = discriminator(real_x.float().to(device), match_c.float().to(device))
                fake_pred = discriminator(fake_x.float().to(device), unmatch_c.float().to(device))
                loss1 = loss_fn(real_pred, valid)
                loss2 = loss_fn(fake_pred, fake)
                loss = loss1*0.9 + loss2*0.1
                loss.backward()
                optimizer_D.step()
                D_real_acc += real_match_pred.cpu().detach().mean(dim=0)
                D_fake_acc += fake_unmatch_pred.cpu().detach().mean(dim=0)
                D_loss += loss.item()
        D_real_acc /= (len(loader) * train_D_iters)
        D_fake_acc /= (len(loader) * train_D_iters)
        D_loss /= (len(loader) * train_D_iters)
        print(f"iter: {iters} | D_loss: {D_loss} | real_acc: {D_real_acc.numpy()} | fake_acc: {D_fake_acc.numpy()}")

            
        #======#
        #  Gen #
        #======#
        #print("start training gen")
        for G_iters in range(train_G_iters):
            generator.train()
            discriminator.eval()
            G_loss = 0
            G_acc = torch.tensor([0,0]).float()
            for step, (batch_x, batch_y) in enumerate(loader):
                optimizer_G.zero_grad()
                unmatch_c = onehot(batch_y, 10, exclusive=True)
                batch_x = torch.transpose(batch_x,1,3)
                batch_x = torch.transpose(batch_x,2,3)
                valid_real, valid_num, valid, fake = target_generator(batch_x.size(0), device)
                noise = torch.randn(batch_x.size(0),8 ,device=device)

                fake_x = generator(batch_x.float().to(device), unmatch_c.float().to(device), noise)
                pred = discriminator(fake_x.float().to(device), unmatch_c.float().to(device))
                loss = loss_fn(pred, valid)
                loss.backward()
                optimizer_G.step()
                pred = pred.cpu().detach()
                G_loss += loss.item()           
                G_acc += pred.float().mean(dim=0)
            G_loss /= (len(loader) )
            G_acc /= (len(loader) )
            print(f"iter: {iters}.{G_iters} | G_loss: {G_loss} | G_accuracy: {G_acc.detach().numpy()}")


        if iters % sample_interval == 0 :
            discriminator.eval()
            generator.eval()
            pick_list = []
            for step, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = torch.transpose(batch_x,1,3)
                batch_x = torch.transpose(batch_x,2,3)
                unmatch_c = onehot(batch_y, 10, exclusive=True)
                unmatch_digits = reverse_onehot(unmatch_c)
                noise = torch.randn(batch_x.size(0),8 ,device=device)

                test_x = generator(batch_x.float().to(device), unmatch_c.float().to(device), noise)
                n = random.randint(0,batch_x.size(0)-1)
                pick_list.append([batch_x[n].cpu().detach().numpy(), unmatch_digits[n].cpu().detach().numpy(), test_x[n].cpu().detach().numpy()])
            pick_list = pick_list[:5]
            fig, axs = plt.subplots(5, 2, figsize=(8, 6))
            fig.tight_layout()
            for no, [origin, ans, fake] in enumerate(pick_list):
                axs[no, 0].text(-20, -2, f'Answer: {ans}')
                axs[no, 0].imshow(origin[0, :, :], cmap='gray')
                axs[no, 0].axis('off')
                axs[no, 1].imshow(fake[0, :, :], cmap='gray')
                axs[no, 1].axis('off')
            if not os.path.isdir(img_dir):
                os.makedirs(img_dir)
            fig.savefig(os.path.join(img_dir, f'{iters}.png'))
            plt.close()
                

            

if __name__ == "__main__":
    train(
        iterations=10000, 
        batch_size=128, 
        sample_interval=2, 
        save_model_interval=100,
        train_D_iters=3, 
        train_G_iters=1, 
        D_lr=0.0001, 
        G_lr=0.00001,
        img_dir='./11_imgs', 
        model_dir='./11_models'
    )






        