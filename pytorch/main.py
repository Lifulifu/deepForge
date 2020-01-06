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

def target_generator(batch_size):
    valid_real = Variable(torch.cat((torch.ones((batch_size, 1)),torch.zeros((batch_size, 1))) ,dim=1), requires_grad = False) #(1,0)
    valid_num = Variable(torch.cat((torch.zeros((batch_size, 1)),torch.ones((batch_size, 1))) ,dim=1), requires_grad = False) #(0,1)
    valid = Variable(torch.ones((batch_size, 2)), requires_grad = False) #(1,1)
    fake = Variable(torch.zeros((batch_size, 2)), requires_grad = False) #(0,0)
    return valid_real, valid_num, valid, fake

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
        self.linear = nn.Linear(64*4*4+10, 64*4*4) #(concat 10-one-hot map to normal shape)
        self.bottomB = nn.Sequential(nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64)) #(64,4,4)
        self.up3 = decoder_block(128,32) 
        self.up2 = decoder_block(64,16) 
        self.up1 = decoder_block(32,1)

    def forward(self, x, c):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        h = self.bottomA(x3)
        h = h.view(-1,64*4*4) 
        h = torch.cat((h,c), dim=1)
        h = self.linear(F.relu(h))
        h = h.view(-1,64,4,4)
        h = self.bottomB(F.relu(h))
        h = self.up3(h,x3)
        h = self.up2(h,x2)
        h = self.up1(h,x1)

        h = nn.Sigmoid()(h)
        out = F.relu(h) + x
        out = out.clamp(max=1,min=0)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = encoder_block(1,16) #(16,16,16)
        self.down2 = encoder_block(16,32) #(32,8,8)
        self.down3 = encoder_block(32,4) #(4,4,4)
        self.linear1 = nn.Linear(4*4*4 + 10, 16) #concat with the one-hot condition
        self.linear2 = nn.Linear(16,4)
        self.linear3 = nn.Linear(4,2) #(fake?, rightnumber?)
    def forward(self, x, c):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = x.view(-1,4*4*4)
        x = torch.cat((x,c),dim=1)
        x = self.linear1(F.relu(x)) 
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))
        x = nn.Sigmoid()(x) 
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
                            train_D_iters=1, train_G_iters=1, lr=0.0001, img_dir='./imgs', model_dir='./models'):

    imgs, digits, test_img, test_digits = load_mnist()
    dataset = Dataset(imgs, digits)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset = Dataset(test_img, test_digits)
    test_loader = DataLoader(dataset,batch_size=batch_size, shuffle=False)
    """
    valid_real = Variable(torch.cat((torch.ones((batch_size, 1)),torch.zeros((batch_size, 1))) ,dim=1), requires_grad = False) #(1,0)
    valid_num = Variable(torch.cat((torch.zeros((batch_size, 1)),torch.ones((batch_size, 1))) ,dim=1), requires_grad = False) #(0,1)
    valid = Variable(torch.ones((batch_size, 2)), requires_grad = False) #(1,1)
    fake = Variable(torch.zeros((batch_size, 2)), requires_grad = False) #(0,0)"""

    generator = Generator()
    generator = generator.float()
    discriminator = Discriminator()
    discriminator = discriminator.float()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    
    print("start training dis")
    for iters in range(iterations):
        #======#
        #  Dis #
        #======#
        for _ in range(train_D_iters):
            discriminator.train()
            generator.eval()
            D_loss = 0
            for step, (batch_x, batch_y) in enumerate(loader):
                batch_x = torch.transpose(batch_x,1,3)
                batch_x = torch.transpose(batch_x,2,3)
                batch_y = batch_y
                match_c = onehot(batch_y, 10)
                unmatch_c = onehot(batch_y, 10, exclusive=True)
                fake_x = generator(batch_x.float(), unmatch_c.float())
                real_x = batch_x
                valid_real, valid_num, valid, fake = target_generator(batch_x.size(0))
                
                optimizer_D.zero_grad()

                real_match_pred = discriminator(real_x.float(), match_c.float())
                real_unmatch_pred = discriminator(real_x.float(),unmatch_c.float())
                fake_match_pred = discriminator(fake_x.float(), unmatch_c.float())
                fake_unmatch_pred = discriminator(fake_x.float(), match_c.float())
                loss1 = loss_fn(real_match_pred, valid)
                loss2 = loss_fn(real_unmatch_pred, valid_real)
                loss3 = loss_fn(fake_match_pred, valid_num)
                loss4 = loss_fn(fake_unmatch_pred, fake)
                loss = loss1 + loss2 + loss3 + loss4
                loss.backward()
                optimizer_D.step()

                D_loss += loss.item()

        D_loss /= (len(loader) * train_D_iters)
        print(f"iter: {iters} | D_loss: {D_loss}")

            
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
                valid_real, valid_num, valid, fake = target_generator(batch_x.size(0))

                fake_x = generator(batch_x.float(), unmatch_c.float())
                pred = discriminator(fake_x.float(), unmatch_c.float())
                loss = loss_fn(pred, valid)
                loss.backward()
                optimizer_G.step()

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
                test_x = generator(batch_x.float(), unmatch_c.float())
                n = random.randint(0,batch_x.size(0)-1)
                pick_list.append([batch_x[n].detach().numpy(), unmatch_digits[n].detach().numpy(), test_x[n].detach().numpy()])
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
        sample_interval=5, 
        save_model_interval=100,
        train_D_iters=1, 
        train_G_iters=5, 
        lr=0.0001, 
        img_dir='./imgs', 
        model_dir='./models'
    )






        