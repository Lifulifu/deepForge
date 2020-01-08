
import torch.nn as nn
import torch


class FrontEnd(nn.Module):
  ''' front end part of discriminator and Q'''

  def __init__(self):
    super(FrontEnd, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
      nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
      nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
      nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True),
      nn.LeakyReLU(0.2, inplace=True)
    )

  def forward(self, x):
    output = self.main(x)
    #print(output.size())
    return output


class D(nn.Module):

  def __init__(self):
    super(D, self).__init__()
    
    self.main = nn.Sequential(
      nn.Conv2d(512, 1, kernel_size=(2, 2), stride=(1, 1), bias=False),
      #n.LeakyReLU()
      nn.Sigmoid()
      
    )
    

  def forward(self, x):
    output = self.main(x).view(-1, 1)
    return output


class Q(nn.Module):

  def __init__(self):
    super(Q, self).__init__()
    self.front = nn.Sequential(
                  nn.Linear(in_features= 2048, out_features=100, bias=True),
                  nn.LeakyReLU(),
                  )

    self.main_c = nn.Sequential(
            #nn.ReLU(),
            nn.Linear(in_features=100, out_features=10, bias=True),
            nn.Softmax(dim=1)
            )
    self.main_v = nn.Sequential(
            nn.Linear(in_features=100, out_features=64, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64, bias=True)
            ) 
            

  def forward(self, x):
    x = x.view(100,-1)
    f_out = self.front(x)
    c = self.main_c(f_out)
    v = self.main_v(f_out)
    
    return c, v


class G(nn.Module):

  def __init__(self):
    super(G, self).__init__()

    self.main = nn.Sequential(
      nn.ConvTranspose2d(128, 512, kernel_size=(4, 4), stride=(1, 1), bias=False),
      nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True),
      #nn.LeakyReLU(0.02, inplace=True),

      nn.ReLU(True),
      nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),bias=False),
      nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
      #nn.LeakyReLU(0.02, inplace=True),

      nn.ReLU(True),
      nn.ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),bias=False),
      nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
      #nn.LeakyReLU(0.02, inplace=True),



      nn.ReLU(True),
      nn.ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),bias=False),
      nn.Tanh(),

    )

  def forward(self, x):
    output = self.main(x) 
    output = output * 0.5 +0.5
    return output

class E(nn.Module):

    def __init__(self):
        super(E, self).__init__()
        self.frontend = FrontEnd()
        self.main = nn.Sequential(
            nn.Linear(in_features= 2048, out_features=100, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=64, bias=True)
        )
        
    def forward(self,x):
      x = self.frontend(x)
      x = x.view(100,-1)
      output = self.main(x)
      return output 

        

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
"""
    nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),bias=False),
      nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
      #nn.LeakyReLU(0.02, inplace=True),
"""