import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.nn.parameter import Parameter
from geomloss import SamplesLoss

class PatchAttentionLayer(nn.Module):
  #ic:input_channel oc:output_channel ks:kernel_size s:stride
  def __init__(self, ic, oc, ks=1, s=1):
    super().__init__()
    self.convQ = nn.Conv2d(ic,oc,ks,s)
    self.convK = nn.Conv2d(ic,oc,ks,s)
    self.convV = nn.Conv2d(ic,oc,ks,s)
    self.normQ = nn.BatchNorm2d(oc)
    self.normK = nn.BatchNorm2d(oc)
    self.normV = nn.BatchNorm2d(oc)
  def forward(self, x):
    b,c,h,w = x.size()
    self.dim = h**-0.5
    Q = self.normQ(self.convQ(x))
    K = self.normK(self.convK(x))
    V = self.normV(self.convV(x))
    patchQ = F.unfold(Q,(1,1))
    patchK = F.unfold(K,(1,1))
    patchV = F.unfold(V,(1,1))

    pixelMap = torch.einsum('bcq,bck->bqk', patchQ, patchK)*self.dim
    patchOutput = torch.einsum('bqk,bck->bcq', pixelMap, patchV)
    """
    print('pixelMap', pixelMap.size())
    print('patchV', patchV.size())
    print('patchOutput size', patchOutput.size())
    """
    return patchOutput.view(b,c,h,w)

class ChannelAttentionLayer(nn.Module):
  #ic:input_channel oc:output_channel ks:kernel_size s:stride
  def __init__(self, ic, oc, ks, s, blocktype):
    super().__init__()
    if blocktype == 'UA':
      self.convQ = nn.ConvTranspose2d(ic,oc,ks,s,1,output_padding=1)
      self.convK = nn.Conv2d(ic,oc,1,1,0)
      self.convV = nn.Conv2d(ic,oc,1,1,0)
    else:
      self.convQ = nn.Conv2d(ic,oc,ks,s,1)
      self.convK = nn.Conv2d(ic,oc,1,1,1)
      self.convV = nn.Conv2d(ic,oc,1,1,1)
    self.normQ = nn.BatchNorm2d(oc)
    self.normK = nn.BatchNorm2d(oc)
    self.normV = nn.BatchNorm2d(oc)

  def forward(self, x):
    b,c,h,w = x.size()
    Q = self.normQ(self.convQ(x))
    K = self.normK(self.convK(x))
    V = self.normV(self.convV(x))
    outputC, outputH, outputW = Q.size()[1:]
    channelQ = F.unfold(Q,(1,1))
    channelK = F.unfold(K,(1,1))
    channelV = F.unfold(V,(1,1))

    channelMap = F.softmax(torch.einsum('bck,bcq->bkq',channelK, channelQ),dim=1)
    channelOutput = torch.einsum('bck,bkq->bcq',channelV, channelMap)

    """
    print('channelK', channelK.size())
    print('channelQ', channelQ.size())
    print('channelMap', channelMap.size())
    print('channelOutput', channelOutput.size())
    """
    return channelOutput.view(b,outputC,outputH,outputW)

class MultiScaleAttentionBlock(nn.Module):
  def __init__(self, ic, oc, blocktype):
    super().__init__()
    if blocktype=='DA':
      self.ks = 3
      self.s = 2
    if blocktype=='SA':
      self.ks = 3
      self.s = 1
    if blocktype=='UA':
      self.ks = 3
      self.s = 2
    self.PAL = PatchAttentionLayer(ic,ic)
    self.CAL = ChannelAttentionLayer(ic, oc, self.ks, self.s, blocktype)

  def forward(self, x):
    x = self.PAL(x)
    x = self.CAL(x)
    return x

class HashCodingLayer(nn.Module):
  def __init__(self, memory_size, feature_size):
    super().__init__()
    self.memory_size = memory_size
    self.feature_size = feature_size
    self.hash_size = 128
    self.memory = Parameter(torch.Tensor(self.memory_size, self.feature_size))
    stdv = 1. / math.sqrt(self.memory.size(1))
    torch.nn.init.uniform_(self.memory,-stdv,stdv)
    self.hash_layer = nn.Linear(self.feature_size, self.hash_size)

  def forward(self, feature):
    b,c,h,w = feature.size()
    feature = feature.view(b,c*h*w)
    hashed_memory = 0.5*(torch.sign(self.hash_layer(self.memory)-0.5)+1)
    hashed_feature = 0.5*(torch.sign(self.hash_layer(feature)-0.5)+1)
    HD_batch = []
    for i in range(hashed_feature.size()[0]):
      HD_batch.append(torch.abs((hashed_memory - hashed_feature[i])).sum(dim=1))
    HD = torch.stack(HD_batch)
    #HD_weight = F.softmax(HD,dim=1)
    HD_indice = torch.argmin(HD, dim=1)
    #print(HD_min)
    recon_feature = []
    for i in HD_indice:
        feature = self.memory[i]
        recon_feature.append(feature)
    recon_feature = torch.stack(recon_feature)
    #HD weight coefficient times each element in memory to reconstruct x
    #recon_feature = torch.einsum('bm,mf->bf', HD_weight, self.memory).view(b,c,h,w)
    """
    print('HD weight', HD_weight.size())
    print('recon_feature', recon_feature.size())
    print('hashed memory',hashed_memory.size())
    print('hashed feature',hashed_feature.size())
    """
    return recon_feature.view(b,c,h,w)
    #Flatten 4d(NCHW) input into 2d(NF) may not be the best choice.....

class MAMANet(nn.Module):
  def __init__(self):
    super().__init__()
    self.DA1 = MultiScaleAttentionBlock(3,32,'DA')
    self.DA2 = MultiScaleAttentionBlock(32,64,'DA')
    self.DA3 = MultiScaleAttentionBlock(64,128,'DA')
    self.semantic_SA = MultiScaleAttentionBlock(128,128,'SA')
    self.reduce_SA1 = MultiScaleAttentionBlock(256,128,'SA')
    self.reduce_SA2 = MultiScaleAttentionBlock(128,64,'SA')
    self.semantic_UA = MultiScaleAttentionBlock(128,64,'UA')
    self.UA1 = MultiScaleAttentionBlock(128,64,'UA')
    self.UA2 = MultiScaleAttentionBlock(64,32,'UA')
    #In order to fix the mismatch of output shape, the third UA block is a must add here
    self.UA3 = MultiScaleAttentionBlock(32,3,'UA')
    self.HSL = HashCodingLayer(memory_size=2000, feature_size=128*5*5)

  def forward(self, x):
    input_x = x
    #print(input_x.size())
    x = self.DA1(x)
    #print(x.size())
    x = self.DA2(x)
    #print(x.size())
    x = self.DA3(x)
    #print(x.size())
    semantic_info = self.semantic_SA(x)
    x = self.HSL(x)
    #print(x.size())
    x = torch.cat((x, semantic_info), dim=1)
    #print(x.size())
    x = self.reduce_SA1(x)
    #print(x.size())
    x = self.UA1(x)
    #print(x.size())
    semantic_info = self.semantic_UA(semantic_info)
    x = torch.cat((x,semantic_info), dim=1)
    #print(x.size())
    x = self.reduce_SA2(x)
    #print(x.size())
    x = self.UA2(x)
    #print(x.size())
    x = self.UA3(x)
    #print(x.size())
    #loss = F.mse_loss(input_x, x)
    #print(loss)

    return x


def imshow(img1,img2, epoch, i):
  show_img = [img1,img2]
  for i in range(len(show_img)):
    show_img[i] = show_img[i]/2+0.5 #unnormalize
    show_img[i] = show_img[i].cpu().numpy()
  plt.figure()
  f, plot = plt.subplots(3,2)
  plot[0,0].imshow(np.transpose(show_img[0][0],(1,2,0)))
  plot[0,1].imshow(np.transpose(show_img[1][0],(1,2,0)))
  plot[1,0].imshow(np.transpose(show_img[0][1],(1,2,0)))
  plot[1,1].imshow(np.transpose(show_img[1][1],(1,2,0)))
  plot[2,0].imshow(np.transpose(show_img[0][2],(1,2,0)))
  plot[2,1].imshow(np.transpose(show_img[1][2],(1,2,0)))
  #plt.imshow(np.transpose(npimg,(1,2,0)))
  plt.savefig('./results/'+str(epoch)+'_.png')
  plt.close(f)
  #plt.show()


if __name__ == '__main__':
  transforms = transforms.Compose(
      [transforms.Resize((40,40)),
        transforms.ToTensor(),
       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
  trainset = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=transforms)
  trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True,num_workers=1)
  minibatch = 1020%16

  if torch.cuda.is_available():
    device = torch.device('cuda:0')
  else:
    device = torch.device('cpu')
  model = MAMANet()
  model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=10e-5)

  for epoch in range(10000):
    running_loss = 0.0
    lemd = 0.0
    lmse = 0.0
    for i, data in enumerate(trainloader,0):
      inputs, labels = data
      b,c,h,w = inputs.size()
      inputs = inputs.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss_mse = F.mse_loss(inputs, outputs)
      emd_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
      loss_emd = (emd_loss(inputs.view(b,c,-1), outputs.view(b,c,-1))).mean()/1000
      lemd += loss_emd.item()
      lmse += loss_mse.item()
      loss = loss_mse + loss_emd
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

      if i== minibatch:
        print(f'Epoch:[{epoch + 1}] loss: {running_loss / minibatch:.3f} -- mse: {lmse/minibatch:.3f} -- emd: {lemd/minibatch:.3f}')
        imshow(inputs[0:3], outputs[0:3].detach(),epoch+1, i)
        running_loss = 0.0
# Earth Mover's Distance...........
