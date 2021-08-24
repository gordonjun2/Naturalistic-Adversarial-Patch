# plot norm
#%%
import torch 
from matplotlib import pyplot as plt
import numpy as np

#%%
epoch=1000
checkpoint_dir           = "../adversarial-attack-ensemble/exp/exp91"
PATH   = str(checkpoint_dir) + "/checkpoint/gan_params_"+str(epoch)+".pt"

#%%
latent_shift_biggan = torch.load(PATH)['latent_shift_biggan']
torch.max(torch.abs(latent_shift_biggan))
torch.norm(latent_shift_biggan, p=1)/latent_shift_biggan.shape[0]

#%%
checkpoint_dir           = "../adversarial-attack-ensemble/exp/exp90"
epoch_want = [10,30,50,100,150,200,300,400,500,600,700]
Linf = []
L1=[]
L2=[]
for i in epoch_want:
    PATH = str(checkpoint_dir) + "/checkpoint/gan_params_"+str(i)+".pt"
    latent_shift_biggan = torch.load(PATH)['latent_shift_biggan']
    Linf.append(torch.max(torch.abs(latent_shift_biggan)))
    L1.append(torch.norm(latent_shift_biggan, p=1)/latent_shift_biggan.shape[0])
    L2.append(torch.norm(latent_shift_biggan, p=2)/latent_shift_biggan.shape[0])

#%%
plt.plot(epoch_want,Linf,'--o')
plt.plot(epoch_want,L1,'--o')
plt.plot(epoch_want,L2,'--o')
plt.legend()