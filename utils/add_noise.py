import random
import numpy as np
import torch

def add_noise(img,epoch = 0,scale_ratio = 0.2 ):
    # scale = random.random()*scale_ratio
    scale = scale_ratio
    noise = np.random.normal(loc=0.0, scale=scale, size=img.shape)
    noise = torch.from_numpy(noise).to(img.device)
    epoch = torch.tensor(epoch)
    noise = noise * torch.exp(-epoch)
    noise_img = img+noise
    return noise_img