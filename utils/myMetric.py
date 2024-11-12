import torch
import torch.nn.functional as F

def calculate_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_value.item()