import warnings

warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.add_noise import add_noise
from mydataset.dataset_synapse import normalization

model = ResUnetPlusPlus(1)
checkpoint = torch.load(r'',map_location=torch.device('cpu'))

# 创建一个新的字典，只包含模型中存在的键值对
filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model.state_dict()}
model.load_state_dict(filtered_state_dict)
model.eval()

input_ = np.load(r'F:\实验\地震图像去噪\地震去噪_无监督\resunet++_bilateralfilter\input_.npy')
img_input_ = input_
# input_ = input_[256:384,150:278]
# input_ = input_[0:128,384-128:384]

input_ = torch.from_numpy(input_)
input_ = torch.unsqueeze(input_, 0)
input_ = torch.unsqueeze(input_, 0)
input_ = normalization(input_)

input_ = add_noise(input_,scale_ratio=0.05)
with torch.no_grad():
    output = model(input_.float())
output = torch.squeeze(output)
output = output.detach().cpu().numpy()
input_ = torch.squeeze(input_)
input_= input_.detach().cpu().numpy()
# np.save(r'F:\实验\地震图像去噪\地震去噪_无监督\resunet++_bilateralfilter\output_test_ours.npy',output)

plt.imshow(input_, cmap='gray')
plt.tight_layout()
plt.axis('off')
#plt.savefig(r'F:\实验\地震图像去噪\地震去噪_无监督\resunet++_bilateralfilter\input_add_gaussion0.3.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.tight_layout()

plt.imshow(output, cmap='gray')
plt.tight_layout()
plt.axis('off')
plt.savefig(r'F:\实验\地震图像去噪\地震去噪_无监督\resunet++_bilateralfilter\output_ours_hechengt0.05.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.tight_layout()

