import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from utils.hparams import HParam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import metrics
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
from utils.losses import CharbonnierLoss
from core.res_unet_plus import ResUnetPlusPlus
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import argparse
import os
import numpy as np
import random
from mydataset.dataset_synapse import MyDataset
from utils.generate_mask import generate_mask_pair, generate_subimages
from utils.add_noise import add_noise
from utils.myMetric import calculate_psnr
import torch.nn.functional as F
from utils.my_ssim import SSIM_LOSS
from utils.filters import bilateral_filter
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='..', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train_npz')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train_npz')
parser.add_argument('--batch_size', type=int,
                    default=64, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=128, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--nepoch', type=int,
                    default=250, help='vit_patches_size, default is 16')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument("--increase_ratio", type=float, default=2.0)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
model_dir = ''
# get model
model = ResUnetPlusPlus(1).cuda()
model.train().cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# dataset amd dataloader
train_dataset = MyDataset(r'')
val_dataset = MyDataset(r'')

# 加载并乱序训练数据集
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=0,
    drop_last=True,
    shuffle=True
)
# 加载测试数据集，测试数据不需要乱序
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    drop_last=True,
    # shuffle=False
)
writer = SummaryWriter('')
criterion = CharbonnierLoss()
# decay LR
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion_simm = SSIM_LOSS().cuda()
# starting params
best_loss = 999
start_epoch = 0
# optionally resume from a checkpoint

step = 0
iter_count_img = 0
iter_count_loss_ones = 0
iter_count_param = 0
iter_count_psnr = 0
for epoch in range(start_epoch, 1000):
    epoch_loss = 0
    model.train()
    print(str(epoch)+'start!!!!!!!!!!!!')
    for idx, data in enumerate(tqdm(train_loader), 0):
        input_ = data[0].cuda()
        if epoch>=0:
            input_ = add_noise(input_,epoch)
        mask1, mask2 = generate_mask_pair(input_)
        sub1 = generate_subimages(input_, mask1)
        sub2 = generate_subimages(input_, mask2)
        with torch.no_grad():
            denoised_img = model(input_.float())
        sub1_denoised = generate_subimages(denoised_img, mask1)
        sub2_denoised = generate_subimages(denoised_img, mask2)

        denoised_sub1 = model(sub1.float())
        target = sub2

        Lambda = epoch / args.n_epoch * args.increase_ratio
        diff = denoised_sub1 - target
        exp_diff = sub1_denoised - sub2_denoised

        # input_poll = F.avg_pool2d(input_, 2, 2)
        # simm_loss = criterion_simm(input_poll.float(), denoised_sub1.float())
        # 双边滤波
        output_bilateral = bilateral_filter(input_.cpu().numpy())
        output_bilateral = torch.from_numpy(output_bilateral).cuda()
        output_bilateral_downsample = F.avg_pool2d(output_bilateral,2,2)
        simm_loss = criterion_simm(output_bilateral_downsample.float(), denoised_sub1.float())

        loss1 = torch.mean(diff**2)
        # loss2 = Lambda * torch.mean((diff-exp_diff)**2)
        loss3 = Lambda * simm_loss
        loss_all = args.Lambda1 * loss1 + args.Lambda2 * loss3
        writer.add_scalar('loss_ones',loss_all.item()/data[0].shape[0],iter_count_loss_ones)
        iter_count_loss_ones+=1
        epoch_loss+=loss_all.item()
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
    lr_scheduler.step()
    writer.add_scalar('loss_epoch',epoch_loss,epoch)
    print(epoch_loss)
    ##evalution##
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(name + '_grad', param.grad, iter_count_param)
    for name, param in model.named_parameters():
        writer.add_histogram(name + '_data', param, iter_count_param)
    iter_count_param+=1
    model.eval()
    for idx, data in enumerate(val_loader):
        idx_save_img = 0
        with torch.no_grad():
            label_path = data[1]
            input_ = data[0].cuda()
            output = model(input_.float())
            idx_save_img = random.randint(0, input_.shape[0] - 1)
            save_noise = input_[idx_save_img:idx_save_img + 1, :, :, :]
            save_clean = output[idx_save_img:idx_save_img + 1, :, :, :]
            diff_img = save_clean - save_noise
            diff_img = (diff_img - torch.min(diff_img)) / (torch.max(diff_img) - torch.min(diff_img))
            img_save = torch.cat((save_noise, save_clean, diff_img))
            writer.add_images('Adele_denoise', img_save, iter_count_img)
            writer.add_text('save_path', label_path[idx_save_img])
        with torch.no_grad():
            clean = input_
            input_ = add_noise(input_)
            output = model(input_.float())
            psnr = calculate_psnr(clean, output)
            writer.add_scalar('psnr', psnr, iter_count_psnr)
            iter_count_psnr += 1
            save_noise = input_[idx_save_img:idx_save_img + 1, :, :, :]
            save_clean = output[idx_save_img:idx_save_img + 1, :, :, :]
            diff_img = save_clean - save_noise
            diff_img = (diff_img - torch.min(diff_img)) / (torch.max(diff_img) - torch.min(diff_img))
            img_save = torch.cat((save_noise, save_clean, diff_img))
            writer.add_images('Adele_add_noise_denoise', img_save, iter_count_img)
            iter_count_img += 1
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))